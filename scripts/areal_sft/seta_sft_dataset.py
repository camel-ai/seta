"""Custom AREAL SFT dataset loader for seta-env trajectories.

Reads SFT data produced by ``seta_env.utils.sft_utils.build_sft_dataset``
and projects each row to the AREAL SFT row format:

    {
        "input_ids": list[int],
        "loss_mask": list[int],   # same length, 1=trainable, 0=context
    }

All other columns (task_id, reward, raw_conv_json, etc.) are dropped — they
are diagnostic metadata for inspection, not training inputs. AREAL's
``pad_sequences_to_tensors`` collator only consumes ``input_ids`` and
``loss_mask``.

Compatible with the AREAL example pattern in
``external/areal/areal/dataset/gsm8k.py::get_gsm8k_sft_dataset`` — same
return shape, same ``max_length`` filter semantics (longer sequences are
*filtered out*, not truncated).

Source resolution
-----------------
``path`` is resolved in this order:

1. **HuggingFace Hub dataset id** (e.g. ``camel-ai/seta-sft-kimi-k2.5``):
   loaded via ``datasets.load_dataset(path, split="train")``.
2. **Local single .jsonl file** produced by ``build_sft_dataset.py``.

The published / built dataset is single-split (``train`` only). The
loader hash-splits the source rows into ``train`` and ``test`` based on
the row index, controlled by ``train_ratio`` (default ``1.0`` — every
row goes to ``train``, ``test`` is empty). Pass ``train_ratio=0.95``
(or any value in ``(0, 1)``) to hold out a deterministic eval slice
without re-publishing the dataset.

Usage from sft_train.py
-----------------------

    from seta_sft_dataset import get_seta_sft_dataset

    train_dataset = get_seta_sft_dataset(
        path=config.train_dataset.path,
        split="train",
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
    )
    valid_dataset = get_seta_sft_dataset(
        path=config.valid_dataset.path,
        split="test",
        tokenizer=tokenizer,
        max_length=config.valid_dataset.max_length,
    )

The returned ``Dataset`` plugs straight into AREAL's
``create_dataloader(..., collate_fn=pad_sequences_to_tensors)`` — no
monkey-patching of ``areal.dataset`` is required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")

# Common dataset/file extensions that disqualify a string from being treated
# as a HuggingFace repo id even if its shape happens to match owner/name.
_FILE_EXTS = (
    ".jsonl", ".json", ".csv", ".tsv", ".parquet", ".arrow",
    ".txt", ".pkl", ".npz", ".npy",
)


def _looks_like_hf_repo_id(path: str) -> bool:
    """Heuristic: ``path`` is a HuggingFace repo id iff:

    - It does not exist on disk, AND
    - It matches ``<owner>/<name>`` (no extra slashes), AND
    - It is not obviously a relative file path (no leading ``./`` or ``../``,
      no trailing file extension like ``.jsonl``).
    """
    if Path(path).exists():
        return False
    if path.startswith(("./", "../", "/")):
        return False
    if path.endswith(_FILE_EXTS):
        return False
    return bool(_HF_REPO_ID_RE.match(path))


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("%s:%d malformed JSON, skipping: %s", path, line_no, e)
    return rows


def _index_in_train_split(idx: int, train_ratio: float) -> bool:
    """Deterministic per-row train/test bucket from sha1(str(idx)).

    Same row index always lands in the same split across reruns and across
    process ranks, so the train/test partition is reproducible without
    needing a separately-stored split column.
    """
    if train_ratio >= 1.0:
        return True
    if train_ratio <= 0.0:
        return False
    h = hashlib.sha1(str(idx).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF  # uniform [0, 1)
    return bucket < train_ratio


def get_seta_sft_dataset(
    path: str,
    split: str,
    tokenizer=None,  # accepted for AREAL signature parity; not used here
    max_length: int | None = None,
    train_ratio: float = 1.0,
    min_assistant_tokens: int = 1,
    **kwargs,
) -> Dataset:
    """Load a seta-env SFT dataset into the AREAL ``input_ids`` + ``loss_mask`` format.

    The source (HF Hub or local jsonl) is single-split (``train`` only).
    This loader hash-splits its rows into ``train`` and ``test`` based on
    the row index, controlled by ``train_ratio``.

    Args:
        path: Either a HuggingFace repo id (``owner/name``) or a local
            ``.jsonl`` file produced by ``build_sft_dataset.py``.
        split: ``"train"`` or ``"test"``.
        tokenizer: Unused. Tokens are pre-computed in the dataset. Accepted
            only so the loader plugs into AREAL's signature.
        max_length: Drop rows whose ``input_ids`` exceed this length.
            Mirrors AREAL's ``max_length`` filter semantics — *no truncation*.
        train_ratio: Fraction of rows assigned to the ``train`` split via a
            deterministic per-index hash. Default ``1.0`` — every row goes
            to ``train``, ``test`` is empty (so the AREAL evaluator iterates
            zero batches and is a no-op). Set to e.g. ``0.95`` to hold out
            a 5%% eval slice.
        min_assistant_tokens: Drop rows with fewer trainable tokens than
            this. Default 1 (no point training on a no-loss row).

    Returns:
        ``datasets.Dataset`` with exactly two columns: ``input_ids`` and
        ``loss_mask``. May be empty (e.g. ``split="test"`` with default
        ``train_ratio=1.0``) — downstream code must tolerate that.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    # 1. Load the source (single-split) into a list of rows.
    if _looks_like_hf_repo_id(path):
        logger.info(
            "[seta_sft] loading from HuggingFace Hub: %s (source split=train)",
            path,
        )
        try:
            hf_ds = load_dataset(path, split="train")
        except Exception as e:
            raise RuntimeError(f"Failed to load HF dataset {path!r}: {e}") from e
        rows = list(hf_ds)
        logger.info("[seta_sft] loaded %d rows from %s (HF Hub)", len(rows), path)
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist on disk and is not a valid "
                f"HF repo id: {path!r}"
            )
        rows = _iter_jsonl(p)
        logger.info(
            "[seta_sft] loaded %d rows from %s (local jsonl)", len(rows), p,
        )

    if not rows:
        logger.warning("[seta_sft] source loaded 0 rows; returning empty Dataset")
        return Dataset.from_dict({"input_ids": [], "loss_mask": []})

    # 2. Hash-split by row index according to train_ratio, then project to
    #    (input_ids, loss_mask) and apply length / mass filters.
    n_total = len(rows)
    n_dropped_no_train = 0
    n_dropped_too_long = 0
    n_dropped_shape = 0
    n_dropped_missing = 0
    n_other_split = 0

    want_train = (split == "train")
    projected: list[dict[str, list[int]]] = []
    for i, r in enumerate(rows):
        if _index_in_train_split(i, train_ratio) != want_train:
            n_other_split += 1
            continue
        input_ids = r.get("input_ids")
        loss_mask = r.get("loss_mask")
        if input_ids is None or loss_mask is None:
            n_dropped_missing += 1
            continue
        if len(input_ids) != len(loss_mask):
            n_dropped_shape += 1
            logger.debug(
                "[seta_sft] shape mismatch idx=%d: ids=%d mask=%d",
                i, len(input_ids), len(loss_mask),
            )
            continue
        if max_length is not None and len(input_ids) > max_length:
            n_dropped_too_long += 1
            continue
        if sum(loss_mask) < min_assistant_tokens:
            n_dropped_no_train += 1
            continue
        projected.append({"input_ids": input_ids, "loss_mask": loss_mask})

    logger.info(
        "[seta_sft] split=%s train_ratio=%s: kept %d / %d  "
        "(in_other_split=%d, too_long=%d, no_train_tokens=%d, "
        "shape_mismatch=%d, missing_fields=%d)",
        split, train_ratio, len(projected), n_total,
        n_other_split, n_dropped_too_long, n_dropped_no_train,
        n_dropped_shape, n_dropped_missing,
    )

    if not projected:
        # Empty splits are legal — happens when train_ratio=1.0 and the
        # trainer asks for the test split. Return an empty Dataset with the
        # correct schema so DistributedSampler / create_dataloader don't
        # crash on a missing column set.
        return Dataset.from_dict({"input_ids": [], "loss_mask": []})

    # 3. Build a HuggingFace Dataset (AREAL collator expects this)
    ds = Dataset.from_list(projected)
    extras = [c for c in ds.column_names if c not in ("input_ids", "loss_mask")]
    if extras:
        ds = ds.remove_columns(extras)
    return ds
