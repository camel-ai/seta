"""Build an SFT-ready, HuggingFace-displayable dataset from eval trial dirs.

For each trial directory under ``--trials-dir``, this script:

  1. Locates ``CAMEL_LOG_DIR`` and picks the LARGEST ``conv_*.json`` file
     in it (the most-complete conversation snapshot at end of rollout).
  2. Reconstructs the full message list as
     ``request.messages + [response.choices[0].message]``.
  3. Reads the per-task reward from ``verifier/ctrf.json``
     (passed / total tests; ``None`` if missing).
  4. Pulls provider-side token counts from ``response.usage``
     (prompt / completion / total / cached tokens).
  5. Applies the target tokenizer's chat template (default: Qwen/Qwen3-8B)
     to produce (a) a flat templated string, (b) the tokenized input_ids,
     and (c) a per-token assistant loss mask suitable for SFT.
  6. Emits one JSONL row per trial with:
        task_id, trial_uid, reward, model, conv_json_path,
        provider_token_counts, local_token_count,
        raw_conv_json (string),
        chat_template_str (string),
        input_ids (list[int]),
        loss_mask (list[int]),

The JSONL file is loadable via:

    from datasets import load_dataset
    ds = load_dataset("json", data_files="path/to/sft_data.jsonl")

Example:

    # Build the SFT JSONL with per-trial inspection artifacts (debug mode)
    python -m seta_env.utils.sft_utils.build_sft_dataset \\
        --trials-dir outputs/eval/eval_tito/<run>_merged/trials \\
        --output     outputs/eval/eval_tito/<run>_merged/sft.jsonl \\
        --debug

    # Build, then push to HuggingFace Hub as a public dataset (single
    # 'train' split, only fully-passing rollouts)
    python -m seta_env.utils.sft_utils.build_sft_dataset \\
        --trials-dir outputs/eval/eval_tito/<run>_merged/trials \\
        --output     outputs/eval/eval_tito/<run>_merged/sft.jsonl \\
        --model      Qwen/Qwen3-8B \\
        --min-reward 1.0 \\
        --push-to-hub camel-ai/seta-sft-kimi-k2.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery + extraction
# ---------------------------------------------------------------------------

def find_largest_conv_json(trial_dir: Path) -> Path | None:
    """Return the largest ``*.json`` file under ``trial_dir/CAMEL_LOG_DIR``.

    The largest file is the most-complete conversation snapshot — earlier
    snapshots represent intermediate turns that were superseded.
    """
    log_root = trial_dir / "CAMEL_LOG_DIR"
    if not log_root.is_dir():
        return None
    candidates = [p for p in log_root.rglob("*.json") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def read_reward(trial_dir: Path) -> float | None:
    """Read passed/total ratio from ``trial_dir/verifier/ctrf.json``."""
    ctrf = trial_dir / "verifier" / "ctrf.json"
    if not ctrf.exists():
        return None
    try:
        data = json.loads(ctrf.read_text())
        s = data["results"]["summary"]
        total = s.get("tests", 0)
        if not total:
            return 0.0
        return s.get("passed", 0) / total
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning("Could not parse %s: %s", ctrf, e)
        return None


def task_id_from_trial_name(trial_name: str) -> str:
    """``foo__task_001_t0_3313f0`` → ``foo__task_001``."""
    m = re.match(r"^(.+)_t\d+_[0-9a-f]+$", trial_name)
    return m.group(1) if m else trial_name


def reconstruct_messages(conv: dict) -> list[dict]:
    """``request.messages + response.choices[0].message`` → full conversation.

    Strips ``cached_tokens`` and other non-message fields. Returns the message
    list as a fresh list of dicts (caller may freely mutate).
    """
    msgs = [dict(m) for m in conv.get("request", {}).get("messages", [])]
    resp = conv.get("response", {})
    choices = resp.get("choices") or []
    if choices:
        final = choices[0].get("message")
        if isinstance(final, dict):
            msgs.append(dict(final))
    return msgs


def extract_provider_token_counts(conv: dict) -> dict:
    """Pull token-count fields from ``response.usage`` if present."""
    usage = (conv.get("response") or {}).get("usage") or {}
    return {
        "prompt_tokens":     usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens":      usage.get("total_tokens"),
        "cached_tokens":     usage.get("cached_tokens"),
    }


# ---------------------------------------------------------------------------
# Chat templating + assistant mask
# ---------------------------------------------------------------------------

_NO_THINK_DIRECTIVE = "/no_think"


def _sanitize_messages_for_template(
    messages: list[dict],
    no_thinking: bool = False,
) -> list[dict]:
    """Drop fields the tokenizer chat template doesn't understand.

    Provider responses can include ``refusal``, ``annotations``, ``audio``,
    etc. Most chat templates only consume ``role`` / ``content`` /
    ``tool_calls`` / ``tool_call_id`` / ``name`` / ``reasoning_content``.
    We pass those through and coerce ``content=None`` to ``""`` (templates
    that index into content crash on None).

    ``reasoning_content`` (the thinking trace from the rollout model) is
    preserved by default and forwarded to the chat template. The Qwen3
    template natively wraps it in ``<think>...</think>`` for assistant
    turns after the last real user message — which for an agent rollout is
    every assistant turn — so the SFT target includes the full chain of
    thought along with the visible content and tool calls.

    Set ``no_thinking=True`` to:
      1. **Prune** ``reasoning_content`` from every assistant message.
      2. **Append** the ``/no_think`` directive to the first system message
         (or prepend a new system message if none exists). This matches the
         runtime behavior of ``seta_env/environments/terminal_env.py`` which
         appends ``\\n/no_think`` to the system prompt when ``thinking:
         false`` is set in the eval config — keeping the SFT training-time
         system prompt distribution identical to the inference-time one.

    Combined with the post-render ``<think>`` strip in
    :func:`apply_template_and_mask`, this produces a thinking-free SFT
    target where assistant turns contain only ``content`` + ``tool_calls``.
    """
    keep = {
        "role", "content", "tool_calls", "tool_call_id", "name",
        "reasoning_content",
    }
    out = []
    for m in messages:
        clean = {k: v for k, v in m.items() if k in keep}
        if clean.get("content") is None:
            clean["content"] = ""
        # Null-coerce reasoning_content so a template that does
        # `reasoning_content is string` doesn't see None and blow up later
        # in the rendering path.
        if "reasoning_content" in clean and clean["reasoning_content"] is None:
            del clean["reasoning_content"]
        # Prune reasoning entirely for non-thinking SFT.
        if no_thinking and "reasoning_content" in clean:
            del clean["reasoning_content"]
        out.append(clean)

    # Append /no_think directive to the system message so the SFT input
    # distribution matches inference (terminal_env appends it at rollout time).
    if no_thinking:
        sys_idx = next(
            (i for i, m in enumerate(out) if m.get("role") == "system"),
            None,
        )
        if sys_idx is not None:
            sys_content = out[sys_idx].get("content") or ""
            if _NO_THINK_DIRECTIVE not in sys_content:
                out[sys_idx]["content"] = sys_content.rstrip() + "\n" + _NO_THINK_DIRECTIVE
        else:
            # No system message at all — prepend a minimal one carrying just
            # the directive so the inference-time distribution still matches.
            out.insert(0, {"role": "system", "content": _NO_THINK_DIRECTIVE})

    return out


def _render(messages: list[dict], tokenizer) -> str:
    """Render messages → templated string (no tokenization)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def _encode(text: str, tokenizer) -> list[int]:
    """Tokenize a rendered template string into a flat list of input ids.

    Avoids ``apply_chat_template(tokenize=True)`` because in transformers
    5.x that returns a ``BatchEncoding`` dict and various branches of the
    chat template (e.g. tool handling) can disagree with the message-list
    code path. Rendering to a string and tokenizing the string is robust.
    """
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _find_assistant_spans(
    input_ids: list[int],
    tokenizer,
) -> list[tuple[int, int]]:
    """Find [start, end) token spans of every assistant turn in ``input_ids``.

    Scans for ``<|im_start|>assistant ...`` blocks and returns the half-open
    span including the opening ``<|im_start|>`` special token and the closing
    ``<|im_end|>`` (plus the single newline that always follows it in
    Qwen-style templates, if present). Everything outside these spans is
    context and should not contribute to the SFT loss.

    This approach is robust to the Qwen3 chat template's merging of
    consecutive tool messages into a single ``<|im_start|>user`` block —
    something the prefix-tokenization approach struggles with because the
    template emits a premature ``<|im_end|>`` when only one tool in a run
    has been rendered.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_start is None or im_start == tokenizer.unk_token_id or \
       im_end   is None or im_end   == tokenizer.unk_token_id:
        raise ValueError(
            "Tokenizer is missing <|im_start|> or <|im_end|> special tokens; "
            "this mask builder only supports im_start/im_end chat formats."
        )

    # Qwen templates emit "<|im_start|>{role}\n..." where {role} is a single
    # token like 'assistant' (or 'user', 'system', 'tool'). We check the token
    # immediately after <|im_start|>.
    assistant_role_id = tokenizer.convert_tokens_to_ids("assistant")
    if assistant_role_id is None or assistant_role_id == tokenizer.unk_token_id:
        # Fall back to decoding; slower but tokenizer-agnostic.
        assistant_role_id = None

    spans: list[tuple[int, int]] = []
    n = len(input_ids)
    i = 0
    while i < n:
        if input_ids[i] != im_start:
            i += 1
            continue
        # Found <|im_start|>. Check the role token.
        if i + 1 >= n:
            break
        role_tok = input_ids[i + 1]
        if assistant_role_id is not None:
            is_assistant = (role_tok == assistant_role_id)
        else:
            is_assistant = tokenizer.decode([role_tok]).strip() == "assistant"
        if not is_assistant:
            i += 1
            continue
        # Find the matching <|im_end|>
        j = i + 1
        while j < n and input_ids[j] != im_end:
            j += 1
        if j >= n:
            # Unterminated assistant block — include to EOF.
            spans.append((i, n))
            break
        end = j + 1  # include <|im_end|>
        # Include the single trailing newline that Qwen templates always
        # append after <|im_end|> inside the assistant rendering.
        if end < n:
            nl_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
            # Greedy: include any number of newline tokens that immediately follow.
            while end < n and input_ids[end] in nl_ids:
                end += 1
        spans.append((i, end))
        i = end
    return spans


_THINK_BLOCK_RE = re.compile(r"<think>\n.*?\n</think>\n\n", re.DOTALL)


def apply_template_and_mask(
    messages: list[dict],
    tokenizer,
    no_thinking: bool = False,
) -> tuple[str, list[int], list[int]]:
    """Apply the tokenizer's chat template and build an assistant loss mask.

    Returns:
        (template_str, input_ids, loss_mask)

    Strategy:
      1. Render the full conversation to a templated string (once).
      2. (no_thinking only) Strip every ``<think>...</think>\\n\\n`` block
         from the rendered string. Pruning ``reasoning_content`` in
         sanitization removes the content for non-final assistant turns,
         but the Qwen3 template still injects an empty ``<think>\\n\\n</think>\\n\\n``
         before the LAST assistant turn (the ``loop.last`` branch unconditionally
         emits one); the regex pass kills that too.
      3. Tokenize the (possibly stripped) string to get ``input_ids``.
      4. Scan ``input_ids`` for every ``<|im_start|>assistant ... <|im_end|>``
         span and mark those tokens as ``loss_mask[i] = 1``.

    This is O(N) total, robust to consecutive-tool-message merging in the
    Qwen3 chat template, and makes the boundary handling explicit: the
    trainable span is [<|im_start|>, <|im_end|>\\n] inclusive for every
    assistant turn, and everything else is context.
    """
    clean = _sanitize_messages_for_template(messages, no_thinking=no_thinking)

    template_str = _render(clean, tokenizer)
    if no_thinking:
        template_str = _THINK_BLOCK_RE.sub("", template_str)
    full_ids = _encode(template_str, tokenizer)
    mask = [0] * len(full_ids)

    for start, end in _find_assistant_spans(full_ids, tokenizer):
        for j in range(start, end):
            mask[j] = 1

    # Sanity check: the number of marked <|im_start|> tokens equals the
    # number of assistant messages that survived sanitization and made it
    # through the template (Qwen3 renders every assistant message with
    # `reasoning_content` — there is no silent drop path).
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    n_asst_in_ids = sum(
        1 for i, tid in enumerate(full_ids)
        if tid == im_start and mask[i] == 1
    )
    n_asst_in_msgs = sum(1 for m in clean if m.get("role") == "assistant")
    if n_asst_in_ids != n_asst_in_msgs:
        logger.warning(
            "assistant span count mismatch: %d in tokens vs %d in messages "
            "(template may have dropped or merged turns)",
            n_asst_in_ids, n_asst_in_msgs,
        )

    return template_str, full_ids, mask


# ---------------------------------------------------------------------------
# Per-trial inspection artifacts (debug mode)
# ---------------------------------------------------------------------------

def write_inspection_artifacts(
    row: dict,
    tokenizer,
    out_dir: Path,
) -> None:
    """Write three human-readable artifacts for one trial:

    - ``inspect.txt``       — boundary-by-boundary token dump (first/last N
      tokens of every ``<|im_start|>...<|im_end|>`` block, with mask + decoded
      text). Use this to eyeball that ``<|im_start|>``, ``<|im_end|>``,
      ``<think>``, ``</think>`` and trailing ``\\n`` tokens are masked
      correctly.
    - ``chat_template.txt`` — plain rendered chat template (the same string
      that was tokenized). Useful for spot-checking ``<think>`` blocks and
      ``<tool_call>`` formatting.
    - ``summary.json``      — top-line metrics for the trial.

    The output dir is created on demand. Existing files in it are overwritten.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = row["input_ids"]
    mask = row["loss_mask"]
    n = len(ids)

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_id    = tokenizer("\n", add_special_tokens=False)["input_ids"]
    nl_id    = nl_id[0] if len(nl_id) == 1 else None

    def decode_tok(tid):
        return repr(tokenizer.decode([tid], skip_special_tokens=False))

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("SFT DATASET BOUNDARY INSPECTION")
    lines.append("=" * 90)
    lines.append(f"task_id:            {row['task_id']}")
    lines.append(f"trial_uid:          {row['trial_uid']}")
    lines.append(f"reward:             {row['reward']}")
    lines.append(f"model:              {row['model']}")
    lines.append(f"n_messages:         {row['n_messages']}")
    lines.append(f"local_token_count:  {n}")
    pct = sum(mask) / n * 100 if n else 0.0
    lines.append(f"n_assistant_tokens: {sum(mask)}  ({pct:.1f}%)")
    lines.append(f"provider_tokens:    {row['provider_token_counts']}")
    lines.append("")
    lines.append("Special token ids:")
    for name in ("<|im_start|>", "<|im_end|>", "<think>", "</think>",
                 "<tool_call>", "</tool_call>", "<tool_response>",
                 "</tool_response>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != tokenizer.unk_token_id:
            lines.append(f"  {name:20s} = {tid}")
    if nl_id is not None:
        lines.append(f"  {'\\n':20s} = {nl_id}")
    lines.append("")
    lines.append("Legend:  [idx | mask | id | decoded]   "
                 "mask=1 → trainable (loss), mask=0 → context")
    lines.append("")

    # Walk every <|im_start|>...<|im_end|>(\n) block
    def fmt_row(i: int) -> str:
        marker = "  ←TRAIN" if mask[i] else ""
        return (f"  [{i:5d} | m={mask[i]} | id={ids[i]:6d} | "
                f"{decode_tok(ids[i]):26s}]{marker}")

    head, tail = 10, 10
    i = 0
    bi = 0
    while i < n:
        if ids[i] != im_start:
            # Stray token outside any block (shouldn't normally happen)
            lines.append(fmt_row(i))
            i += 1
            continue
        # Find the matching <|im_end|>, then absorb trailing newlines
        j = i + 1
        while j < n and ids[j] != im_end:
            j += 1
        end = j + 1 if j < n else n
        while end < n and ids[end] == nl_id:
            end += 1
        role_tok = ids[i + 1] if i + 1 < n else None
        role = (tokenizer.decode([role_tok]).strip()
                if role_tok is not None else "?") or "?"
        block_len = end - i
        block_mask = sum(mask[i:end])
        ratio = block_mask / block_len * 100 if block_len else 0.0
        lines.append("─" * 90)
        lines.append(
            f"BLOCK {bi:2d}  role={role:10s}  pos={i:5d}..{end-1:5d}  "
            f"len={block_len:5d}  mask_sum={block_mask}/{block_len} ({ratio:3.0f}%)"
        )
        lines.append("─" * 90)
        if block_len <= head + tail + 4:
            idxs = list(range(i, end))
        else:
            idxs = list(range(i, i + head)) + [None] + list(range(end - tail, end))
        for k in idxs:
            if k is None:
                lines.append(
                    f"     ...   ({block_len - head - tail} tokens elided)   ..."
                )
            else:
                lines.append(fmt_row(k))
        bi += 1
        i = end

    (out_dir / "inspect.txt").write_text("\n".join(lines))
    (out_dir / "chat_template.txt").write_text(row["chat_template_str"])

    summary = {
        "task_id":               row["task_id"],
        "trial_uid":             row["trial_uid"],
        "reward":                row["reward"],
        "model":                 row["model"],
        "n_messages":            row["n_messages"],
        "local_token_count":     row["local_token_count"],
        "n_assistant_tokens":    row["n_assistant_tokens"],
        "assistant_ratio":       (
            round(row["n_assistant_tokens"] / row["local_token_count"], 4)
            if row["local_token_count"] else 0.0
        ),
        "provider_token_counts": row["provider_token_counts"],
        "n_blocks":              bi,
        "chat_template_chars":   len(row["chat_template_str"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# HuggingFace Hub upload
# ---------------------------------------------------------------------------

def push_jsonl_to_hub(jsonl_path: Path, repo_id: str) -> None:
    """Load the produced JSONL and push as a single-split ``DatasetDict``
    (``train`` only) to HuggingFace Hub, **preserving every column**.

    The published dataset carries the full per-trial diagnostic record:
    ``task_id``, ``trial_uid``, ``reward``, ``model``, ``conv_json_path``,
    ``provider_token_counts``, ``local_token_count``, ``n_assistant_tokens``,
    ``n_messages``, ``raw_conv_json``, ``chat_template_str``, ``input_ids``,
    ``loss_mask``. The AREAL-shape projection to ``(input_ids, loss_mask)``
    happens at **load time** in
    :func:`scripts.areal_sft.seta_sft_dataset.get_seta_sft_dataset`, so the
    trainer still gets the lean two-column shape it expects.

    Keeping the diagnostic columns on the Hub means downstream consumers
    can:
      - inspect the full conversation that produced each row,
      - re-tokenize with a different tokenizer / chat template,
      - filter by reward, source model, or token count without re-running
        the build pipeline.

    Args:
        jsonl_path: Path to the JSONL written by build_row().
        repo_id:    HuggingFace repo id, e.g. ``camel-ai/seta-sft-kimi-k2.5``.
    """
    from datasets import Dataset, DatasetDict

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN env var not set; cannot push to HuggingFace Hub.")
        return

    logger.info("[push] reading %s (all columns preserved)...", jsonl_path)

    train_rows: list[dict] = []
    n_total = 0
    n_dropped = 0

    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_dropped += 1
                continue
            ids = row.get("input_ids")
            mask = row.get("loss_mask")
            if ids is None or mask is None or len(ids) != len(mask):
                n_dropped += 1
                continue
            # Keep every field — the loader projects at training time.
            train_rows.append(row)

    logger.info(
        "[push] %d rows total: %d train, %d dropped",
        n_total, len(train_rows), n_dropped,
    )

    if not train_rows:
        logger.error("[push] nothing to push (no valid rows)")
        return

    ds = DatasetDict({"train": Dataset.from_list(train_rows)})
    logger.info("[push] pushing to %s ...", repo_id)
    try:
        ds.push_to_hub(repo_id=repo_id, token=hf_token)
    except Exception as e:
        logger.error("[push] push_to_hub failed: %s", e)
        return

    print()
    print("=" * 64)
    print("  HUGGINGFACE PUSH SUMMARY")
    print("=" * 64)
    print(f"  repo:        {repo_id}")
    print(f"  train rows:  {len(train_rows)}")
    print(f"  → https://huggingface.co/datasets/{repo_id}")
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# Main row builder
# ---------------------------------------------------------------------------

def build_row(
    trial_dir: Path,
    tokenizer,
    min_reward: float | None = 0.0,
    no_thinking: bool = False,
) -> dict[str, Any] | None:
    """Build one JSONL row for a single trial dir, or None if unprocessable
    or below the reward threshold.

    Args:
        trial_dir: per-trajectory dir under <merged>/trials/.
        tokenizer: HuggingFace tokenizer for the chat template + tokenization.
        min_reward: drop trials whose reward is None (no ctrf.json) OR is
            strictly less than this value. Default 0.0 — keep every trial
            that produced a verifier reward (including 0/N rollouts), drop
            only the ones with no ctrf.json at all (errored / timed out
            before verification ran). Pass ``None`` to keep every trial
            regardless of reward, including missing-ctrf ones (diagnostic
            builds). Pass e.g. 1.0 to only keep fully-passing rollouts.
        no_thinking: if True, prune ``reasoning_content`` from every assistant
            message and strip any leftover ``<think>...</think>`` blocks
            from the rendered template, producing a thinking-free SFT
            target. Default False (keep reasoning).
    """
    # Reward gate first — cheap, filters out failed/errored trials before
    # we spend time tokenizing 12k+ tokens of conversation that we'd discard.
    reward = read_reward(trial_dir)
    if min_reward is not None:
        if reward is None:
            logger.debug(
                "[%s] reward=None (no ctrf.json) — skipping (min_reward=%s)",
                trial_dir.name, min_reward,
            )
            return None
        if reward < min_reward:
            logger.debug(
                "[%s] reward=%.3f < %.3f — skipping",
                trial_dir.name, reward, min_reward,
            )
            return None

    conv_path = find_largest_conv_json(trial_dir)
    if conv_path is None:
        logger.warning("[%s] no CAMEL_LOG_DIR/*.json — skipping", trial_dir.name)
        return None

    try:
        conv = json.loads(conv_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[%s] could not read %s: %s", trial_dir.name, conv_path, e)
        return None

    messages = reconstruct_messages(conv)
    if not messages:
        logger.warning("[%s] empty messages in %s", trial_dir.name, conv_path)
        return None

    try:
        template_str, input_ids, loss_mask = apply_template_and_mask(
            messages, tokenizer, no_thinking=no_thinking,
        )
    except Exception as e:
        logger.error("[%s] chat template failed: %s", trial_dir.name, e)
        return None

    return {
        "task_id":        task_id_from_trial_name(trial_dir.name),
        "trial_uid":      trial_dir.name,
        "reward":         reward,
        "model":          conv.get("model"),
        "conv_json_path": str(conv_path),
        "provider_token_counts": extract_provider_token_counts(conv),
        "local_token_count":     len(input_ids),
        "n_assistant_tokens":    sum(loss_mask),
        "n_messages":            len(messages),
        "raw_conv_json":         json.dumps(conv, ensure_ascii=False),
        "chat_template_str":     template_str,
        "input_ids":             input_ids,
        "loss_mask":        loss_mask,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trials-dir",
        required=True,
        type=Path,
        help="Directory containing per-trial subfolders (each with CAMEL_LOG_DIR/ "
             "and verifier/ctrf.json).",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model id (or local path) for the chat template + tokenizer. "
             "Default: Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging AND write per-trial inspection artifacts "
             "(inspect.txt, chat_template.txt, summary.json) into "
             "<output>.inspect/<trial_uid>/ alongside the squashed JSONL. "
             "Use --inspect-dir to override the location.",
    )
    parser.add_argument(
        "--inspect-dir",
        type=Path,
        default=None,
        help="Override the per-trial inspection output dir. Implies --debug. "
             "Default (under --debug): <output>.inspect/",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        metavar="REPO_ID",
        help="After writing the JSONL, also push it to HuggingFace Hub as a "
             "public dataset under <REPO_ID> (e.g. 'camel-ai/seta-sft-kimi-k2.5'). "
             "Requires HF_TOKEN env var with write access. The dataset is "
             "pushed as a single 'train' split containing only the columns "
             "AREAL needs (input_ids, loss_mask) — diagnostic columns are dropped.",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Drop trials whose reward (passed/total tests from "
             "verifier/ctrf.json) is None OR strictly below this value. "
             "Default 0.0 — keep every trial that produced a verifier reward, "
             "drop only the ones with no ctrf.json. Set to e.g. 1.0 to only "
             "keep fully-passing rollouts.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Prune reasoning_content from every assistant message and "
             "strip leftover <think>...</think> blocks from the rendered "
             "template. Produces a thinking-free SFT target where assistant "
             "turns contain only content + tool_calls. Default off.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.trials_dir.is_dir():
        parser.error(f"trials-dir does not exist: {args.trials_dir}")

    # Lazy import so the module is importable for unit testing without transformers
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Discover trial dirs
    candidates = sorted(
        p for p in args.trials_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_build")
    )
    logger.info("Processing %d trials", len(candidates))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Debug/inspect mode: per-trial artifacts alongside the squashed JSONL.
    inspect_root: Path | None = None
    if args.inspect_dir is not None:
        inspect_root = args.inspect_dir
        args.debug = True  # passing --inspect-dir implies debug
    elif args.debug:
        inspect_root = args.output.with_suffix(args.output.suffix + ".inspect")
    if inspect_root is not None:
        inspect_root.mkdir(parents=True, exist_ok=True)
        logger.info("Per-trial inspection artifacts → %s", inspect_root)

    logger.info("dropping trials with reward < %s (or no ctrf.json)", args.min_reward)

    n_ok = 0
    n_skip = 0
    with open(args.output, "w") as fh:
        for trial in candidates:
            row = build_row(
                trial, tokenizer,
                min_reward=args.min_reward,
                no_thinking=args.no_thinking,
            )
            if row is None:
                n_skip += 1
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
            logger.info(
                "[%s] reward=%s msgs=%d local_tokens=%d assistant_tokens=%d "
                "provider_total=%s",
                trial.name,
                row["reward"],
                row["n_messages"],
                row["local_token_count"],
                row["n_assistant_tokens"],
                (row["provider_token_counts"] or {}).get("total_tokens"),
            )
            if inspect_root is not None:
                try:
                    write_inspection_artifacts(
                        row, tokenizer, inspect_root / row["trial_uid"],
                    )
                except Exception as e:
                    logger.warning(
                        "[%s] failed to write inspection artifacts: %s",
                        trial.name, e,
                    )

    print()
    print("=" * 64)
    print("  SFT DATASET BUILD SUMMARY")
    print("=" * 64)
    print(f"  trials processed:  {n_ok}")
    print(f"  trials skipped:    {n_skip}")
    print(f"  → {args.output}")
    if inspect_root is not None:
        print(f"  → {inspect_root}/<trial_uid>/  (inspect.txt, chat_template.txt, summary.json)")
    print("=" * 64)
    print()

    if args.push_to_hub:
        push_jsonl_to_hub(
            jsonl_path=args.output,
            repo_id=args.push_to_hub,
        )


if __name__ == "__main__":
    sys.exit(main())
