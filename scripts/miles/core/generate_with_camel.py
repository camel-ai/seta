"""Custom rollout function for miles — POSTs the prompt to env_service /step
and converts the response into a miles ``Sample``.

Wire-protocol on /step is documented in
``scripts/miles/docs/sample_contract.md``: env_service runs the CAMEL agent
loop in venv_cpu, talks to SGLang directly, and returns
``response["sample"]`` already in miles-Sample shape (token-level TITO,
response-only loss_mask + rollout_log_probs).

This function is loaded by miles' ``--custom-generate-function-path``
([miles/rollout/sglang_rollout.py:243-254](/root/miles/miles/rollout/sglang_rollout.py)).
The signature matches the canonical pattern in
``examples/retool/generate_with_retool.py:215`` and
``examples/strands_sglang/generate_with_strands.py:50``.

Environment variables (set in launcher):
    CAMEL_ENV_SERVICE_URL   default http://127.0.0.1:8002
    CAMEL_DATASET_NAME      default seta-env-final
    CAMEL_TRIAL_NAME        default camel_step3
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from argparse import Namespace
from typing import Any

import numpy as np

from miles.utils.http_utils import post
from miles.utils.types import Sample


def _populate_failed_sample_placeholders(sample: Sample, args: Namespace) -> None:
    """Stamp minimal-valid placeholders on a failed-trajectory Sample.

    miles' rollout-to-train assembly at miles/ray/rollout.py:406-410 ONLY
    inspects ``samples[0]`` to decide whether to include ``rollout_routed_experts``
    / ``rollout_indexer_topk`` / ``rollout_log_probs`` in train_data. If the
    first sample in the batch happens to be a failed trajectory whose
    rollout_routed_experts is None, the field is dropped from train_data
    entirely → _fill_replay_data() raises "rollout_routed_experts is required
    in rollout_data for replay."

    This populates a length-1 dummy that satisfies Sample.validate() and
    keeps the field non-None, so miles always materialises the dict key.
    The actual values are zero placeholders; the sample's remove_sample=True
    (set by reward_func) zeros out its loss_mask contribution anyway.
    """
    sample.tokens = [0, 0]                  # 1 prompt sentinel + 1 response sentinel
    sample.response_length = 1
    sample.loss_mask = [0]                  # zero gradient contribution
    sample.rollout_log_probs = [0.0]
    if getattr(args, "use_rollout_routing_replay", False):
        num_layers = int(getattr(args, "num_layers", 0) or 43)
        moe_topk = int(getattr(args, "moe_router_topk", 0) or 6)
        sample.rollout_routed_experts = np.zeros(
            (1, num_layers, moe_topk), dtype=np.int32
        )
    if getattr(args, "use_rollout_indexer_replay", False):
        indexer_layers = int(getattr(args, "num_indexer_layers", 0) or 43)
        indexer_topk = int(getattr(args, "index_topk", 0) or 512)
        sample.rollout_indexer_topk = np.zeros(
            (1, indexer_layers, indexer_topk), dtype=np.int32
        )


def _decode_routing(
    b64: str | None,
    expected_token_count: int,
    num_layers: int,
    topk: int,
    field: str,
) -> np.ndarray | None:
    """Decode SGLang's base64 int32 routing buffer to shape (T-1, layers, topk).

    Mirrors the same routine in scripts/miles/megatron/build_rollout_dump.py but
    runs at rollout time so the in-memory Sample returned to miles carries
    routing for the direct rollout→train path (no dump file in the middle).
    """
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64.encode("ascii"))
        arr = np.frombuffer(raw, dtype=np.int32)
        expected_entries = expected_token_count - 1
        return arr.reshape(expected_entries, num_layers, topk)
    except Exception as e:
        logger.warning("[camel_terminal] %s decode/reshape failed: %s", field, e)
        return None

logger = logging.getLogger(__name__)


# Map env_service "status" string → miles Sample.Status enum.
# env_service emits the value of ``Sample.Status.value`` ("completed",
# "truncated", "aborted", "failed", "pending") so this is just a passthrough.
_STATUS_MAP = {s.value: s for s in Sample.Status}


def _resolve_env_service_url() -> str:
    return os.getenv("CAMEL_ENV_SERVICE_URL", "http://127.0.0.1:8002")


def _resolve_dataset_name() -> str:
    """Return CAMEL_DATASET_NAME from env. Fail loudly if unset.

    This MUST be set in the Ray worker env (via the launcher's runtime_env's
    `env_vars` block — see run_deepseek_v4.py's `extra_env_vars` injection).
    Silently falling back to a default ("seta-env-final") previously caused
    env_service to look up tasks in the wrong dataset dir for ~16% of GRPO
    prompts (camel_uuid / camel_numeric labels that only exist under the
    camel-combined pool), producing silent zero-reward training data.
    """
    v = os.getenv("CAMEL_DATASET_NAME")
    if not v:
        raise RuntimeError(
            "CAMEL_DATASET_NAME is not set in this Ray worker. Make sure the "
            "launcher injects it into ray job submit's runtime_env.env_vars "
            "(see run_deepseek_v4.py:_train extra_env_vars). Refusing to fall "
            "back to a default — that would silently route /step requests to "
            "the wrong dataset_root subdir."
        )
    return v


def _resolve_trial_name() -> str:
    """Return CAMEL_TRIAL_NAME from env. Fail loudly if unset.

    Same propagation requirement as _resolve_dataset_name. Without it, trial
    dirs land at HARBOR_ROOT/trials/camel_step3/ instead of the per-run folder.
    """
    v = os.getenv("CAMEL_TRIAL_NAME")
    if not v:
        raise RuntimeError(
            "CAMEL_TRIAL_NAME is not set in this Ray worker. Same fix as "
            "CAMEL_DATASET_NAME — inject it via runtime_env.env_vars."
        )
    return v


def _instruction_from_sample(sample: Sample) -> str:
    """Extract the task instruction text from a miles Sample.

    Miles populates ``sample.prompt`` either as a string (plain instruction)
    or as a list-of-dicts (chat-formatted prompt). For seta-env-final, our
    dataset wraps each task's instruction.md into the prompt; we return the
    string form directly. For chat-formatted prompts we extract the last
    user-role content as the instruction.
    """
    p = sample.prompt
    if isinstance(p, str):
        return p
    if isinstance(p, list):
        for msg in reversed(p):
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str):
                    return c
        # Last resort — concat all string contents
        return "\n".join(
            str(m.get("content", "")) for m in p if isinstance(m, dict)
        )
    return str(p)


def _coerce_metadata(md) -> dict:
    """``sample.metadata`` may arrive as a dict or as a JSON string (parquet
    stores it as string when prepare_data.py uses ``json.dumps``). Coerce.
    """
    if md is None:
        return {}
    if isinstance(md, dict):
        return md
    if isinstance(md, str):
        try:
            return json.loads(md)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _task_name_from_sample(sample: Sample) -> str:
    """Pull a stable task name from the sample. Falls back to index/uid."""
    md = _coerce_metadata(sample.metadata)
    # Different dataset loaders may stash this differently — try a few
    for key in ("task_name", "instance_id", "task_id", "id"):
        v = md.get(key)
        if v:
            return str(v)
    if sample.index is not None:
        return f"sample_{sample.index}"
    return f"sample_{int(time.time() * 1000)}"


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """Rollout one trajectory by delegating to env_service.

    The env_service runs the full CAMEL multi-turn agent loop (talking to
    SGLang directly) and returns a miles-Sample-shaped payload that we copy
    onto ``sample`` and return. The SGLang URL is configured server-side in
    seta_env_config.yaml; we don't pass it from here.

    Returns the same Sample object, mutated in place.
    """
    env_service_url = _resolve_env_service_url()
    dataset_name = _resolve_dataset_name()
    trial_name = _resolve_trial_name()

    task_name = _task_name_from_sample(sample)
    instruction = _instruction_from_sample(sample)
    uid = f"{task_name}_{trial_name}_{sample.index if sample.index is not None else 0}"

    # Miles auto-assigns the SGLang router host/port at startup
    # (miles/ray/rollout.py:666-668). Pass the resolved URL to env_service so
    # its V4 model backend hits miles' Ray-spawned SGLang, not the
    # default http://localhost:30000 baked into seta_env_config.yaml (which
    # is only correct for the standalone-SGLang flow).
    sglang_ip = getattr(args, "sglang_router_ip", None) or "localhost"
    sglang_port = getattr(args, "sglang_router_port", None) or 30000
    sglang_url = f"http://{sglang_ip}:{sglang_port}"

    request = {
        "task": {
            "task_name": task_name,
            "instruction": instruction,
        },
        "uid": uid,
        "traj_i": sample.group_index or 0,
        # env_service overrides model_config["url"] with this value per
        # request (env_service.py:308-309). The V4 backend's base_url then
        # becomes this — pointing the SGLangClient at miles' actual SGLang.
        "model_url": sglang_url,
        "model_api_key": "dummy",
        "dataset_name": dataset_name,
        "task_name": task_name,
        "trial_name": trial_name,
        # R3 / indexer routing capture toggles. miles enables --enable-return-
        # routed-experts on its SGLang launch when these args are on
        # (sglang_engine.py:562-565); we must also ask for the data per-request
        # in our /generate payload, otherwise the response's meta_info won't
        # carry routed_experts and tito_state will lack the field needed for
        # downstream R3 replay (training-side per-token expert pinning).
        "return_routed_experts": bool(getattr(args, "use_rollout_routing_replay", False)),
        "return_indexer_topk": bool(getattr(args, "use_rollout_indexer_replay", False)),
    }

    t0 = time.time()
    try:
        response = await post(f"{env_service_url}/step", request)
    except Exception as e:
        logger.error(
            "[camel_terminal] /step failed for %s: %s", uid, e, exc_info=True
        )
        sample.status = Sample.Status.FAILED
        _populate_failed_sample_placeholders(sample, args)
        return sample
    dt = time.time() - t0

    err = response.get("error")
    if err:
        logger.warning("[camel_terminal] env_service error for %s: %s", uid, err)

    sample_data = response.get("sample")
    if sample_data is None:
        # The trajectory may have failed before the model dumped anything
        # (e.g. daytona sandbox provisioning failed). Mark as failed so the
        # outer rollout loop can drop or retry.
        run_info = response.get("run_info") or {}
        err_info = run_info.get("error_info") or {}
        logger.warning(
            "[camel_terminal] no sample payload for %s (stage=%s, err=%s)",
            uid,
            err_info.get("stage"),
            err_info.get("error_message", "")[:200],
        )
        sample.status = Sample.Status.FAILED
        _populate_failed_sample_placeholders(sample, args)
        return sample

    # ── copy miles-Sample-shaped fields onto the output Sample ──────────────
    sample.tokens = list(sample_data["tokens"])
    sample.response_length = int(sample_data["response_length"])
    sample.loss_mask = list(sample_data["loss_mask"])
    sample.rollout_log_probs = [float(x) for x in sample_data["rollout_log_probs"]]
    sample.response = sample_data.get("response") or ""

    # R3 routing — decode if env_service surfaced it (i.e. SGLang server has
    # --enable-return-routed-experts and miles' use_rollout_routing_replay is
    # on). Shape: (len(tokens)-1, num_layers, topk). miles Sample.validate
    # enforces the length match.
    routing_token_count = int(
        sample_data.get("rollout_routing_token_count") or len(sample.tokens)
    )

    # Replay buffers must be SELF-DESCRIBING: the serving model embeds the buffer
    # shape (num_layers, topk) in the sample payload, sourced from the model
    # config / engine. We do NOT guess with hardcoded per-model constants — a
    # missing shape (or a missing buffer for a non-empty trajectory) means the
    # rollout data is unusable for replay, so we fail loudly rather than train on
    # a silently-wrong reshape.
    def _require_dim(sample_key: str, field: str) -> int:
        v = sample_data.get(sample_key)
        if not v:
            raise ValueError(
                f"[camel_terminal] {uid}: replay enabled but env_service did not "
                f"return '{sample_key}' for {field}; cannot decode the buffer "
                f"without its shape. The serving model must surface routing dims "
                f"via dump_tito_state (DeepSeekV4SGLangModel._resolve_routing_dims "
                f"/ meta_info.indexer_topk_num_layers). Refusing to guess."
            )
        return int(v)

    def _decode_replay(b64_key: str, layers_key: str, topk_key: str, field: str):
        b64 = sample_data.get(b64_key)
        if b64:
            arr = _decode_routing(
                b64,
                expected_token_count=routing_token_count,
                num_layers=_require_dim(layers_key, field),
                topk=_require_dim(topk_key, field),
                field=field,
            )
            if arr is None:
                raise ValueError(
                    f"[camel_terminal] {uid}: {field} buffer was present but "
                    f"failed to decode/reshape (see warning above)."
                )
            return arr
        if sample.response_length > 0:
            # Generated tokens but no replay buffer => the engine wasn't launched
            # with the capture flag (or the /step request didn't ask for it).
            # This is a misconfiguration, not benign data — fail loudly.
            raise ValueError(
                f"[camel_terminal] {uid}: replay enabled and the trajectory "
                f"generated {sample.response_length} tokens, but env_service "
                f"returned no '{b64_key}'. Ensure the SGLang engine was launched "
                f"with the capture flag (--use-rollout-routing-replay / "
                f"--use-rollout-indexer-replay) and the /step request set the "
                f"corresponding return_* toggle."
            )
        # response_length == 0 → empty generation, no routing exists; the
        # empty-response guard below backfills a removed placeholder.
        return None

    if getattr(args, "use_rollout_routing_replay", False):
        sample.rollout_routed_experts = _decode_replay(
            "rollout_routed_experts_b64",
            "rollout_routed_experts_num_layers",
            "rollout_routed_experts_topk",
            "rollout_routed_experts",
        )
    if getattr(args, "use_rollout_indexer_replay", False):
        sample.rollout_indexer_topk = _decode_replay(
            "rollout_indexer_topk_b64",
            "rollout_indexer_num_layers",
            "rollout_indexer_topk_k",
            "rollout_indexer_topk",
        )

    # GUARD: an empty-response sample (response_length=0, e.g. model emitted
    # nothing) has no routing data → rollout_routed_experts decodes to None.
    # Such a sample can still land in a kept group (if its group has reward
    # variance), and miles' R3 replay then crashes on torch.from_numpy(None).
    # Backfill the same length-1 zero placeholder used for failed samples and
    # mark remove_sample=True so it contributes zero gradient.
    if getattr(args, "use_rollout_routing_replay", False) and sample.rollout_routed_experts is None:
        logger.warning(
            "[camel_terminal] %s: completed with no routing data "
            "(response_length=%d) — backfilling placeholder + remove_sample=True",
            uid, sample.response_length,
        )
        _populate_failed_sample_placeholders(sample, args)
        sample.remove_sample = True

    # status: the dump doesn't track trajectory-level status yet (see
    # HANDOFF "outstanding questions"). Infer from run_info.
    run_info = response.get("run_info") or {}
    summary = run_info.get("agent_summary") or {}

    # Surface SGLang prefix-cache accounting → miles' wandb eval/train metric
    # `prefix_cache_hit_rate`. env_service (TerminalEnvironment.step) reads
    # cumulative cached_tokens/prompt_tokens off the underlying
    # DeepSeekV4SGLangModel and embeds them in run_info.cache_stats. Without
    # this hand-off, the metric reports 0/0 = 0.0 because miles' standard
    # SGLang path (sglang_rollout.py:212 update_from_meta_info) is bypassed
    # by our custom-generate-fn entirely.
    cache_stats = run_info.get("cache_stats") or {}
    if cache_stats.get("prompt_tokens"):
        # Call prefix_cache_info.add() directly (NOT update_from_meta_info())
        # — the latter ALSO touches finish_reason/spec_info which we don't
        # carry here, and would KeyError on missing fields.
        sample.prefix_cache_info.add(cache_stats)
    term = summary.get("important_termination_reason") or summary.get(
        "termination_reason"
    )
    if term in ("task_finished", "stop", None):
        sample.status = Sample.Status.COMPLETED
    elif term in ("max_iteration", "context_length_exceeded", "length"):
        sample.status = Sample.Status.TRUNCATED
    elif term in ("abort",):
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED

    # Set sample.reward directly from env_service's pass_ratio result.
    # Miles' rollout layer (sglang_rollout.py:266-277) explicitly checks
    # `sample.reward is None` and only calls the custom-rm-fn for samples
    # that don't already have a reward — exactly the multi-turn pattern
    # ("for multi agent system, the reward of some sample is calculated
    # during generation"). Setting it here skips the redundant RM round-trip.
    #
    # INVALID-SAMPLE CONTRACT: when env_service can't compute a reward (None
    # or non-numeric), this trajectory is unusable for training. We:
    #  - set `sample.remove_sample = True` so miles zeros the loss_mask
    #    (ray/rollout.py:389) — no gradient contribution from this sample
    #  - set `sample.status = Sample.Status.FAILED` for downstream visibility
    #  - assign a placeholder `sample.reward = 0.0` so torch.tensor in the
    #    dynamic filter (check_reward_nonzero_std) doesn't crash on None.
    #    The 0.0 is purely a structural placeholder; the zeroed loss_mask
    #    is what actually prevents this sample from updating model params.
    raw_reward = response.get("reward")
    if raw_reward is None:
        logger.warning(
            "[camel_terminal] env_service returned reward=None for %s — "
            "marking sample invalid (remove_sample=True, status=FAILED)",
            uid,
        )
        sample.reward = 0.0
        sample.remove_sample = True
        sample.status = Sample.Status.FAILED
    else:
        try:
            sample.reward = float(raw_reward)
        except (TypeError, ValueError):
            logger.warning(
                "[camel_terminal] non-numeric reward %r for %s — "
                "marking sample invalid (remove_sample=True, status=FAILED)",
                raw_reward, uid,
            )
            sample.reward = 0.0
            sample.remove_sample = True
            sample.status = Sample.Status.FAILED

    # Stash diagnostic extras into metadata for downstream consumers /
    # debugging. The reward_func is now a fallback only (env_service errored
    # out, no reward returned).
    sample.metadata = sample.metadata or {}
    sample.metadata.update(
        {
            "camel_uid": uid,
            "camel_trial_name": trial_name,
            "camel_run_info": run_info,
            "camel_step_seconds": dt,
            "camel_env_service_reward": raw_reward,
        }
    )

    # Self-validate before returning — catch shape bugs at the source rather
    # than letting miles' loss-fn blow up later.
    try:
        sample.validate()
    except AssertionError as e:
        logger.error(
            "[camel_terminal] Sample.validate() failed for %s: %s", uid, e
        )
        sample.status = Sample.Status.FAILED

    logger.info(
        "[camel_terminal] %s: tokens=%d response_length=%d "
        "loss_mask_sum=%d status=%s in %.1fs",
        uid,
        len(sample.tokens),
        sample.response_length,
        sum(sample.loss_mask) if sample.loss_mask else 0,
        sample.status.value,
        dt,
    )

    return sample
