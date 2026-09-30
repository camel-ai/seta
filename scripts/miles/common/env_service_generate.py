"""Miles generate function that runs one trajectory on the seta env_service directly.

    --custom-generate-function-path common.env_service_generate.generate

The alternative to the session-server path (``common.env_service_agent``): no Miles
session server is involved. ``generate`` posts the task and the SGLang router URL to
the env_service (``POST /step``). The env_service runs the CAMEL agent against SGLang
with a model backend that records the tokens itself (a ``model_platform`` that dumps
token-in/token-out state, e.g. ``sglang_deepseek_v4``) and returns
``response["sample"]`` already in Miles ``Sample`` shape:

    tokens, response_length, loss_mask (response tokens only), rollout_log_probs,
    response, and optionally the rollout routing-replay buffers
    rollout_routed_experts_b64 / _num_layers / _topk and
    rollout_indexer_topk_b64 / rollout_indexer_num_layers / rollout_indexer_topk_k
    (base64 int32, shape (len(tokens) - 1, layers, topk)).

``sample.reward`` is set from the env_service's reward, so the custom reward function
only runs for samples without one.

Environment (forwarded to the Ray rollout workers by the launcher):
    CAMEL_ENV_SERVICE_URL   env_service base URL (default http://127.0.0.1:8002)
    CAMEL_DATASET_NAME      required: env_service resolves DATASET_ROOT/<name>/<task>
    CAMEL_TRIAL_NAME        required: groups the trial folders of one run
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

from common.env_service_agent import prompt_to_instruction

logger = logging.getLogger(__name__)


def _populate_failed_sample_placeholders(sample: Sample, args: Namespace) -> None:
    """Give a failed trajectory minimal valid token and replay fields.

    When Miles assembles the training batch it decides from the FIRST sample alone
    whether ``rollout_routed_experts`` / ``rollout_indexer_topk`` /
    ``rollout_log_probs`` are included. A failed first sample with None there drops
    the field for the whole batch, and routing replay then fails with
    "rollout_routed_experts is required in rollout_data for replay". A length-1 zero
    placeholder keeps the field present; the sample carries no gradient because its
    loss mask is zero (and the reward function marks it remove_sample).
    """
    sample.tokens = [0, 0]  # one prompt sentinel + one response sentinel
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    if getattr(args, "use_rollout_routing_replay", False):
        num_layers = int(getattr(args, "num_layers", 0) or 43)
        moe_topk = int(getattr(args, "moe_router_topk", 0) or 6)
        sample.rollout_routed_experts = np.zeros((1, num_layers, moe_topk), dtype=np.int32)
    if getattr(args, "use_rollout_indexer_replay", False):
        indexer_layers = int(getattr(args, "num_indexer_layers", 0) or 43)
        indexer_topk = int(getattr(args, "index_topk", 0) or 512)
        sample.rollout_indexer_topk = np.zeros((1, indexer_layers, indexer_topk), dtype=np.int32)


def _decode_routing(
    b64: str | None,
    expected_token_count: int,
    num_layers: int,
    topk: int,
    field: str,
) -> np.ndarray | None:
    """Decode SGLang's base64 int32 routing buffer to shape (T - 1, layers, topk)."""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64.encode("ascii"))
        arr = np.frombuffer(raw, dtype=np.int32)
        expected_entries = expected_token_count - 1
        return arr.reshape(expected_entries, num_layers, topk)
    except Exception as e:
        logger.warning("[env_service_generate] %s decode/reshape failed: %s", field, e)
        return None


def _resolve_env_service_url() -> str:
    return os.getenv("CAMEL_ENV_SERVICE_URL", "http://127.0.0.1:8002")


def _resolve_dataset_name() -> str:
    """CAMEL_DATASET_NAME from the environment; there is deliberately no default.

    A wrong dataset name makes the env_service look tasks up in the wrong directory,
    and those trajectories then train as silent zero rewards.
    """
    v = os.getenv("CAMEL_DATASET_NAME")
    if not v:
        raise RuntimeError(
            "CAMEL_DATASET_NAME is not set in this Ray worker. The launcher must forward it "
            "through the Ray runtime env (extra_env_vars of execute_train)."
        )
    return v


def _resolve_trial_name() -> str:
    """CAMEL_TRIAL_NAME from the environment; same forwarding requirement."""
    v = os.getenv("CAMEL_TRIAL_NAME")
    if not v:
        raise RuntimeError(
            "CAMEL_TRIAL_NAME is not set in this Ray worker. The launcher must forward it "
            "through the Ray runtime env (extra_env_vars of execute_train)."
        )
    return v


def _coerce_metadata(md) -> dict:
    """sample.metadata may be a dict or, from parquet, a JSON string."""
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
    """A stable task name from the sample metadata; falls back to the sample index."""
    md = _coerce_metadata(sample.metadata)
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
    """Roll out one trajectory on the env_service; returns ``sample`` mutated in place."""
    env_service_url = _resolve_env_service_url()
    dataset_name = _resolve_dataset_name()
    trial_name = _resolve_trial_name()

    task_name = _task_name_from_sample(sample)
    instruction = prompt_to_instruction(sample.prompt)
    uid = f"{task_name}_{trial_name}_{sample.index if sample.index is not None else 0}"

    # Miles picks the SGLang router host/port at startup; the env_service replaces the
    # model URL of its config with this one for every request.
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
        "model_url": sglang_url,
        "model_api_key": "dummy",
        "dataset_name": dataset_name,
        "task_name": task_name,
        "trial_name": trial_name,
        # Miles launches SGLang with routing capture when replay is on, but each request
        # must also ask for it, or the response carries no routing data to replay.
        "return_routed_experts": bool(getattr(args, "use_rollout_routing_replay", False)),
        "return_indexer_topk": bool(getattr(args, "use_rollout_indexer_replay", False)),
    }

    t0 = time.time()
    try:
        response = await post(f"{env_service_url}/step", request)
    except Exception as e:
        logger.error("[env_service_generate] /step failed for %s: %s", uid, e, exc_info=True)
        sample.status = Sample.Status.FAILED
        _populate_failed_sample_placeholders(sample, args)
        return sample
    dt = time.time() - t0

    err = response.get("error")
    if err:
        logger.warning("[env_service_generate] env_service error for %s: %s", uid, err)

    sample_data = response.get("sample")
    if sample_data is None:
        # The trajectory failed before the model produced anything (e.g. the sandbox
        # could not be created).
        run_info = response.get("run_info") or {}
        err_info = run_info.get("error_info") or {}
        logger.warning(
            "[env_service_generate] no sample payload for %s (stage=%s, err=%s)",
            uid,
            err_info.get("stage"),
            err_info.get("error_message", "")[:200],
        )
        sample.status = Sample.Status.FAILED
        _populate_failed_sample_placeholders(sample, args)
        return sample

    sample.tokens = list(sample_data["tokens"])
    sample.response_length = int(sample_data["response_length"])
    sample.loss_mask = list(sample_data["loss_mask"])
    sample.rollout_log_probs = [float(x) for x in sample_data["rollout_log_probs"]]
    sample.response = sample_data.get("response") or ""

    routing_token_count = int(sample_data.get("rollout_routing_token_count") or len(sample.tokens))

    # Replay buffers must describe their own shape (num_layers, topk), taken from the
    # serving model. Guessing per-model constants would silently train on a wrong
    # reshape, so a missing shape, or a missing buffer for a non-empty trajectory,
    # fails loudly.
    def _require_dim(sample_key: str, field: str) -> int:
        v = sample_data.get(sample_key)
        if not v:
            raise ValueError(
                f"[env_service_generate] {uid}: replay is enabled but the env_service did not "
                f"return '{sample_key}' for {field}; the buffer cannot be decoded without its "
                f"shape. The serving model backend must report the routing dimensions."
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
                    f"[env_service_generate] {uid}: {field} buffer was present but failed to "
                    f"decode/reshape (see the warning above)."
                )
            return arr
        if sample.response_length > 0:
            # Tokens were generated but no buffer came back: the engine was launched
            # without the capture flag or the request did not ask for it.
            raise ValueError(
                f"[env_service_generate] {uid}: replay is enabled and the trajectory generated "
                f"{sample.response_length} tokens, but the env_service returned no '{b64_key}'. "
                f"Launch SGLang with the capture flag (--use-rollout-routing-replay / "
                f"--use-rollout-indexer-replay)."
            )
        # Empty generation: no routing exists; the guard below backfills a placeholder.
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

    # An empty response has no routing data. Such a sample can still land in a kept
    # group, and routing replay would then fail on a None buffer, so backfill the
    # placeholder and remove the sample from the gradient.
    if getattr(args, "use_rollout_routing_replay", False) and sample.rollout_routed_experts is None:
        logger.warning(
            "[env_service_generate] %s: completed with no routing data (response_length=%d); "
            "backfilling a placeholder and setting remove_sample=True",
            uid,
            sample.response_length,
        )
        _populate_failed_sample_placeholders(sample, args)
        sample.remove_sample = True

    run_info = response.get("run_info") or {}
    summary = run_info.get("agent_summary") or {}

    # This path bypasses Miles' own SGLang client, so hand over the prefix-cache
    # counters the env_service collected; otherwise prefix_cache_hit_rate reads 0.
    # prefix_cache_info.add() rather than update_from_meta_info(), which also expects
    # fields (finish_reason, spec info) this payload does not carry.
    cache_stats = run_info.get("cache_stats") or {}
    if cache_stats.get("prompt_tokens"):
        sample.prefix_cache_info.add(cache_stats)

    # Trajectory status from the agent's termination reason.
    term = summary.get("important_termination_reason") or summary.get("termination_reason")
    if term in ("task_finished", "stop", None):
        sample.status = Sample.Status.COMPLETED
    elif term in ("max_iteration", "context_length_exceeded", "length"):
        sample.status = Sample.Status.TRUNCATED
    elif term in ("abort",):
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED

    # The reward is known here, so set it and skip the reward function (Miles only
    # calls it for samples whose reward is None). Without a usable reward the sample
    # is invalid: remove_sample=True (zero loss mask, no gradient), FAILED, and a 0.0
    # placeholder so reward tensors never hold None.
    raw_reward = response.get("reward")
    if raw_reward is None:
        logger.warning(
            "[env_service_generate] env_service returned reward=None for %s; marking the sample invalid",
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
                "[env_service_generate] non-numeric reward %r for %s; marking the sample invalid",
                raw_reward,
                uid,
            )
            sample.reward = 0.0
            sample.remove_sample = True
            sample.status = Sample.Status.FAILED

    # Diagnostics for common.env_service_metrics and debugging.
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

    # Catch shape errors here rather than in the loss function.
    try:
        sample.validate()
    except AssertionError as e:
        logger.error("[env_service_generate] Sample.validate() failed for %s: %s", uid, e)
        sample.status = Sample.Status.FAILED

    logger.info(
        "[env_service_generate] %s: tokens=%d response_length=%d loss_mask_sum=%d status=%s in %.1fs",
        uid,
        len(sample.tokens),
        sample.response_length,
        sum(sample.loss_mask) if sample.loss_mask else 0,
        sample.status.value,
        dt,
    )

    return sample
