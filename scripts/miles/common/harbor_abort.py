"""Abort-aware wrapper around Miles' ``agentic_tool_call.generate``.

Use as ``--custom-generate-function-path common.harbor_abort.generate`` on
Miles versions whose generate loop does not call the agent function's ``abort``
hook itself.

Why: in colocated agentic RL, once generate() has its rollout_batch_size groups
it aborts the rest (``GenerateState.aborted = True``). The surplus oversampled
trajectories are still running as Harbor trials and keep sending turns to the
session server, so the engines never drain and the offload's flush_cache()
times out. Miles' engine-level abort does not stop Harbor trials.

Fix (no Miles source edits): when ``GenerateState.aborted`` flips, post
``/flush`` for this session server so Harbor cancels its in-flight trials.

It also backfills zero routing/indexer replay tensors for empty turns under
rollout routing replay (R3); see ``_backfill_missing_replay``.

Every log line is tagged ``[harbor-abort]``: when /flush fires, how many trials
it cancelled, and each trajectory's final status, so you can confirm that only
surplus trajectories are aborted.
"""

import asyncio
import logging
import os
import time

import numpy as np

from miles.rollout.generate_hub.agentic_tool_call import generate as _inner_generate
from miles.utils.http_utils import post

logger = logging.getLogger("harbor_abort")
logger.setLevel(logging.INFO)


def _backfill_missing_replay(sample, args):
    """Backfill zero MoE-routing/indexer replay tensors on a sample that has none.

    WHY: an empty / degenerate agent turn (model emits ~0 new tokens) captures no
    routing, so sglang omits ``routed_experts`` and the merged Sample decodes to
    ``rollout_routed_experts=None``. Under --use-rollout-routing-replay, miles calls
    fill_replay_data() unconditionally (actor.py) which RAISES "rollout_routed_experts
    is required in rollout_data for replay" when the key is missing, and R3 replay
    also crashes on ``torch.from_numpy(None)``. The upstream merge itself hits
    ``None.shape`` in _merge_sample_pair (guarded in the Miles checkout). So: fill a
    length-matching (``len(tokens)-1``) zero placeholder to keep the field present and
    Sample.validate()-consistent, and set ``remove_sample=True`` so the sample
    contributes ZERO gradient -- the zero routing is never actually trained on. Mirrors
    the env_service/camel generate path's placeholder handling.
    """
    n = max(len(getattr(sample, "tokens", None) or []) - 1, 0)
    if n <= 0:
        return False
    changed = False
    if getattr(args, "use_rollout_routing_replay", False) and getattr(
        sample, "rollout_routed_experts", None
    ) is None:
        num_layers = int(getattr(args, "num_layers", 0) or 43)
        moe_topk = int(getattr(args, "moe_router_topk", 0) or 6)
        sample.rollout_routed_experts = np.zeros((n, num_layers, moe_topk), dtype=np.int32)
        changed = True
    if getattr(args, "use_rollout_indexer_replay", False) and getattr(
        sample, "rollout_indexer_topk", None
    ) is None:
        idx_layers = int(getattr(args, "num_indexer_layers", 0) or getattr(args, "num_layers", 0) or 43)
        idx_topk = int(getattr(args, "index_topk", 0) or 512)
        sample.rollout_indexer_topk = np.zeros((n, idx_layers, idx_topk), dtype=np.int32)
        changed = True
    if changed:
        sample.remove_sample = True
    return changed


async def generate(input):  # noqa: ANN001  (miles GenerateFnInput)
    state = input.state
    args = input.args
    agent_url = os.getenv("AGENT_SERVER_URL")
    ssid = getattr(args, "session_server_instance_id", None)
    traj = getattr(input.sample, "index", getattr(input.sample, "uid", "?"))
    t0 = time.monotonic()

    inner = asyncio.ensure_future(_inner_generate(input))
    flushed = False
    while not inner.done():
        if getattr(state, "aborted", False):
            flushed = True
            # POST /flush on EVERY abort wave. No cross-wave dedup: state is a
            # process-wide singleton (id never changes) and state.aborted resets to
            # False each rollout, so a dedup keyed on the state object would only ever
            # fire once for the whole run -> later rollouts wouldn't cancel their
            # surplus -> engine refill -> flush_cache timeout crash. /flush is
            # idempotent (cancels in-flight for this ssid; later calls cancel 0), and
            # the first /flush cascades-cancels the other surplus trajectories so only
            # 1-2 actually fire per wave.
            if agent_url and ssid:
                try:
                    logger.info(
                        "[harbor-abort] batch full -> POST /flush ssid=%s (traj=%s)",
                        ssid, traj,
                    )
                    resp = await post(f"{agent_url}/flush", {"session_server_instance_id": ssid})
                    logger.info("[harbor-abort] /flush ok: cancelled=%s resp=%s",
                                (resp or {}).get("cancelled"), resp)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[harbor-abort] /flush FAILED: %r", e)
            break
        await asyncio.sleep(0.3)

    out = await inner
    # Backfill missing routing/indexer replay tensors (empty-turn -> None) so miles'
    # fill_replay_data / R3 replay never sees None. remove_sample=True => zero gradient.
    try:
        _s = out.samples
        for _one in (_s if isinstance(_s, list) else [_s]):
            if _backfill_missing_replay(_one, args):
                logger.warning("[harbor-abort] traj=%s backfilled missing replay tensors "
                               "(empty turn) -> remove_sample=True", traj)
    except Exception as e:  # noqa: BLE001
        logger.warning("[harbor-abort] replay-backfill failed: %r", e)
    # --- per-trajectory outcome (so we can verify we abort only surplus) ---
    try:
        s = out.samples
        s0 = s[0] if isinstance(s, list) else s
        status = getattr(getattr(s0, "status", None), "name", str(getattr(s0, "status", None)))
        dt = time.monotonic() - t0
        tag = "FLUSHED" if flushed else "NORMAL"
        logger.info("[harbor-abort] traj=%s outcome=%s status=%s dt=%.1fs", traj, tag, status, dt)
        # RED FLAG: a flushed trajectory that still came back COMPLETED means we may
        # have aborted a group that could have been kept. Expect this to be ~0.
        if flushed and status == "COMPLETED":
            logger.warning("[harbor-abort] *** OVER-ABORT? traj=%s was FLUSHED but COMPLETED ***", traj)
    except Exception as e:  # noqa: BLE001
        logger.warning("[harbor-abort] outcome-log failed: %r", e)
    return out


def _add_arguments(parser):
    # delegate to the wrapped module's arg registration (registers
    # --custom-agent-function-path, --generate-multi-samples, --max-seq-len)
    from miles.rollout.generate_hub.agentic_tool_call import _add_arguments as _inner_args
    _inner_args(parser)


# miles (arguments.py:1870) calls `fn.add_arguments(parser)` on the loaded generate
# function, exactly as agentic_tool_call does (`generate.add_arguments = _add_arguments`).
generate.add_arguments = _add_arguments
