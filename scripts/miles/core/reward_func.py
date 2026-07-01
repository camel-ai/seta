"""Fallback reward function for miles — only invoked when generate didn't
set ``sample.reward`` (e.g. env_service errored out or returned reward=None).

Wired via ``--custom-rm-path``. Miles' rollout layer
([sglang_rollout.py:266-277](/root/miles/miles/rollout/sglang_rollout.py)) only calls
this function for samples where ``sample.reward is None`` — for the happy
path our ``generate_with_camel.generate`` sets ``sample.reward`` directly
from env_service's pass_ratio result, so this function is bypassed entirely.

The convention in env_service is that ``reward`` is the per-trajectory
``pass_ratio`` of the verifier's pytest tests (set via env.reward_fn in
seta_env_config.yaml — values typically in [0.0, 1.0]).
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Fallback path: env_service didn't return a usable reward.

    Reaching here means generate_with_camel set sample.reward = None
    (env_service /step failed, returned no payload, or signaled a
    non-recoverable error). Per the invalid-sample contract:

      - Mark sample.remove_sample = True so miles zeros the loss_mask
        (ray/rollout.py:389), i.e. this sample contributes 0 gradient.
      - Set sample.status = Sample.Status.FAILED for visibility.
      - Return 0.0 as a structural placeholder so torch.tensor in the
        dynamic-sample filter doesn't crash on None. The actual training
        impact is gated by the zeroed loss_mask, not by this 0.0.

    Reward keys, in priority order:
      - ``reward``: the SESSION-SERVER path. agentic_tool_call.generate merges
        seta_agent_function.run()'s return dict into sample.metadata
        (agentic_tool_call.py:113 ``s.metadata.update(agent_metadata)``) and
        never sets sample.reward, so this RM is the real reward path here, not
        a fallback. seta_agent_function returns reward in [0.0, 1.0] for any
        completed trajectory and only returns None (no metadata merged) on a
        hard env_service failure — so "key present" == "valid trajectory".
      - ``camel_env_service_reward``: the legacy custom-backend path
        (generate_with_camel.generate), kept for compatibility.
    A real reward of 0.0 is valid, so test ``is not None`` (not truthiness).
    """
    md = sample.metadata or {}
    raw = md.get("reward")
    if raw is None:
        raw = md.get("camel_env_service_reward")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "[camel_terminal.reward_func] non-numeric metadata reward %r "
                "for index=%s — marking invalid",
                raw, sample.index,
            )

    logger.warning(
        "[camel_terminal.reward_func] no env_service reward for index=%s — "
        "marking sample invalid (remove_sample=True, status=FAILED). "
        "This trajectory will NOT update model params; the 0.0 reward "
        "is a placeholder for torch.tensor.",
        sample.index,
    )
    sample.remove_sample = True
    sample.status = Sample.Status.FAILED
    return 0.0
