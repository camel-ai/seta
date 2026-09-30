"""Miles reward function for the seta env_service rollout.

    --custom-rm-path common.env_service_reward.reward_func

Miles calls it only for samples whose ``sample.reward`` is still None.

* Session-server path (``common.env_service_agent``): Miles' agentic generate
  function merges the agent's return dict into ``sample.metadata`` and never sets
  ``sample.reward``, so this is the reward path for every sample. The agent returns
  ``reward`` (the env_service's verifier reward, in [0, 1]) for every completed call
  and nothing at all when the call failed, so a present key means a valid trajectory.
* Direct path (``common.env_service_generate``): generate sets ``sample.reward``
  itself; this function only sees samples for which the env_service returned no
  usable reward. It reads ``camel_env_service_reward`` from the metadata.

A sample without a usable reward is invalid: it is marked ``remove_sample=True``
(Miles zeroes its loss mask, so it contributes no gradient) and ``FAILED``. The
returned 0.0 is only a placeholder so that reward tensors never hold None.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    md = sample.metadata or {}
    # A real reward of 0.0 is valid, so test for None rather than truthiness.
    raw = md.get("reward")
    if raw is None:
        raw = md.get("camel_env_service_reward")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "[env_service_reward] non-numeric metadata reward %r for index=%s; marking invalid",
                raw,
                sample.index,
            )

    logger.warning(
        "[env_service_reward] no env_service reward for index=%s; marking the sample invalid "
        "(remove_sample=True, status=FAILED). It will not update the model; the 0.0 reward "
        "is a placeholder.",
        sample.index,
    )
    sample.remove_sample = True
    sample.status = Sample.Status.FAILED
    return 0.0
