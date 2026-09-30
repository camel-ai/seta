"""Reward, dynamic-sampling filters, and rollout metrics for Harbor rollouts.

The generate function is Miles' stock
``miles.rollout.generate_hub.agentic_tool_call.generate`` with
``--custom-agent-function-path common.harbor_agent.run``. The Harbor
agent server verifies each trial, so the reward is already in
``sample.metadata["reward"]`` whatever the task type.

Components:
  - reward_func (``--custom-rm-path common.harbor_rollout.reward_func``):
    reads the pre-computed reward from sample metadata.
  - check_no_infra_failure / check_no_infra_failure_and_nonzero_std
    (``--dynamic-sampling-filter-path``): drop groups hit by infrastructure
    failures, optionally also zero-variance groups.
  - RolloutFn (``--rollout-function-path common.harbor_rollout.RolloutFn``):
    Miles' inference rollout plus aggregated agent timing/turn metrics.
"""

import logging

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# -- Reward --


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Reward is pre-computed by the agent environment during generate().

    Handles both single-sample calls (from ``async_rm``) and batched calls
    (from ``batched_async_rm`` when ``--custom-rm-path`` is set).
    """
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


# -- Dynamic-sampling filter --

# Agent-server exit statuses that describe the infrastructure or the harness,
# not the policy: the trial never reached verification. Scoring them 0 teaches
# the policy that its actions failed when a pod, tunnel, or an operator flush
# failed instead. In one 185-sample training step we saw 68 AgentError and
# 56 Flushed samples, which would all have been trained on as failures.
# TimeLimitExceeded and SequenceLengthLimitExceeded stay: the agent-phase
# budget and the response cap are part of the evaluated contract.
INFRA_EXIT_STATUSES = frozenset({"AgentError", "Flushed", "Unknown"})


def check_no_infra_failure(args, samples, **kwargs):
    """Reject a group if any sample was aborted or ended in an infrastructure exit status.

    Extends Miles' ``check_no_aborted`` (which sees only ``Sample.Status.ABORTED``)
    with the harness's own outcome classification carried in
    ``sample.metadata["exit_status"]``. A rejected group is refilled by the
    fully-async driver exactly like an aborted one.
    """
    from miles.rollout.filter_hub.dynamic_sampling_filters import DynamicFilterOutput, check_no_aborted

    aborted = check_no_aborted(args, samples, **kwargs)
    if not aborted.keep:
        return aborted
    flat = []
    for group in samples:
        flat.extend(group if isinstance(group, list) else [group])
    for s in flat:
        status = (getattr(s, "metadata", None) or {}).get("exit_status")
        if status in INFRA_EXIT_STATUSES:
            return DynamicFilterOutput(keep=False, reason=f"group_has_infra_failure:{status}")
    return DynamicFilterOutput(keep=True)


def check_no_infra_failure_and_nonzero_std(args, samples, **kwargs):
    """Reject a group that had an infrastructure failure OR carries no reward spread.

    GRPO's baseline is the group mean, so a group whose members all scored the
    same produces an advantage of exactly zero for every member and contributes
    nothing to the gradient. Miles supplies ``check_reward_nonzero_std`` for
    that test; this composes it with the harness's infrastructure check so a
    rejected group is refilled by the fully-async driver either way.

    Order matters: the infrastructure check runs first, so a group destroyed by
    AgentError is reported as an infrastructure failure rather than as a
    zero-spread group it only looks like.

    Cost: refill is not free. On a dataset where a share ``d`` of groups are
    degenerate, filling one accepted batch samples about ``1/(1-d)`` times as
    many trajectories. Measure ``d`` before enabling this on a hard dataset --
    at d=0.95 the rollout cost per optimizer step rises roughly twentyfold.
    """
    from miles.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std

    infra = check_no_infra_failure(args, samples, **kwargs)
    if not infra.keep:
        return infra
    return check_reward_nonzero_std(args, samples, **kwargs)


# -- Agent Metrics Aggregation --


def _collect_values(all_metrics: list[dict], key: str) -> list[float]:
    return [m.get(key, 0) for m in all_metrics]


def _agg_mean(metrics: dict, all_metrics: list[dict], keys: list[str], prefix: str = "agent/", suffix: str = "_mean"):
    for key in keys:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"{prefix}{key}{suffix}"] = sum(values) / len(values)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging."""
    all_metrics = [
        s.metadata.get("agent_metrics", {})
        for s in samples
        if hasattr(s, "metadata") and s.metadata and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return {}

    metrics = {}

    for key in ["turns", "tool_calls"]:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    _agg_mean(metrics, all_metrics, ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"])
    _agg_mean(metrics, all_metrics, ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"], suffix="")
    _agg_mean(metrics, all_metrics, ["model_time_ratio", "env_time_ratio", "eval_time_ratio"], suffix="")

    values = _collect_values(all_metrics, "total_time")
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    return metrics


# -- Rollout Function --


class RolloutFn(InferenceRolloutFn):
    """Rollout function with agent metrics aggregation."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        all_samples = []
        for group in output.samples:
            if isinstance(group, list):
                all_samples.extend(group)
            else:
                all_samples.append(group)

        agent_metrics = aggregate_agent_metrics(all_samples)
        if agent_metrics:
            metrics = output.metrics or {}
            metrics.update(agent_metrics)
            output.metrics = metrics
            logger.info(f"Agent metrics for rollout {input.rollout_id}: {agent_metrics}")

        return output
