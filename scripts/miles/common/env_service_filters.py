"""Group filter for Miles dynamic sampling on the seta env_service rollout.

    --dynamic-sampling-filter-path common.env_service_filters.filter_group

One filter function whose predicates are switched on by environment variables. The
launcher must forward those variables to the Ray rollout workers (``extra_env_vars``
of ``execute_train``); unset, every predicate is off and no group is dropped.
Predicates run in the order below; the first that fires returns ``keep=False`` with a
reason, and Miles counts drops per reason.

A sample counts as env-failed when it is ``remove_sample`` or its status is ABORTED
or FAILED: the sandbox or the env_service failed, so its reward says nothing about
the policy.

Environment variables
---------------------
GROUP_FILTER_MAX_ENV_FAILURES       default 1000000000 (off)
    Drop the group when at least this many samples are env-failed. 1 admits only
    groups in which every sample ran, so infrastructure failures never enter the
    GRPO baseline.
GROUP_FILTER_MAX_ENV_FAILURE_RATE   default 1.1 (off)
    Drop when ``env_failed / group_size >= rate``, e.g. 0.5.
GROUP_FILTER_PASS_REWARD            default 1.0
GROUP_FILTER_MAX_PASS_RATE          default 1.1 (off)
    Drop when the share of valid samples with reward >= PASS_REWARD is >= this, e.g.
    0.875 (14 of 16): a group at the reward ceiling has no learning signal.
GROUP_FILTER_FAIL_REWARD            default 0.0
GROUP_FILTER_MAX_FAIL_RATE          default 1.1 (off)
    Drop when the share of valid samples with reward <= FAIL_REWARD is >= this.
GROUP_FILTER_MIN_REWARD_STD         default -1.0 (off)
    Drop when the population std of the valid rewards is below this. A tiny value
    such as 1e-8 drops exactly the groups where every sample got the same reward,
    i.e. zero GRPO advantage.

Rates and the std use valid samples only; a group with no valid sample is dropped.
Under a sandbox-provider outage most groups get dropped and the rollout slows down
rather than training on failures; watch the drop counters.
"""

import os
import statistics

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

ENV_FAILURE_STATUSES = {Sample.Status.ABORTED, Sample.Status.FAILED}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _is_env_failed(s: Sample) -> bool:
    return s.remove_sample or s.status in ENV_FAILURE_STATUSES


def filter_group(args, samples, **kwargs):
    """Return a DynamicFilterOutput; predicates are evaluated in a fixed order."""
    n = len(samples)
    n_env_failed = sum(1 for s in samples if _is_env_failed(s))

    # 0. absolute number of env-failed samples
    max_env_fail_count = int(float(os.environ.get("GROUP_FILTER_MAX_ENV_FAILURES", "1000000000")))
    if n_env_failed >= max_env_fail_count:
        return DynamicFilterOutput(keep=False, reason=f"env_failure_{n_env_failed}_of_{n}")

    # 1. env-failure rate over the whole group
    max_env_fail = _env_float("GROUP_FILTER_MAX_ENV_FAILURE_RATE", 1.1)
    if n > 0 and n_env_failed / n >= max_env_fail:
        return DynamicFilterOutput(keep=False, reason=f"env_failure_{n_env_failed}_of_{n}")

    # The remaining predicates look at valid samples only.
    valid_rewards = [s.get_reward_value(args) for s in samples if not _is_env_failed(s)]
    n_valid = len(valid_rewards)
    if n_valid == 0:
        return DynamicFilterOutput(keep=False, reason=f"env_failure_{n_env_failed}_of_{n}")

    # 2. saturated: nearly every valid sample passed
    pass_threshold = _env_float("GROUP_FILTER_PASS_REWARD", 1.0)
    max_pass = _env_float("GROUP_FILTER_MAX_PASS_RATE", 1.1)
    n_pass = sum(1 for r in valid_rewards if r >= pass_threshold)
    if n_pass / n_valid >= max_pass:
        return DynamicFilterOutput(keep=False, reason=f"high_pass_rate_{n_pass}_of_{n_valid}")

    # 3. collapsed: nearly every valid sample failed
    fail_threshold = _env_float("GROUP_FILTER_FAIL_REWARD", 0.0)
    max_fail = _env_float("GROUP_FILTER_MAX_FAIL_RATE", 1.1)
    n_fail = sum(1 for r in valid_rewards if r <= fail_threshold)
    if n_fail / n_valid >= max_fail:
        return DynamicFilterOutput(keep=False, reason=f"high_fail_rate_{n_fail}_of_{n_valid}")

    # 4. reward spread too small
    min_std = _env_float("GROUP_FILTER_MIN_REWARD_STD", -1.0)
    if min_std >= 0:
        rstd = statistics.pstdev(valid_rewards) if n_valid > 1 else 0.0
        if rstd < min_std:
            return DynamicFilterOutput(keep=False, reason=f"low_std_{rstd:.3f}")

    return DynamicFilterOutput(keep=True)
