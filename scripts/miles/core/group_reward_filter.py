"""Group-level reward-distribution filter for miles dynamic sampling.

Registered via miles' ``--dynamic-sampling-filter-path``. One filter
function, several composable predicates configured by env vars. Default
behavior is pass-through (no group dropped) — every predicate is
disabled by an off-sentinel default.

Usage
-----
Activate by passing the filter path to miles train:

    --dynamic-sampling-filter-path \\
        scripts.miles.group_reward_filter.filter_group

Configure predicates via env vars (set them in the launcher; they are
auto-propagated into Ray actors by run_deepseek_v4.py — see the
``GROUP_*`` env-var propagation block there). Predicates evaluate in
order; the first one that fires returns ``keep=False`` with a reason
string. miles' MetricGatherer auto-publishes a per-reason counter to
wandb under ``rollout/dynamic_filter/drop_<reason>``.

Env vars
--------
GROUP_FILTER_MAX_ENV_FAILURE_RATE   default 1.1 (off)
    Drop the group if ``n_env_failed / n_total >= this``.
    Recommended: 0.5 (paper's threshold — drop when half-or-more are
    env-failed). Compose with group_reward_post_process.py to give the
    surviving (minority-invalid) groups correct GRPO statistics.

GROUP_FILTER_PASS_REWARD            default 1.0
    Reward at-or-above this counts as a "full pass" for the
    GROUP_FILTER_MAX_PASS_RATE predicate.

GROUP_FILTER_MAX_PASS_RATE          default 1.1 (off)
    Drop if ``count(r >= GROUP_FILTER_PASS_REWARD) / n_valid >= this``.
    Catches saturated groups where most samples are at the reward
    ceiling — GRPO has no learning signal. Recommended: 0.875 (14/16).

GROUP_FILTER_FAIL_REWARD            default 0.0
    Reward at-or-below this counts as a "full fail" for the
    GROUP_FILTER_MAX_FAIL_RATE predicate.

GROUP_FILTER_MAX_FAIL_RATE          default 1.1 (off)
    Drop if ``count(r <= GROUP_FILTER_FAIL_REWARD) / n_valid >= this``.
    Catches stuck groups where nearly every sample failed.
    Recommended: 0.875 (14/16).

GROUP_FILTER_MIN_REWARD_STD         default -1.0 (off)
    Drop if ``std(valid rewards) < this``. Catches near-zero-variance
    groups not captured by the pass/fail-rate predicates (e.g. all
    samples cluster around the same partial-credit reward).
    Recommended: 0.05 if needed.

Notes
-----
- Ratios are over VALID samples (env-failed excluded from numerator AND
  denominator) EXCEPT the env-failure-rate predicate, which uses the
  full group as denominator.
- A group with ``n_valid == 0`` (all env-failed) is treated as
  env-failure-majority and dropped.
- Compose with group_reward_post_process.py so surviving partial-invalid
  groups get GRPO mean/std computed over valid samples only.
- Drops cause miles to oversample replacements. Under cluster-wide env
  outages (e.g. Daytona quota exhaustion) oversampling will also fail;
  consider also enabling group_reward_log.py to monitor the env_failure
  rate and stop the run if it stays high.
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
    """Return DynamicFilterOutput. Predicates evaluated in fixed order."""
    n = len(samples)
    n_env_failed = sum(1 for s in samples if _is_env_failed(s))

    # 0. absolute env-failure COUNT — the strict "< 1 failure per group" rule.
    #    GROUP_FILTER_MAX_ENV_FAILURES = N drops the group if n_env_failed >= N.
    #    Set to 1 to require a FULLY-clean group (every sample's env came up and
    #    the agent ran) — any single env failure invalidates the whole group so
    #    it is recycled (re-rolled), never counted toward the batch. This makes a
    #    training step admit ONLY fully-valid task groups. Default 10**9 = off.
    #    Stricter than (and composes with) the rate predicate below.
    max_env_fail_count = int(float(os.environ.get("GROUP_FILTER_MAX_ENV_FAILURES", "1000000000")))
    if n_env_failed >= max_env_fail_count:
        return DynamicFilterOutput(
            keep=False, reason=f"env_failure_{n_env_failed}_of_{n}",
        )

    # 1. env-failure rate (denominator = whole group)
    max_env_fail = _env_float("GROUP_FILTER_MAX_ENV_FAILURE_RATE", 1.1)
    if n > 0 and n_env_failed / n >= max_env_fail:
        return DynamicFilterOutput(
            keep=False, reason=f"env_failure_{n_env_failed}_of_{n}",
        )

    # Remaining predicates operate over VALID samples.
    valid_rewards = [
        s.get_reward_value(args) for s in samples if not _is_env_failed(s)
    ]
    n_valid = len(valid_rewards)
    if n_valid == 0:
        # No valid samples → can't evaluate; treat as catastrophic drop.
        return DynamicFilterOutput(
            keep=False, reason=f"env_failure_{n_env_failed}_of_{n}",
        )

    # 2. saturation (high pass rate)
    pass_threshold = _env_float("GROUP_FILTER_PASS_REWARD", 1.0)
    max_pass = _env_float("GROUP_FILTER_MAX_PASS_RATE", 1.1)
    n_pass = sum(1 for r in valid_rewards if r >= pass_threshold)
    if n_pass / n_valid >= max_pass:
        return DynamicFilterOutput(
            keep=False, reason=f"high_pass_rate_{n_pass}_of_{n_valid}",
        )

    # 3. collapse (high fail rate)
    fail_threshold = _env_float("GROUP_FILTER_FAIL_REWARD", 0.0)
    max_fail = _env_float("GROUP_FILTER_MAX_FAIL_RATE", 1.1)
    n_fail = sum(1 for r in valid_rewards if r <= fail_threshold)
    if n_fail / n_valid >= max_fail:
        return DynamicFilterOutput(
            keep=False, reason=f"high_fail_rate_{n_fail}_of_{n_valid}",
        )

    # 4. low reward std
    min_std = _env_float("GROUP_FILTER_MIN_REWARD_STD", -1.0)
    if min_std >= 0:
        rstd = statistics.pstdev(valid_rewards) if n_valid > 1 else 0.0
        if rstd < min_std:
            return DynamicFilterOutput(
                keep=False, reason=f"low_std_{rstd:.3f}",
            )

    return DynamicFilterOutput(keep=True)
