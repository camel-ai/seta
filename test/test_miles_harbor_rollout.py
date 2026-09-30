"""scripts/miles/common/harbor_rollout.py: infrastructure-failure and zero-spread group filters.

Needs Miles installed (runs inside the Miles container); skipped otherwise.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

miles_filters = pytest.importorskip("miles.rollout.filter_hub.dynamic_sampling_filters")
Sample = pytest.importorskip("miles.utils.types").Sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "miles"))
from common.harbor_rollout import (  # noqa: E402
    INFRA_EXIT_STATUSES,
    check_no_infra_failure,
    check_no_infra_failure_and_nonzero_std,
)


def _sample(exit_status=None, status=None, reward=0.0):
    s = Sample(prompt="p", response="r", reward=reward)
    if status is not None:
        s.status = status
    s.metadata = {"exit_status": exit_status} if exit_status else {}
    return s


def test_submitted_group_is_kept():
    out = check_no_infra_failure(SimpleNamespace(), [[_sample("Submitted"), _sample("Submitted")]])
    assert out.keep


@pytest.mark.parametrize("status", sorted(INFRA_EXIT_STATUSES))
def test_infra_status_rejects_group(status):
    out = check_no_infra_failure(SimpleNamespace(), [[_sample("Submitted"), _sample(status)]])
    assert not out.keep
    assert status in out.reason


@pytest.mark.parametrize("status", ["TimeLimitExceeded", "SequenceLengthLimitExceeded"])
def test_contract_outcomes_are_kept(status):
    """Budget and response-cap outcomes are the policy's own; they train as reward 0."""
    out = check_no_infra_failure(SimpleNamespace(), [[_sample("Submitted"), _sample(status)]])
    assert out.keep


def test_aborted_sample_still_rejects():
    out = check_no_infra_failure(SimpleNamespace(), [[_sample("Submitted", status=Sample.Status.ABORTED)]])
    assert not out.keep
    assert out.reason == "group_has_aborted"


# -- check_no_infra_failure_and_nonzero_std --

# check_reward_nonzero_std reads args.reward_key to pick the reward field; the
# real Miles args always carry it, so the stub must too.
ARGS = SimpleNamespace(reward_key=None)


def test_spread_group_is_kept():
    """A group with both a success and a failure carries the gradient GRPO needs."""
    group = [_sample("Submitted", reward=1.0), _sample("Submitted", reward=0.0)]
    assert check_no_infra_failure_and_nonzero_std(ARGS, [group]).keep


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_zero_spread_group_is_rejected(reward):
    """All-fail and all-pass both give every member advantage 0; neither trains."""
    group = [_sample("Submitted", reward=reward) for _ in range(4)]
    out = check_no_infra_failure_and_nonzero_std(ARGS, [group])
    assert not out.keep
    assert "zero_std" in out.reason


def test_infra_failure_reported_before_zero_spread():
    """A group destroyed by AgentError is an infrastructure failure, not a flat group."""
    group = [_sample("Submitted", reward=0.0), _sample("AgentError", reward=0.0)]
    out = check_no_infra_failure_and_nonzero_std(ARGS, [group])
    assert not out.keep
    assert "infra_failure" in out.reason


def test_base_filter_still_ignores_reward_spread():
    """The composed filter must not change the behaviour of the one recorded runs use."""
    group = [_sample("Submitted", reward=0.0) for _ in range(4)]
    assert check_no_infra_failure(SimpleNamespace(), [group]).keep
