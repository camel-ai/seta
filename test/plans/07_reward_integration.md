# Plan 07 — Reward Function Integration (with Real Verifier Output)

## Source
`seta_env/verifiers/reward_fn.py`
`seta_env/environments/terminal_env.py` (lines 193–211, `calculate_reward`)

## Test File
`test/test_reward_integration.py`

## Dependencies
- A running Docker runtime (from Plan 03)
- `Verifier` working correctly (Plan 06 must pass)
- `load_main_trajectory` (from utils) for the trajectory path

## What This Tests

`reward_factory` receiving real `evaluation_results` dicts produced by `Verifier.verify()`,
and the `calculate_reward()` logic in `TerminalEnvironment` that wraps it.

The `calculate_reward()` method (terminal_env.py lines 193–211):
```python
async def calculate_reward(self) -> float:
    try:
        try:
            trajectory = await load_main_trajectory(str(self.output_path / "CAMEL_LOG_DIR"))
        except FileNotFoundError:
            trajectory = None   # <-- trajectory is optional

        reward = await reward_factory(
            self.reward_fn,
            evaluation_results=self.evaluation_results,
            trajectory=trajectory
        )
        return reward
    except Exception as e:
        logger.error(f"Error in calculating reward: {e}")
        return None
```

## Test Cases

### Integration: `reward_factory` with real `Verifier` output

Run `Verifier.verify()` against each task variant (A, B, C from Plan 06),
then pass the result to `reward_factory`.

| Verifier variant | `evaluation_results` | `reward_factory("pass_ratio", ...)` |
|---|---|---|
| Variant A (reward.txt = 1.0) | `{"reward": 1.0}` | `1.0` |
| Variant B (reward.txt = 0.0) | `{"reward": 0.0}` | `0.0` |
| Variant C (reward.json mixed) | `{"test1": 1, "test2": 0}` | `0.5` |

### `reward_factory` with `trajectory=None` (no CAMEL_LOG_DIR)

Mirror how `calculate_reward()` handles `FileNotFoundError`:
```python
reward = await reward_factory(
    "pass_ratio",
    evaluation_results={"test1": 1},
    trajectory=None
)
```
| Check | Expected |
|---|---|
| Returns float | `1.0` |
| `trajectory` kwarg silently ignored | True (`reward_fn_pass_ratio` uses `**kwargs` but ignores trajectory) |

### `reward_factory` with `trajectory` string present

Pass a non-None trajectory string (as returned by `load_main_trajectory`).
```python
reward = await reward_factory(
    "pass_ratio",
    evaluation_results={"test1": 1, "test2": 0},
    trajectory="[User]: do something\n[Assistant]: done",
)
```
| Check | Expected |
|---|---|
| Returns `0.5` | True (trajectory does not affect pass_ratio) |

### `calculate_reward()` path — no trajectory file

Simulate `output_path / "CAMEL_LOG_DIR"` not existing (no agent has run).
```python
self.output_path = tmp_path / "nonexistent"
self.evaluation_results = {"test1": 1}
self.reward_fn = "pass_ratio"
reward = await env.calculate_reward()
```
| Check | Expected |
|---|---|
| Returns `1.0` (not None) | True — `FileNotFoundError` caught internally, `trajectory=None` passed |

### `calculate_reward()` path — unknown reward_fn

```python
self.reward_fn = "nonexistent_fn"
self.evaluation_results = {"test1": 1}
reward = await env.calculate_reward()
```
| Check | Expected |
|---|---|
| Returns `None` | True — `ValueError` from `reward_factory` caught, returns None |

## Setup

These tests can be run without a full `TerminalEnvironment` by directly calling
`reward_factory` with dict inputs, and by calling `calculate_reward()` on a
minimally-constructed `TerminalEnvironment` with the required attributes patched.

```python
import pytest
from seta_env.verifiers.reward_fn import reward_factory

@pytest.mark.asyncio
async def test_pass_ratio_with_real_verifier_output():
    evaluation_results = {"test1": 1, "test2": 0}  # as returned by Verifier
    reward = await reward_factory("pass_ratio", evaluation_results=evaluation_results)
    assert reward == 0.5
```
