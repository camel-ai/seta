# Plan 01 — Reward Functions (Unit)

## Source
`seta_env/verifiers/reward_fn.py`

## Test File
`test/test_reward_functions.py`

## Dependencies
None. Pure async Python functions, no imports beyond stdlib.

## Functions Under Test

```python
# seta_env/verifiers/reward_fn.py

async def reward_fn_pass_ratio(evaluation_results: dict, **kwargs) -> float:
    """
    evaluation_results: {test_name: 0 or 1}
    Returns: passed / total. Returns 0.0 if empty.
    """

async def reward_factory(reward_fn_name: str, **kwargs) -> float:
    """
    Dispatches to the right reward function by name.
    kwargs passed through — reward_fn_pass_ratio uses kwargs["evaluation_results"].
    Raises ValueError for unknown names.
    """
```

## Test Cases

### `reward_fn_pass_ratio`

| Scenario | Input | Expected |
|---|---|---|
| All pass | `{"t1": 1, "t2": 1, "t3": 1}` | `1.0` |
| All fail | `{"t1": 0, "t2": 0}` | `0.0` |
| Mixed 3/5 pass | `{"t1":1,"t2":1,"t3":1,"t4":0,"t5":0}` | `0.6` |
| Single pass | `{"t1": 1}` | `1.0` |
| Single fail | `{"t1": 0}` | `0.0` |
| Empty dict | `{}` | `0.0` |
| Float values | `{"t1": 0.5, "t2": 0.5}` | `0.5` (sum/len) |

### `reward_factory`

| Scenario | Input | Expected |
|---|---|---|
| Known name | `reward_fn_name="pass_ratio", evaluation_results={"t1":1}` | `1.0` |
| Passes kwargs through | `evaluation_results={"t1":1,"t2":0}` | `0.5` |
| Unknown name | `reward_fn_name="nonexistent"` | raises `ValueError` |
| trajectory kwarg ignored | `reward_fn_name="pass_ratio", evaluation_results={"t1":1}, trajectory="anything"` | `1.0` (trajectory not used by pass_ratio) |

## Setup

```python
import pytest
import asyncio
from seta_env.verifiers.reward_fn import reward_fn_pass_ratio, reward_factory
```

No fixtures needed. Mark all tests `@pytest.mark.asyncio`.
