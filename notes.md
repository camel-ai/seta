# Notes

## asyncio + uvloop subprocess bug in eval.py

### Symptom
`eval.py` fails with `NotImplementedError` when `asyncio.create_subprocess_exec` is called inside `DockerEnvironment._run_docker_compose_command`. `harbor run` works fine with the same docker environment.

```
File ".../asyncio/events.py", line 645, in get_child_watcher
    raise NotImplementedError
```

### Root cause
`seta_env.orchestrators.grpo_rollout` transitively imports `camel` → `sglang`, and sglang sets `asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())` **at module level** (e.g. `sglang/srt/entrypoints/engine.py:87`).

In `eval.py`, `GRPORollout` was imported lazily *inside* `main_async`, which runs after `asyncio.run()` has already created a **default** event loop. When sglang then swaps the policy to uvloop mid-flight, `create_subprocess_exec` calls `uvloop.EventLoopPolicy().get_child_watcher()` which raises `NotImplementedError` because uvloop doesn't implement the child watcher API.

`harbor run` is unaffected because it never imports sglang.

### Fix
Move the `GRPORollout` import to the **top level** of `eval.py` (before `asyncio.run()` is called). This ensures uvloop replaces the policy before the event loop is created, so `asyncio.run()` creates a proper uvloop event loop that handles subprocesses natively — no `get_child_watcher()` call needed.

```python
# eval.py — top-level imports
from seta_env.orchestrators.grpo_rollout import GRPORollout  # must be before asyncio.run()
```

**File changed:** `scripts/evaluation/eval.py`
