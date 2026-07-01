# Plan 08 — TerminalEnvironment Full End-to-End

## Source
`seta_env/environments/terminal_env.py`

## Test File
`test/test_terminal_environment.py`

## Dependencies
- All prior plans passing (runtime, toolkit, agent, verifier, reward)
- Docker daemon running locally
- `ANTHROPIC_API_KEY` set
- A real Harbor task directory (see structure below)
- `camel-ai` installed

## Class Under Test

```python
# seta_env/environments/terminal_env.py line 33

class TerminalEnvironment:
    def __init__(
        self,
        agent_config: Dict[str, Any],
        model_config: Dict[str, Any],
        runtime_config: Dict[str, Any],
        env_config: Dict[str, Any],
    )

    async def step(self, task: dict, uid: str, traj_i: int = 0) -> Tuple[dict, float]:
        """
        Runs 5 stages via async_timer:
            1_reset_env   → _reset_env(task, uid)
            2_run_agent   → run_agent()
            3_evaluate    → evaluate()
            4_calculate_reward → calculate_reward()
            5_close       → close()
        Returns (run_info, reward).
        Catches ALL exceptions — writes to self.error_info, returns in run_info.
        """
```

### `run_info` structure (always returned even on error):
```python
{
    "task_name": str,
    "uid": str,
    "traj_i": int,
    "error_info": {},                 # empty on success
    "timings": {                      # stage name → {"start", "end", "elapsed"}
        "1_reset_env": {...},
        "2_run_agent": {...},
        ...
    },
    "reward": float | None,
    "evaluation": dict | {},
    "agent_summary": dict | {},       # agent.meta_info_record after run
}
```

### Config Structure

```python
agent_config = {
    "system_message": str,
    "max_total_tokens": int,
    "max_iteration": int,
    "working_directory": str,         # for NoteTakingToolkit
    "tool_names": List[str],          # filter which tools agent can use
}

model_config = {
    "model": ModelBackend,            # Claude model instance
}

runtime_config = {
    "task_dir": str,
    "trial_root": str,
    "session_id": str,
    "environment_type": str,          # "docker"
    # OR: "environment": BaseEnvironment  (skip factory)
}

env_config = {
    "reward_fn": str,                 # e.g. "pass_ratio"
}
```

## Task for Testing

Use a simple self-contained task where the agent can succeed in 1–3 tool calls:

**Instruction:** `"Create a file at /workdir/result.txt containing the text 'done'."`

The test script checks for this file:
```bash
#!/bin/bash
mkdir -p $VERIFIER_DIR
if [ -f /workdir/result.txt ] && grep -q "done" /workdir/result.txt; then
    echo '{"file_created": 1}' > $VERIFIER_DIR/reward.json
else
    echo '{"file_created": 0}' > $VERIFIER_DIR/reward.json
fi
```

## Fixtures

```python
@pytest.fixture
def agent_config():
    return {
        "system_message": "You are a developer agent. Use shell tools to complete the task.",
        "max_total_tokens": 8000,
        "max_iteration": 5,
        "working_directory": "/workdir",
        "tool_names": ["shell_exec", "shell_write_content_to_file", "shell_view"],
    }

@pytest.fixture
def model_config():
    model = ModelFactory.create(
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type="claude-haiku-4-5-20251001",
    )
    return {"model": model}

@pytest.fixture
def runtime_config(task_dir, tmp_path):
    return {
        "task_dir": str(task_dir),
        "trial_root": str(tmp_path / "trials"),
        "session_id": f"env_test_{uuid.uuid4().hex[:8]}",
        "environment_type": "docker",
    }

@pytest.fixture
def env_config():
    return {"reward_fn": "pass_ratio"}

@pytest.fixture
def terminal_env(agent_config, model_config, runtime_config, env_config):
    return TerminalEnvironment(agent_config, model_config, runtime_config, env_config)
```

## Test Cases

### Happy Path — `step()` completes all stages

```python
task = {
    "task_name": "create_file",
    "task_path": str(task_dir),
    "instruction": "Create a file at /workdir/result.txt containing the text 'done'.",
}
run_info, reward = await env.step(task, uid="uid_001", traj_i=0)
```

| Check | Expected |
|---|---|
| No exception | True |
| `run_info["error_info"]` | `{}` (empty) |
| `run_info["task_name"]` | `"create_file"` |
| `run_info["uid"]` | `"uid_001"` |
| `run_info["traj_i"]` | `0` |
| `run_info["timings"]` has 5 keys | `1_reset_env`, `2_run_agent`, `3_evaluate`, `4_calculate_reward`, `5_close` |
| Each timing has `start`, `end`, `elapsed` | True |
| `run_info["evaluation"]` | dict (e.g. `{"file_created": 1}`) |
| `reward` | float between 0.0 and 1.0 |
| `run_info["agent_summary"]` has keys | `iteration_count`, `termination_reason` etc. |

### Stage 1 Failure — Runtime Reset Fails

Pass a broken `task_dir` (missing required Harbor files) or invalid Docker image.

| Check | Expected |
|---|---|
| Returns `(run_info, None)` without crash | True |
| `run_info["error_info"]["stage"]` | `"1_reset_env"` |
| `run_info["error_info"]["error_message"]` | non-empty string |
| `reward` | `None` |
| Stages 2–5 not in timings | True (stopped at stage 1) |

### Stage 2 Failure — Agent Errors Mid-Run

Use a broken model (mock model that raises) or invalid API key.

| Check | Expected |
|---|---|
| `run_info["error_info"]["stage"]` | `"2_run_agent"` |
| `reward` | `None` |

### Stage 3 Failure — Verifier Fails (no reward file written)

Use Variant D task dir from Plan 06 (test script writes no reward file).

| Check | Expected |
|---|---|
| `run_info["error_info"]["stage"]` | `"3_evaluate"` |
| `reward` | `None` |

### Pre-initialized Environment Path

Pass `environment=harbor_env` in `runtime_config` instead of `environment_type`.

```python
runtime_config["environment"] = pre_started_harbor_env
del runtime_config["environment_type"]
del runtime_config["task_dir"]
```

| Check | Expected |
|---|---|
| `DockerHarborRuntime` skips factory | True (no new container created) |
| `step()` completes normally | True |

### `close()` called even on error

The `finally` block in `step()` always returns `run_info`. Verify by checking
that after a stage 2 failure, the runtime container is still stoppable (i.e., `close()`
ran and `runtime.stop()` was called).

### `run_info["timings"]` accuracy

| Check | Expected |
|---|---|
| `1_reset_env.elapsed` >= 0 | True |
| Each elapsed sums to roughly total wall time | True (within 10%) |
| All stages present on successful run | True |

### `agent_summary` completeness

After a successful run check `run_info["agent_summary"]` (= `agent.meta_info_record`):

| Key | Expected |
|---|---|
| `iteration_count` | int > 0 |
| `termination_reason` | a `TerminationReason` enum value |
| `parse_error_count` | int >= 0 |
| `total_tool_calls` | int >= 0 |

## Notes

- Each test creates a fresh Docker container. Always verify cleanup (no dangling containers).
- Use Claude Haiku to minimize cost; these are integration tests, not capability tests.
- Tests will take 60–180 seconds each due to container startup + model inference.
- Set `pytest-timeout` to 300s per test.
- Run with `pytest -s` to see timing output from `async_timer`.
