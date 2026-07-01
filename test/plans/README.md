# seta_env Test Plans

## Overview

This folder contains concrete, self-contained test plans for the `seta_env` package.
Each plan has enough context (file paths, signatures, behavior details) to implement
tests in a fresh session without re-reading the package source.

## Package Root
```
<REPO_ROOT>/seta_env/
```

## Testing Order (bottom-up)

| # | Plan File | Test File | What |
|---|-----------|-----------|------|
| 1 | [01_reward_functions.md](01_reward_functions.md) | `test/test_reward_functions.py` | Pure reward math, no deps |
| 2 | [02_utils.md](02_utils.md) | `test/test_utils.py` | `async_timer` + `load_main_trajectory` only |
| 3 | [03_runtime.md](03_runtime.md) | `test/test_runtimes.py` | `DockerHarborRuntime` with docker / daytona / modal |
| 4 | [04_toolkit_runtime.md](04_toolkit_runtime.md) | `test/test_terminal_toolkit.py` | `TerminalToolkit` wired to live Docker runtime |
| 5 | [05_agent_toolkit_runtime.md](05_agent_toolkit_runtime.md) | `test/test_agent.py` | `AgentTrain` + toolkit + runtime, Claude API model |
| 6 | [06_verifier.md](06_verifier.md) | `test/test_verifier.py` | `Verifier` against live runtime + real task dir |
| 7 | [07_reward_integration.md](07_reward_integration.md) | `test/test_reward_integration.py` | `reward_factory` with real verifier output |
| 8 | [08_terminal_environment.md](08_terminal_environment.md) | `test/test_terminal_environment.py` | `TerminalEnvironment` full end-to-end |
| 9 | [09_task_manager.md](09_task_manager.md) | `test/test_task_manager.py` | `TaskManager` — dataset registry, queue, sampler, status tracking |
| 10 | [10_orchestrator.md](10_orchestrator.md) | `test/test_workflow.py` | `SeTaEnvWorkflow` — AReal RolloutWorkflow backed by Redis TaskClient |

## Key Conventions

- All tests use `pytest` + `pytest-asyncio`.
- Async tests need `@pytest.mark.asyncio`.
- A real Harbor task directory is needed from stage 3 onwards — see each plan for the required structure.
- Claude API key must be set in `ANTHROPIC_API_KEY` env var for stages 5–8.
- For Docker runtime tests, Docker daemon must be running locally.
- Tests should be run in order — each stage depends on the layer below being stable.

## Harbor Task Directory Structure (required for stages 3–8)

```
<task_dir>/
├── task.toml          # task config (name, verifier config)
├── instruction.md     # task instruction text
├── environment/       # Dockerfile or docker-compose for the container
└── tests/
    ├── run_tests.sh   # executable test script run by Verifier
    └── ...            # any supporting test files
```

The test script (`run_tests.sh`) must write a reward to one of:
- `$VERIFIER_DIR/reward.txt`  — a single float, e.g. `0.5`
- `$VERIFIER_DIR/reward.json` — a dict, e.g. `{"test1": 1, "test2": 0}`


## Available dataset

Choose one of the task from one of the dataset to test.

each dataset has subfolders, each subfolder is exactly one harbor format task.

    - <REPO_ROOT>/dataset/seta-env-harbor
    - <REPO_ROOT>/dataset/terminal-bench-core_migrated
    - <REPO_ROOT>/dataset/terminal-bench-core-0.1.1_migrated
    - <REPO_ROOT>/dataset/terminal-bench-2.0

## Chosen Task for Testing (stages 3–8)

```
<REPO_ROOT>/dataset/terminal-bench-core_migrated/analyze-access-logs
```

- **Instruction:** Analyze `/app/access_log`, write summary to `/app/report.txt`
- **Test script:** `tests/test.sh` — installs uv+pytest, runs `tests/test_outputs.py`
- **Reward:** writes `1` or `0` to `/logs/verifier/reward.txt`
- **Difficulty:** easy, category: data-science