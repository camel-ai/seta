# Plan 06 — Verifier Connected to Live Runtime

## Source
`seta_env/verifiers/verifier.py`

## Test File
`test/test_verifier.py`

## Dependencies
- A running Docker runtime (`DockerHarborRuntime` after `reset()`)
- A real Harbor task directory with a `tests/` folder
- `harbor` package: `Task`, `TrialPaths`, `EnvironmentPaths`, `VerifierResult`, `resolve_env_vars`

## Class Under Test

```python
# seta_env/verifiers/verifier.py line 33

class Verifier:
    def __init__(
        self,
        task: Task,                        # Harbor Task object loaded from task_dir
        trial_paths: TrialPaths,           # paths for trial output (log, reward files)
        environment: BaseEnvironment,      # the running harbor environment
        logger: logging.Logger | None = None,
    )

    async def verify(self) -> dict:
        """
        1. Uploads task.paths.tests_dir → /tests inside runtime
        2. chmod +x test_script_path
        3. Runs: test_script_path > test_stdout_path 2>&1
        4. Downloads verifier_dir from runtime if environment is not mounted
        5. Reads reward from reward_text_path or reward_json_path
        6. Returns dict (reward values)

        Raises:
            AddTestsDirError: if upload_dir fails
            DownloadVerifierDirError: if download_dir fails (non-mounted envs)
            RewardFileNotFoundError: if neither reward.txt nor reward.json exists
        """

    def _parse_reward_text(self) -> dict[str, float | int]:
        """
        Reads trial_paths.reward_text_path.
        Returns {"reward": float(content)}.
        Raises RewardFileEmptyError if file is empty.
        Raises VerifierOutputParseError if content is not a valid float.
        """

    def _parse_reward_json(self) -> dict[str, float | int]:
        """
        Reads trial_paths.reward_json_path.
        Returns parsed JSON dict.
        Raises RewardFileEmptyError if file is empty.
        Raises VerifierOutputParseError if not valid JSON.
        """
```

### Key Harbor Paths Used:
```python
# Inside runtime (container):
EnvironmentPaths.tests_dir    # where tests are uploaded to in the container
EnvironmentPaths.verifier_dir # where reward files are written inside container

# On host (trial output):
trial_paths.reward_text_path  # host path to reward.txt
trial_paths.reward_json_path  # host path to reward.json
trial_paths.test_stdout_path  # host path to test stdout log
```

## Fixtures

### Task Directory for Tests

Create three variants of task dirs to cover different scenarios:

**Variant A — Passing test (writes reward.txt)**
```
task_dir_pass/
├── task.toml
├── environment/Dockerfile
└── tests/
    └── run_tests.sh
```
`run_tests.sh`:
```bash
#!/bin/bash
mkdir -p $VERIFIER_DIR
echo "1.0" > $VERIFIER_DIR/reward.txt
```

**Variant B — Failing test (writes reward.txt = 0)**
```bash
#!/bin/bash
mkdir -p $VERIFIER_DIR
echo "0.0" > $VERIFIER_DIR/reward.txt
```

**Variant C — JSON reward**
```bash
#!/bin/bash
mkdir -p $VERIFIER_DIR
echo '{"test1": 1, "test2": 0}' > $VERIFIER_DIR/reward.json
```

**Variant D — No reward file written**
```bash
#!/bin/bash
exit 0   # writes nothing
```

```python
@pytest.fixture
async def runtime_and_verifier(task_dir_pass, tmp_path):
    trial_root = tmp_path / "trials"
    session_id = f"verifier_test_{uuid.uuid4().hex[:8]}"

    rt = DockerHarborRuntime(
        task_dir=str(task_dir_pass),
        trial_root=str(trial_root),
        session_id=session_id,
        environment_type="docker",
    )
    await rt.reset()

    verifier = Verifier(
        task=rt._task,
        trial_paths=rt._trial_paths,
        environment=rt.harbor_env,
    )
    yield rt, verifier
    await rt.stop(delete=True)
```

## Test Cases

### `verify()` — reward.txt path (Variant A)

| Check | Expected |
|---|---|
| Returns a dict | True |
| `result["reward"]` | `1.0` |
| `test_stdout_path` exists on host | True |
| No exception | True |

### `verify()` — reward.txt = 0.0 (Variant B)

| Check | Expected |
|---|---|
| `result["reward"]` | `0.0` |

### `verify()` — reward.json path (Variant C)

| Check | Expected |
|---|---|
| Returns dict with multiple keys | `{"test1": 1, "test2": 0}` |

### `verify()` — No reward file written (Variant D)

| Check | Expected |
|---|---|
| Raises `RewardFileNotFoundError` | True |

### `_parse_reward_text()` — unit tests (no runtime needed, use tmp_path)

| Scenario | File content | Expected |
|---|---|---|
| Valid float | `"0.75\n"` | `{"reward": 0.75}` |
| Not a float | `"not_a_number"` | raises `VerifierOutputParseError` |
| Empty file | `""` | raises `RewardFileEmptyError` |

### `_parse_reward_json()` — unit tests (no runtime needed, use tmp_path)

| Scenario | File content | Expected |
|---|---|---|
| Valid JSON dict | `'{"t1": 1, "t2": 0}'` | `{"t1": 1, "t2": 0}` |
| Invalid JSON | `"not json"` | raises `VerifierOutputParseError` |
| Empty file | `""` | raises `RewardFileEmptyError` |

### Upload failure

Mock `environment.upload_dir` to raise an exception. Verify `verify()` raises `AddTestsDirError`.

### Env var resolution (verifier.env in task.toml)

If `task.config.verifier.env` contains `{"MY_VAR": "$SOME_ENV_VAR"}`, verify `resolve_env_vars` is called and the script receives the resolved value.

## Notes

- The `_parse_reward_text` and `_parse_reward_json` unit tests only need a temp file on disk — no runtime required.
- For Docker (non-mounted) environments, `verify()` calls `environment.download_dir()`. This copies reward files from container to host `trial_paths.verifier_dir`. Ensure this download path is tested.
- If the environment `is_mounted=True` (Docker with bind mount), the download step is skipped.
