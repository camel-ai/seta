# Plan 03 — DockerHarborRuntime with Different Environment Types

## Source
`seta_env/runtimes/docker_harbor_runtime.py`

## Test File
`test/test_runtimes.py`

## Dependencies
- Docker daemon running locally (for docker tests)
- `DAYTONA_API_KEY`, `DAYTONA_API_URL` env vars (for daytona tests)
- Modal credentials (for modal tests)
- A real Harbor task directory (see structure below)

## Class Under Test

```python
# seta_env/runtimes/docker_harbor_runtime.py line 32

class DockerHarborRuntime(ABC):
    def __init__(
        self,
        task_dir: str = None,       # path to Harbor task dir
        trial_root: str = None,     # root for trial outputs
        session_id: str = None,     # unique name (also Docker container name)
        environment_type: str = None,  # "docker" | "daytona" | "modal"
        environment: BaseEnvironment = None,  # pre-built env, skips factory
        **kwargs
    )

    async def reset(self, force_build: bool = False) -> None:
        """Calls harbor_env.start(force_build=force_build)"""

    async def stop(self, delete: bool = False) -> None:
        """Calls harbor_env.stop(delete=delete), closes logger"""

    async def get_tools(self) -> List[FunctionTool]:
        """Creates TerminalToolkit(working_directory='/workdir', ...) and wraps 7 methods as FunctionTool"""

    def __getattr__(self, name: str):
        """Proxies unknown attrs to self.harbor_env"""

    async def __aenter__(self): ...   # returns self
    async def __aexit__(self, ...):   # calls stop()
```

### Validated environment types (line 69):
```python
assert environment_type in [
    EnvironmentType.DOCKER.value,    # "docker"
    EnvironmentType.DAYTONA.value,   # "daytona"
    EnvironmentType.MODAL.value(),   # "modal"
]
```

## Fixtures Needed

### Minimal Harbor Task Directory
Create a temp task dir with this structure for tests:
```
<task_dir>/
├── task.toml
├── environment/
│   └── Dockerfile      # minimal ubuntu image
└── tests/
    └── run_tests.sh    # writes reward to $VERIFIER_DIR/reward.txt
```

Minimal `task.toml`:
```toml
[task]
name = "test_task"

[environment]
# empty or minimal docker config

[verifier]
test_path = "tests/run_tests.sh"
```

Minimal `run_tests.sh`:
```bash
#!/bin/bash
echo "1.0" > $VERIFIER_DIR/reward.txt
```

### `trial_root` and `session_id`
Use `tmp_path / "trials"` and `session_id = "test_session_<uuid>"`.

## Test Cases

### Initialization — Validation

| Scenario | Call | Expected |
|---|---|---|
| Valid type "docker" | `DockerHarborRuntime(task_dir=..., trial_root=..., session_id=..., environment_type="docker")` | No error, `harbor_env` created |
| Invalid type | `environment_type="k8s"` | `AssertionError` with message "Unsupported environment type: k8s" |
| Pre-built env | `DockerHarborRuntime(environment=some_env)` | Skips factory; `harbor_env = some_env`; `session_id` + `trial_dir` taken from env |

### Docker Runtime (requires Docker daemon)

| Scenario | Steps | Expected |
|---|---|---|
| `reset()` starts container | `runtime.reset()` | No exception; container is running (verify via `harbor_env.exec("echo ok")`) |
| `stop(delete=False)` | `runtime.stop(delete=False)` | No exception; container stopped but not removed |
| `stop(delete=True)` | `runtime.stop(delete=True)` | Container removed |
| `async with` context manager | `async with DockerHarborRuntime(...) as r: await r.reset()` | `stop()` called on exit; no resource leak |
| Logger initialized | After init | `_file_handler` is not None; log file exists at `_trial_paths.log_path` |
| Logger closed on stop | After `stop()` | `_file_handler` is None |
| `get_tools()` returns 7 tools | After reset | list of 7 `FunctionTool` (shell_exec, shell_view, shell_wait, shell_write_to_process, shell_kill_process, shell_write_content_to_file, shell_image_read, shell_ask_user_for_help = 8 actually — verify against `TerminalToolkit.get_tools()`) |
| `__getattr__` proxy | `runtime.exec("echo hi")` | Delegates to `harbor_env.exec("echo hi")` |

### Daytona Runtime (requires `DAYTONA_API_KEY`, `DAYTONA_API_URL`)

| Scenario | Steps | Expected |
|---|---|---|
| `reset()` creates workspace | `runtime.reset()` | Workspace created and started; no exception |
| `stop()` | `runtime.stop()` | Workspace stopped; no exception |
| Missing env vars | Unset `DAYTONA_API_KEY` | Error raised on `reset()` or during factory creation |

### Modal Runtime (requires Modal credentials)

| Scenario | Steps | Expected |
|---|---|---|
| `reset()` starts sandbox | `runtime.reset()` | Modal sandbox started; no exception |
| `stop()` | `runtime.stop()` | Sandbox released |

## Setup Notes

- Tag Docker-only tests with `@pytest.mark.docker` and skip if Docker unavailable.
- Tag Daytona tests with `@pytest.mark.daytona` and skip if env vars missing.
- Tag Modal tests with `@pytest.mark.modal` and skip if Modal not configured.
- Always call `await runtime.stop(delete=True)` in teardown to avoid leftover containers.

```python
import pytest
import asyncio
import uuid
from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime

@pytest.fixture
def task_dir(tmp_path):
    # create minimal harbor task structure
    ...

@pytest.fixture
def session_id():
    return f"test_{uuid.uuid4().hex[:8]}"
```
