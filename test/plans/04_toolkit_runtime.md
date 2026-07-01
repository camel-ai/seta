# Plan 04 — TerminalToolkit Connected to Live Runtime

## Source
`seta_env/toolkits/terminal_toolkit.py`

## Test File
`test/test_terminal_toolkit.py`

## Dependencies
- A running `DockerHarborRuntime` (from Plan 03 — start with `reset()` before tests)
- Docker daemon running locally

## Class Under Test

```python
# seta_env/toolkits/terminal_toolkit.py line 44

class TerminalToolkit(BaseToolkit):
    def __init__(
        self,
        timeout: float = 20.0,
        working_directory: Optional[str] = None,  # path inside container, e.g. "/workdir"
        session_logs_dir: str = "./session_logs",  # local dir for terminal.log
        safe_mode: bool = False,
        runtime: Any = None,  # BaseEnvironment — REQUIRED, raises ValueError if None
    )
```

Key constants:
```python
TRUNCATION_THRESHOLD = 2000   # chars
TRUNCATION_HEAD = 500
TRUNCATION_TAIL = 500
```

## How TerminalToolkit Is Created in Production
```python
# docker_harbor_runtime.py line 110
self.terminal_toolkit = TerminalToolkit(
    working_directory="/workdir",
    session_logs_dir=str(self.trial_dir / "terminal_logs"),
    runtime=self  # DockerHarborRuntime proxies exec() to harbor_env.exec()
)
```

## Fixtures

```python
@pytest.fixture
async def runtime(task_dir, tmp_path):
    """Start a docker runtime, yield it, stop+delete after test."""
    rt = DockerHarborRuntime(
        task_dir=str(task_dir),
        trial_root=str(tmp_path / "trials"),
        session_id=f"toolkit_test_{uuid.uuid4().hex[:8]}",
        environment_type="docker",
    )
    await rt.reset()
    yield rt
    await rt.stop(delete=True)

@pytest.fixture
def toolkit(runtime, tmp_path):
    return TerminalToolkit(
        working_directory="/workdir",
        session_logs_dir=str(tmp_path / "logs"),
        runtime=runtime,
    )
```

## Test Cases

### Initialization

| Scenario | Expected |
|---|---|
| `runtime=None` | raises `ValueError("Runtime is required.")` |
| `working_directory=None` | raises `AssertionError` |
| Valid init | `_log_file` created at `session_logs_dir/terminal.log` |
| `get_tools()` | returns list of 8 `FunctionTool` objects |

### `shell_exec(id, command, block=True)`

The blocking path calls `runtime.exec(command, cwd=..., timeout_sec=...)` directly — no tmux.

| Scenario | Command | Expected |
|---|---|---|
| Simple command | `"echo hello"` | Output contains `"hello"` |
| Command with stderr | `"echo err >&2"` | stderr merged into output |
| Command with exit code != 0 | `"exit 1"` or `"false"` | Output returned (not raised); may be empty |
| Working dir respected (safe_mode=True) | `safe_mode=True`, `"pwd"` | Output contains `/workdir` |
| Output logged | Any command | `terminal.log` contains the output |
| Timeout respected | `timeout=2`, `"sleep 10"` | Returns within ~2s (timeout caps at 60s per line 292) |

### `shell_exec(id, command, block=False)`

Non-blocking path: creates tmux session, sends keys, collects initial output.

| Scenario | Expected |
|---|---|
| Session created | `id` appears in `toolkit.shell_sessions` |
| Returns session start message | Output string starts with `"Session <id> started"` |
| Short command completes quickly | Initial output contains `[completed]` |
| Long command still running | Initial output contains `[still running]` |
| tmux installed | After call, `_tmux_checked = True` |

### `shell_view(id)`

| Scenario | Expected |
|---|---|
| Unknown session id | Returns `"Error: No session '<id>'."`  |
| No new output since last view | Returns `""` |
| New output available | Returns the new text; ANSI codes stripped |
| Offset advances | Second call after output returns `""` (already consumed) |

### `shell_write_to_process(id, command)`

Start an interactive process first (e.g. `shell_exec("s1", "python3", block=False)`), then:

| Scenario | Expected |
|---|---|
| Unknown session | Returns `"Error: No active session '<id>'."`  |
| Send input to python REPL | `shell_write_to_process("s1", "print(1+1)")` → output contains `2` |

### `shell_wait(id, wait_seconds)`

| Scenario | Expected |
|---|---|
| Unknown session | Returns `"Error: No session '<id>'."`  |
| Waits and collects output | Start a background command that prints over time; `shell_wait("s1", 3)` returns non-empty output |
| Duration respected | Call with `wait_seconds=2`; actual wait ≈ 2s (within ±1s) |

### `shell_kill_process(id)`

| Scenario | Expected |
|---|---|
| Unknown session | Returns `"Error: No session '<id>'."`  |
| Kill live session | Returns `"Session '<id>' terminated."` |
| Session removed from state | `id` no longer in `toolkit.shell_sessions` |
| tmux session gone | `runtime.exec("tmux ls")` no longer lists the killed session |

### `shell_write_content_to_file(content, file_path)`

| Scenario | Expected |
|---|---|
| Write to valid path | Returns `"Content written to '<path>'."`; file readable inside runtime |
| Verify file content | After write, `runtime.exec("cat <file_path>")` returns the written content |
| Write empty string | Returns success message; file exists and is empty |

### `shell_image_read(image_path)`

| Scenario | Expected |
|---|---|
| Non-existent path | Returns `ToolResult(text="Error: File '...' does not exist in runtime.")` |
| Valid PNG inside runtime | Returns `ToolResult` with `images=["data:image/png;base64,..."]` |

### Output Truncation

| Scenario | Expected |
|---|---|
| Output exactly 2000 chars | Not truncated; returned as-is |
| Output 2001+ chars | Returned string contains `"[Output truncated. Full output saved inside runtime at:"` |
| Truncated output | Starts with first 500 chars, ends with last 500 chars |
| Full output saved | `runtime.exec(f"cat {remote_path}")` returns full original output |

## Notes

- `terminal.log` is written after every public method call via `_log_entry`. Verify it grows after each operation.
- `_ensure_tmux()` installs tmux on first non-blocking call; the Docker image may not have it preinstalled — this is expected behavior.
