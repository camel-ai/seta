# Terminal Toolkit — Unified Tool Behavior

All three toolkit configurations follow this contract:

1. **TerminalToolkitDocker** + local DockerEnvironment (sync, Docker API)
2. **TerminalToolkit (tmux)** + local DockerEnvironment (async, tmux → docker compose exec)
3. **TerminalToolkit (tmux)** + RemoteDockerEnvironment (async, tmux → HTTP → node_manager → docker compose exec)

## Constants

- **Truncation threshold**: 1000 chars (500 head + 500 tail)
- **Truncation message**: `... [Output truncated. Full output saved at: <path>] ...`
- **Polling interval**: 0.5s (shell_wait, blocking exec timeout loop)

## Environment Layer Guarantees

All three configurations produce identical `ExecResult` for the same command:

- **stdout**: merged stdout+stderr (stderr is folded into stdout)
- **stderr**: always `None` under normal operation
- **return_code**: raw process exit code (no coercion)
- **Timeout**: returns `ExecResult(stdout="", stderr=<error msg>, return_code=-1)` — never raises
- **TTY**: no TTY allocation (`-T` flag) — output is raw, no ANSI escapes injected by Docker
- **ANSI stripping**: both toolkits strip ANSI escape sequences from all output before returning to the agent

## Logging

Both toolkits write to a single `terminal.log` file with structured entries:

```
================================================================
[ISO_TIMESTAMP] METHOD_NAME | session: SESSION_ID
----------------------------------------------------------------
key1: value1
key2: value2

OUTPUT:
{possibly_truncated_output}
================================================================
```

All public methods log through this format. No separate log files per session or per blocking/non-blocking mode.

---

## shell_exec(id, command, block=True)

### block=True — command completes within timeout
- Returns: raw stdout+stderr (ANSI-stripped, truncated if >1000 chars)
- Session is **not** tracked — no entry in `shell_sessions`

### block=True — command exceeds timeout
- Process **keeps running** (not killed)
- Session is **converted to non-blocking** — added to `shell_sessions`
- Returns:
  ```
  Command did not complete within {timeout} seconds. Session '{id}' is still running.

  You can use:
    - shell_view('{id}') - get current output
    - shell_wait('{id}', wait_seconds=30) - wait for completion
    - shell_kill_process('{id}') - terminate

  [Partial output so far]:
  {truncated partial output}
  ```

### block=True — error (tmux/Docker start failure)
- Returns error string: `Error: Failed to start session '{id}': {details}`
- Does **not** raise an exception

### block=False
- Process starts in background, tracked in `shell_sessions`
- Returns:
  ```
  Session '{id}' started.

  [Initial Output]:
  {output collected until idle}
  ```

---

## shell_view(id)

### Session does not exist
- Returns: `Error: No session '{id}'.`

### Session running, new output available
- Returns: new output since last call (truncated if >1000)

### Session running, no new output
- Returns: `""` (empty string)

### Session completed (process exited)
- Returns: remaining output + `\n[completed]`

---

## shell_wait(id, wait_seconds=5.0)

### Session does not exist
- Returns: `Error: No session '{id}'.`

### Session already completed
- Returns: `Session is no longer running. Use shell_view to get final output.`

### During wait — process produces output
- Collects all new output across polling intervals
- Returns: concatenated raw output (truncated if >1000)
- **No** `[completed]` marker — use `shell_view` to check status

### During wait — process exits before wait_seconds
- Collects remaining output, returns immediately (stops early)
- Returns: concatenated raw output (truncated if >1000)

### During wait — process still running at wait_seconds
- Returns: whatever output was collected during the wait period

---

## shell_write_to_process(id, command)

### Session does not exist or not running
- Returns: `Error: No active session '{id}'.`

### Normal operation
- Flushes pending output, sends `command + '\n'` to stdin
- Waits until process becomes idle (no new output for ~0.3s)
- Returns: raw output collected after the write
- **No** status tags — raw output only

---

## shell_kill_process(id)

### Session does not exist or not running
- Returns: `Error: No active session '{id}'.`

### Normal operation
- Terminates process, cleans up session state (pops from `shell_sessions`)
- Returns: `Session '{id}' terminated.`

---

## shell_write_content_to_file(content, file_path)

### Normal operation
- Writes content to file_path inside the container
- Returns: `Content written to '{file_path}'.`

### Error (permission denied, invalid path, etc.)
- Returns: `Error writing to '{file_path}': {details}`

---

## shell_image_read(image_path) — tmux toolkit only

Not registered in runtime tools (text-only model). Available via `get_tools()` on the toolkit directly.

### File does not exist
- Returns: ToolResult with text `does not exist in runtime at '{path}'`

### Valid image
- Returns: ToolResult with base64 data URI in `images` list

---

## Runtime Integration: toolkit="auto"

- Local DockerEnvironment → selects **TerminalToolkitDocker** (Docker API)
- RemoteDockerEnvironment → selects **TerminalToolkit** (tmux)
- Explicit `toolkit="tmux"` → always selects TerminalToolkit regardless of environment
- Explicit `toolkit="docker"` → always selects TerminalToolkitDocker (will crash for remote)
