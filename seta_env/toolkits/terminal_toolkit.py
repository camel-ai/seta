# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import asyncio
import atexit
import base64
import hashlib
import mimetypes
import re
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

from camel.logger import get_logger
from camel.toolkits.base import BaseToolkit, manual_timeout
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer
from camel.utils.tool_result import ToolResult

logger = get_logger(__name__)

# Commands that paint the screen rather than stream lines
# TODO: use capture-pane for these instead of pipe-pane + strip_ansi
SCREEN_ORIENTED_COMMANDS = {"vim", "vi", "nano", "htop", "top", "less", "man"}

TRUNCATION_THRESHOLD = 8000  # chars (~2-3K tokens) — keep in sync with terminal_toolkit_docker.py
TRUNCATION_HEAD = 4000
TRUNCATION_TAIL = 4000

# Hard cap on the bytes we keep from a shell session's polled output, applied
# inside _shell_view / _collect_output_until_idle AND inside _log_entry's
# "full output save" path. Defends against pathological shell commands
# producing megabytes of output that would feed Daytona SDK's demux_log a
# huge CPU-bound parse on the event loop. (The earlier upload-back overflow
# concern is gone — _log_entry's save runs entirely inside the sandbox now.)
# Independent of the agent-visible TRUNCATION_THRESHOLD above — this is a
# pipeline-internal cap to keep per-poll parsing cheap.
SHELL_OUTPUT_BYTE_CAP = 64 * 1024  # 64 KB


# @MCPServer()
class TerminalToolkit(BaseToolkit):
    """
    Async-native terminal toolkit that manages tmux sessions inside a
    BaseEnvironment runtime. Supports blocking exec, non-blocking interactive
    sessions, and file operations.

    All public methods are async. Use asyncio.run() or await directly.

    Session state lives in self.shell_sessions keyed by agent-provided id.
    All output is logged to a single unified log file (thread-safe via asyncio.Lock).
    Long outputs are truncated for the agent but saved in full inside the runtime.

    Args:
        timeout (float | None): Per-method auto-wrap timeout consumed by
            CAMEL's BaseToolkit.__init_subclass__. When set, EVERY async
            method on this class gets wrapped with asyncio.wait_for(timeout) —
            including setup helpers like _ensure_tmux. Pass None to disable
            the wrap (recommended in env_service where per-stage timeouts
            in terminal_env supply the real budget). NOT used internally
            by shell_exec for its block deadline — see `block_timeout`.
        block_timeout (float): Deadline for shell_exec(block=True) in
            seconds. After this many seconds the command is left running
            in its tmux session and the agent gets a timeout message
            instructing it to use shell_view/shell_wait/shell_kill_process.
            Default 20s matches CAMEL's historical behavior.
        working_directory (Optional[str]): Working directory inside runtime;
            used as cwd for exec calls when safe_mode is True.
        session_logs_dir (str): Local directory for the unified terminal.log.
        safe_mode (bool): Restrict exec to working_directory when True.
        runtime: A BaseEnvironment instance (harbor).
    """

    def __init__(
        self,
        timeout: float = None,
        working_directory: Optional[str] = None,
        session_logs_dir: str = "./session_logs",
        safe_mode: bool = False,
        runtime: Any = None,  # BaseEnvironment instance
        block_timeout: float = 20.0,
    ):
        super().__init__()
        self.timeout = timeout
        self.block_timeout = block_timeout
        self.working_directory = working_directory
        self.session_logs_dir = session_logs_dir
        self.safe_mode = safe_mode
        self.runtime = runtime

        # session state: id -> {tmux_name, log_path, last_offset}
        self.shell_sessions: Dict[str, Dict[str, Any]] = {}

        # Per-session asyncio.Lock. Held by every public method that touches
        # shell_sessions[id] or the pipe-pane log offset, so concurrent calls
        # on the same session id serialize (within a session) while calls on
        # different ids run in parallel. Lazily populated via _session_lock().
        # The agent loop dispatches tool calls via asyncio.gather without
        # knowing this toolkit's per-id contract — concurrency safety lives
        # here, not in the dispatcher.
        self._session_locks: Dict[str, asyncio.Lock] = {}

        # unified log file — all tool calls from all sessions go here sequentially
        self._log_file = Path(session_logs_dir) / "terminal.log"
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        # asyncio.Lock guards all writes to _log_file
        self._log_lock = asyncio.Lock()

        if not self.runtime:
            raise ValueError("Runtime is required.")
        assert self.working_directory is not None, "working_directory must be set"

        # Safety net: clear session dict on exit.
        # Container teardown via stop() kills all tmux sessions inside it.
        def _sync_cleanup():
            self.shell_sessions.clear()
        atexit.register(_sync_cleanup)

    def __await__(self):
        """
        Makes the toolkit awaitable so callers do:

            toolkit = await TerminalToolkit(...)

        Runs tmux install and working-directory setup eagerly at construction
        time, raising RuntimeError immediately if tmux cannot be installed.
        """
        async def _setup():
            await self._ensure_tmux()
            await self._ensure_working_directory()
            return self
        return _setup().__await__()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _session_lock(self, id: str) -> asyncio.Lock:
        """Return the asyncio.Lock for session `id`, creating it on first use.

        Single-threaded asyncio dict access is atomic between awaits, so
        setdefault is safe here. The lock dict grows monotonically and is
        cleaned up alongside the session in shell_kill_process and
        shell_exec(block=True) completion paths.
        """
        return self._session_locks.setdefault(id, asyncio.Lock())

    # Full ESC sequence handling. Order of alternatives is critical: each
    # introducer with its own tail (CSI, OSC, SS3, DCS/SOS/PM/APC) MUST be
    # matched before the generic two-char fallback, otherwise the fallback
    # greedily consumes only the 2-byte introducer and leaves the tail.
    # Sequence shapes (xterm / ECMA-48):
    #   CSI: ESC '[' params intermediates final     final ∈ 0x40-0x7E
    #   OSC: ESC ']' params (BEL | ST=ESC '\')
    #   SS3: ESC 'O' final
    #   DCS / SOS / PM / APC: ESC P/X/^/_ data ST
    #   Plain Fe/Fs: ESC <single byte 0x20-0x7E>
    _ANSI_ESCAPE = re.compile(
        r'\x1B'
        r'(?:'
            r'\[[0-?]*[ -/]*[@-~]'                  # CSI
            r'|\][^\x07\x1B]*(?:\x07|\x1B\\)?'      # OSC (BEL or ST terminator; tolerate unterminated)
            r'|O[@-~]'                              # SS3
            r'|[PX^_][^\x1B\x07]*(?:\x07|\x1B\\)?'  # DCS / SOS / PM / APC
            r'|[ -~]'                               # generic 2-byte fallback (any final byte 0x20-0x7E)
        r')'
    )
    # Orphaned escape-sequence TAIL when the leading bytes were consumed in a
    # prior read chunk:
    #   "\x1b"      consumed last chunk → "[?25h"  arrives next → starts with "["
    #   "\x1b["     consumed last chunk → "?25h"   arrives next → starts with [0-?]+
    #   "\x1b]"     consumed last chunk → "11;?\x07" arrives next → starts with digits;...BEL
    # MULTILINE so ^ matches after \n (catches mid-stream tails surviving past
    # the per-chunk strip — observed in fully-async multi-turn data).
    _ANSI_ORPHAN = re.compile(
        r'^\[[0-?]*[ -/]*[@-~]'
        r'|^[0-?]+[ -/]*[@-~]'
        r'|^\d+;[^\x07]*\x07',
        re.MULTILINE,
    )

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences from raw PTY output."""
        text = self._ANSI_ESCAPE.sub('', text)
        text = self._ANSI_ORPHAN.sub('', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        return text

    async def _log_entry(
        self,
        method_name: str,
        session_id: str,
        output: str,
        log_path: Optional[str] = None,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Truncate output if needed (saving full ANSI-stripped copy inside the
        sandbox), write a structured entry to the unified terminal.log, and
        return the agent-facing (possibly truncated) output string.

        method_name: raw name of the calling method.
        session_id: session this entry belongs to.
        output: command output or status message.
        log_path/start_offset/end_offset: when provided, identify the raw byte
            range in the session's pipe-pane log that produced `output`. Used
            to save the full output inside the sandbox via a sandbox-only
            shell pipeline (tail | head | sed-strip-ANSI | tr-CR) — no bytes
            are uploaded from host to sandbox.
        kwargs: any additional input args rendered as key: value in the log.
        """
        # --- truncation ---
        if len(output) > TRUNCATION_THRESHOLD:
            timestamp = int(time.time())
            remote_dir = "/tmp"
            remote_path = f"{remote_dir}/full_output_{session_id}_{timestamp}.txt"

            n_bytes = (
                end_offset - start_offset
                if (end_offset is not None and start_offset is not None)
                else 0
            )
            if log_path is None or n_bytes <= 0:
                remote_path = "(not saved: no source log range)"
            elif n_bytes > SHELL_OUTPUT_BYTE_CAP:
                remote_path = (
                    f"(not saved: {n_bytes}-byte range exceeds "
                    f"{SHELL_OUTPUT_BYTE_CAP}-byte save cap)"
                )
            else:
                try:
                    # Sandbox-side save: copy the byte range from the
                    # pipe-pane log, strip ANSI escapes, normalize CR→LF.
                    # The entire pipeline runs inside the sandbox — no bytes
                    # cross the host/sandbox boundary. LC_ALL=C avoids the
                    # locale-dependent "Invalid range end" error sed raises
                    # in UTF-8 locales on bracket ranges like [0-?].
                    await self.runtime.exec(f"mkdir -p {shlex.quote(remote_dir)}")
                    inner = (
                        f"tail -c +{start_offset + 1} {shlex.quote(log_path)} "
                        f"| head -c {n_bytes} "
                        f"| sed -E 's,\\x1B(\\[[0-?]*[ -/]*[@-~]|[ -_]),,g' "
                        f"| tr '\\r' '\\n' "
                        f"> {shlex.quote(remote_path)}"
                    )
                    result = await self.runtime.exec(
                        f"LC_ALL=C bash -c {shlex.quote(inner)}"
                    )
                    if result.return_code != 0:
                        raise RuntimeError(result.stderr or "sandbox save failed")
                except Exception as e:
                    logger.warning(
                        "Failed to save full output at %s: %s", remote_path, e,
                    )
                    remote_path = "(failed to save full output)"
            head = output[:TRUNCATION_HEAD]
            tail = output[-TRUNCATION_TAIL:]
            output = (
                head
                + f"\n... [Output truncated. Maximum {TRUNCATION_THRESHOLD} characters"
                f" per shell output. Full output saved in sandbox at {remote_path}."
                f" Do not read the full file back — output is capped at"
                f" {TRUNCATION_THRESHOLD} chars per call.] ...\n"
                + tail
            )

        # --- formatting ---
        ts = datetime.now().isoformat()
        entry = f"\n{'=' * 64}\n[{ts}] {method_name} | session: {session_id}\n{'-' * 64}\n"
        for key, value in kwargs.items():
            if value is not None:
                entry += f"{key}: {value}\n"
        entry += f"\nOUTPUT:\n{output}\n{'=' * 64}\n"

        async with self._log_lock:
            async with aiofiles.open(self._log_file, 'a') as f:
                await f.write(entry)

        return output

    async def _ensure_tmux(self) -> None:
        """Install tmux if absent. Raises RuntimeError if it cannot be installed."""
        result = await self.runtime.exec("which tmux")
        if result.return_code != 0:
            for install_cmd in [
                "apt-get update -qq && apt-get install -y -qq tmux",
                "yum install -y tmux",
                "dnf install -y tmux",
                "apk add --no-cache tmux",
            ]:
                result = await self.runtime.exec(install_cmd)
                if result.return_code == 0:
                    break
            if (await self.runtime.exec("which tmux")).return_code != 0:
                raise RuntimeError(
                    "tmux is not available in the runtime and could not be installed."
                )

    async def _ensure_working_directory(self) -> None:
        """Create working_directory inside the runtime if it does not exist."""
        result = await self.runtime.exec(f"mkdir -p {shlex.quote(self.working_directory)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to create working_directory '{self.working_directory}': {result.stderr}"
            )

    @manual_timeout
    async def _collect_output_until_idle(
        self,
        id: str,
        check_interval: float = 0.1,
        consecutive_empty_limit: int = 3,
        max_wait: float = 30.0,
    ) -> str:
        """
        Poll shell_view every check_interval seconds.
        Return when consecutive_empty_limit empty responses received,
        max_wait exceeded, or accumulated output reaches
        SHELL_OUTPUT_BYTE_CAP (to prevent multi-MB accumulators that
        overflow Daytona's session-exec POST size limit downstream).
        """
        output_parts = []
        total_bytes = 0
        consecutive_empty = 0
        start_time = time.time()

        while time.time() - start_time < max_wait:
            new_output = await self._shell_view(id)
            if new_output:
                # Trim incoming chunk to the remaining cap budget before
                # appending; this is a hard cap (vs append-then-check, which
                # overshoots by one chunk). Slice on bytes to avoid breaking
                # UTF-8 multibyte characters.
                remaining = SHELL_OUTPUT_BYTE_CAP - total_bytes
                chunk_bytes = new_output.encode()
                if len(chunk_bytes) > remaining:
                    new_output = chunk_bytes[:remaining].decode(errors="ignore")
                output_parts.append(new_output)
                total_bytes += len(new_output.encode())
                consecutive_empty = 0
                if total_bytes >= SHELL_OUTPUT_BYTE_CAP:
                    # Stop accumulating — the remainder is still in the
                    # session log and can be retrieved via shell_view.
                    break
            else:
                consecutive_empty += 1
                if consecutive_empty >= consecutive_empty_limit:
                    break
            await asyncio.sleep(check_interval)

        return "".join(output_parts)

    def _tmux_start_session(self, tmux_name: str, log_path: str) -> str:
        """
        Single shell command that creates the tmux session and wires up
        pipe-pane logging in one shot, following the terminus_2 pattern.

        Uses `script -qc` to allocate a PTY so tmux doesn't complain about
        a missing terminal, and chains new-session + pipe-pane with `\\;`.
        """
        return (
            f"export TERM=xterm-256color && "
            f"export SHELL=/bin/bash && "
            f'script -qc "'
            f"tmux new-session -d -s {shlex.quote(tmux_name)} "
            f"-x 220 -y 50 'bash' \\; "
            f"pipe-pane -t {shlex.quote(tmux_name)} "
            f"'cat > {log_path}'"
            f'" /dev/null'
        )

    async def _is_process_running(
        self,
        session_id: str,
        *,
        settle_polls: int = 3,
        settle_interval: float = 0.3,
    ) -> bool:
        """
        Check whether the foreground process in the tmux pane is still running.

        Uses `pane_current_command`. Returns True the moment a non-shell
        process is observed. For multi-stage bash scripts that alternate
        between subprocesses and shell builtins (e.g. `for f in ...; do
        echo "..."; ffmpeg ... ; done`), the pane briefly shows "bash"
        between commands; without a settle window we'd race-detect "done"
        mid-script and the caller would kill the session.

        Settle window: require `settle_polls` consecutive shell observations
        (spaced by `settle_interval` seconds) to declare the pane truly idle.
        Returns True early on any non-shell observation. Worst-case latency
        when truly idle: settle_polls × (RTT + settle_interval).
        """
        tmux_name = self.shell_sessions[session_id]["tmux_name"]
        for i in range(settle_polls):
            result = await self.runtime.exec(
                f"tmux display-message -t {shlex.quote(tmux_name)} -p '#{{pane_current_command}}'"
            )
            if result.return_code != 0:
                return False
            current_cmd = (result.stdout or "").strip()
            if current_cmd not in ("bash", "sh", "zsh", ""):
                return True  # observed a non-shell process — really running
            # shell observation; settle further (skip the final sleep)
            if i < settle_polls - 1:
                await asyncio.sleep(settle_interval)
        return False

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    @manual_timeout
    async def shell_exec(
        self,
        id: str,
        command: str,
        block: bool = True,
    ) -> str:
        r"""Execute a shell command inside a tmux session.

        The command can run in blocking mode (waits for completion) or
        non-blocking mode (runs in the background).

        block=True (default):
            Waits for the command to complete up to ``self.timeout`` seconds.
            If the command finishes in time, returns the full output.
            If timeout is exceeded, the process keeps running and the session
            is converted to a non-blocking session — use shell_view /
            shell_wait / shell_kill_process to monitor it.

        block=False:
            Starts the command in the background and returns immediately with
            initial output. Use shell_view, shell_write_to_process, shell_wait,
            and shell_kill_process to interact with the session.

        Args:
            id (str): A unique identifier for the session.
            command (str): The shell command to execute.
            block (bool): Whether to wait for completion. Defaults to True.

        Returns:
            str: If block is True, returns complete stdout/stderr or a timeout
                message with session monitoring guidance.
                If block is False, returns a start message with initial output.
        """
        async with self._session_lock(id):
            tmux_name = f"terminal_{id}"
            log_path = f"{self.working_directory}/session_{id}.log"

            start_result = await self.runtime.exec(
                self._tmux_start_session(tmux_name, log_path)
            )
            if start_result.return_code != 0:
                error_msg = f"Error: Failed to start session '{id}': {start_result.stderr or start_result.stdout}"
                return await self._log_entry("shell_exec", id, error_msg, command=command, block=block)

            await self.runtime.exec(
                f"tmux send-keys -t {shlex.quote(tmux_name)} {shlex.quote(command)} Enter"
            )

            self.shell_sessions[id] = {
                "tmux_name": tmux_name,
                "log_path": log_path,
                "last_offset": 0,
            }

            if not block:
                output = await self._collect_output_until_idle(id)
                end_offset = self.shell_sessions[id]["last_offset"]
                output = await self._log_entry(
                    "shell_exec", id, output,
                    log_path=log_path, start_offset=0, end_offset=end_offset,
                    command=command, block=False,
                )
                return f"Session '{id}' started.\n\n[Initial Output]:\n{output}"

            # block=True: poll until done or self.block_timeout — no restart,
            # process keeps running in the same tmux session if deadline is
            # exceeded. self.block_timeout is independent of self.timeout (which
            # only controls CAMEL's per-method auto-wrap and may be None).
            poll_interval = 0.5
            deadline = time.time() + self.block_timeout

            while time.time() < deadline:
                await asyncio.sleep(poll_interval)
                if not await self._is_process_running(id):
                    break
            else:
                # Timed out — leave session alive for the agent to monitor
                timeout_msg = (
                    f"Command did not complete within {self.block_timeout} seconds. "
                    f"Session '{id}' is still running.\n\n"
                    f"You can use:\n"
                    f"  - shell_view('{id}') - get current output\n"
                    f"  - shell_wait('{id}', wait_seconds=30) - wait for completion\n"
                    f"  - shell_kill_process('{id}') - terminate"
                )

                # Include partial output collected so far
                partial_start = self.shell_sessions[id]["last_offset"]
                partial = await self._shell_view(id)
                partial_end = self.shell_sessions[id]["last_offset"]
                if partial:
                    if len(partial) > TRUNCATION_THRESHOLD:
                        partial = (
                            partial[:TRUNCATION_HEAD]
                            + "\n... [Output truncated] ...\n"
                            + partial[-TRUNCATION_TAIL:]
                        )
                    timeout_msg += f"\n\n[Partial output so far]:\n{partial}"

                return await self._log_entry(
                    "shell_exec", id, timeout_msg,
                    log_path=log_path, start_offset=partial_start, end_offset=partial_end,
                    command=command, block=True,
                )

            # Completed within timeout — collect full output and clean up
            output = await self._shell_view(id)
            end_offset = self.shell_sessions[id]["last_offset"]
            await self.runtime.exec(f"tmux kill-session -t {shlex.quote(tmux_name)}")
            self.shell_sessions.pop(id)
            # Drop the session lock entry too — session no longer exists.
            self._session_locks.pop(id, None)
            return await self._log_entry(
                "shell_exec", id, output,
                log_path=log_path, start_offset=0, end_offset=end_offset,
                command=command, block=block,
            )

    async def _shell_view(self, id: str) -> str:
        r"""
        Return new output from a non-blocking session since last call.
        Uses file offset on pipe-pane log: tail -c +{offset+1} {log_path}
        Strips ANSI escape sequences and updates last_offset.
        Returns "" if no new output.

        Args:
            id (str): The unique session ID created with shell_exec(block=False).

        Returns:
            str: New output since last shell_view call, or "" if none.
        """
        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        session = self.shell_sessions[id]
        log_path = session["log_path"]
        offset = session["last_offset"]

        # Cap how many bytes we read from the session log per poll. Avoids
        # giving Daytona SDK's demux_log (and downstream string ops) a huge
        # interleaved blob to parse on the event loop.
        result = await self.runtime.exec(
            f"tail -c +{offset + 1} {shlex.quote(log_path)} | head -c {SHELL_OUTPUT_BYTE_CAP}"
        )
        raw = result.stdout or ""
        stripped = self._strip_ansi(raw)

        # advance offset by raw byte length
        session["last_offset"] += len(raw.encode())


        return stripped

    @manual_timeout
    async def shell_view(self, id: str) -> str:
        r"""Retrieve new output from a non-blocking session since the last call.

        If the process has completed, appends a ``[completed]`` marker.
        If the tmux session no longer exists, cleans up and returns an error.

        Args:
            id (str): The session ID created with shell_exec(block=False)
                or converted from a timed-out blocking exec.

        Returns:
            str: New output, or empty string if nothing new.
        """
        async with self._session_lock(id):
            if id not in self.shell_sessions:
                return f"Error: No session '{id}'."

            # Check if the tmux session still exists
            tmux_name = self.shell_sessions[id]["tmux_name"]
            check = await self.runtime.exec(f"tmux has-session -t {shlex.quote(tmux_name)} 2>/dev/null")
            if check.return_code != 0:
                self.shell_sessions.pop(id, None)
                return f"Error: No session '{id}'."

            try:
                session = self.shell_sessions[id]
                log_path = session["log_path"]
                start_offset = session["last_offset"]
                stripped = await self._shell_view(id)
                end_offset = session["last_offset"]
                running = await self._is_process_running(id)
                if not running:
                    stripped += "\n[completed]"
                return await self._log_entry(
                    "shell_view", id, stripped,
                    log_path=log_path, start_offset=start_offset, end_offset=end_offset,
                )
            except Exception as e:
                return await self._log_entry("shell_view", id, f"Error occurred: {e}")
        
    @manual_timeout
    async def shell_write_to_process(self, id: str, command: str) -> str:
        r"""Send input to a running non-blocking session.

        Sends the given text to the process's standard input. A newline
        ``\n`` is automatically appended. Returns the output collected after
        the process becomes idle again.

        Args:
            id (str): The session ID created with shell_exec(block=False).
            command (str): The text to write to the process's stdin.

        Returns:
            str: Output collected after sending the command.
        """
        async with self._session_lock(id):
            if id not in self.shell_sessions:
                return f"Error: No active session '{id}'."

            if not await self._is_process_running(id):
                return f"Error: No active session '{id}'."

            # flush pending output before sending new input
            await self._shell_view(id)

            session = self.shell_sessions[id]
            log_path = session["log_path"]
            start_offset = session["last_offset"]
            tmux_name = session["tmux_name"]
            await self.runtime.exec(
                f"tmux send-keys -t {shlex.quote(tmux_name)} {shlex.quote(command)} Enter"
            )

            output = await self._collect_output_until_idle(id)
            end_offset = session["last_offset"]
            return await self._log_entry(
                "shell_write_to_process", id, output,
                log_path=log_path, start_offset=start_offset, end_offset=end_offset,
                command=command,
            )

    @manual_timeout
    async def shell_wait(self, id: str, wait_seconds: float = 5.0) -> str:
        r"""Wait for a non-blocking process to produce more output or terminate.

        Polls the session every 0.5 seconds for the specified duration and
        collects all output produced during the wait. Uses wall-clock time
        so slow HTTP round-trips count against the budget.

        Args:
            id (str): The session ID created with shell_exec(block=False).
            wait_seconds (float): Maximum seconds to wait (capped at 10.0).
                Defaults to 5.0.

        Returns:
            str: All output collected during the wait period.
        """
        wait_seconds = min(wait_seconds, 10.0)

        async with self._session_lock(id):
            if id not in self.shell_sessions:
                return f"Error: No session '{id}'."

            if not await self._is_process_running(id):
                return "Session is no longer running. Use shell_view to get final output."

            session = self.shell_sessions[id]
            log_path = session["log_path"]
            start_offset = session["last_offset"]

            parts = []
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                await asyncio.sleep(0.5)
                chunk = await self._shell_view(id)
                if chunk:
                    parts.append(chunk)
                # Stop early if the process has exited
                if not await self._is_process_running(id):
                    # One final drain to catch any trailing output
                    final = await self._shell_view(id)
                    if final:
                        parts.append(final)
                    break

            output = "".join(parts)
            end_offset = session["last_offset"]
            return await self._log_entry(
                "shell_wait", id, output,
                log_path=log_path, start_offset=start_offset, end_offset=end_offset,
                wait_seconds=wait_seconds,
            )

    @manual_timeout
    async def shell_kill_process(self, id: str) -> str:
        r"""Terminate a running non-blocking session.

        Kills the tmux session, removes the pipe-pane log, and cleans up
        session state.

        Args:
            id (str): The session ID to terminate.

        Returns:
            str: Confirmation message.
        """
        async with self._session_lock(id):
            if id not in self.shell_sessions:
                return f"Error: No active session '{id}'."

            session = self.shell_sessions.pop(id)
            tmux_name = session["tmux_name"]
            log_path = session["log_path"]

            await self.runtime.exec(f"tmux kill-session -t {shlex.quote(tmux_name)}")
            await self.runtime.exec(f"rm -f {shlex.quote(log_path)}")

            msg = f"Session '{id}' terminated."
        # Drop the lock entry now that the session is gone. Done OUTSIDE the
        # async-with so we hold the lock across the kill, then remove the
        # entry. A concurrent caller arriving after this pop will setdefault
        # a fresh lock and find no session -> clean error path.
        self._session_locks.pop(id, None)
        return await self._log_entry("shell_kill_process", id, msg)

    @manual_timeout
    async def shell_write_content_to_file(self, content: str, file_path: str) -> str:
        r"""Write content to a file inside the runtime, then verify via sha256.

        Encodes content as base64 on the host and decodes it inside the
        sandbox via a single runtime.exec call. Creates parent directories
        as needed. After write, sha256s the file inside the sandbox and
        compares to the host-computed hash; mismatch surfaces as an error
        so silent corruption (truncated POST, encoding mishap, ARG_MAX
        clipping) doesn't masquerade as a successful write. No size cap —
        caller is responsible for staying under Daytona's session-exec POST
        limit and the shell's ARG_MAX.

        Args:
            content (str): The content to write.
            file_path (str): Destination path inside the runtime.

        Returns:
            str: Success or error message.
        """
        try:
            content_bytes = content.encode()
            b64_content = base64.b64encode(content_bytes).decode()
            host_hash = hashlib.sha256(content_bytes).hexdigest()
            quoted_path = shlex.quote(file_path)
            # mkdir -p → write → sha256sum, chained with && so any failure
            # short-circuits. Final stdout line is the remote sha256 to
            # compare against the host hash.
            result = await self.runtime.exec(
                f"mkdir -p \"$(dirname {quoted_path})\" && "
                f"echo {shlex.quote(b64_content)} | base64 -d > {quoted_path} && "
                f"sha256sum {quoted_path} | awk '{{print $1}}'"
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "write failed")
            remote_hash = (result.stdout or "").strip().split()[-1] if result.stdout else ""
            if remote_hash != host_hash:
                raise RuntimeError(
                    f"sha256 mismatch after write: expected "
                    f"{host_hash[:16]}…, got "
                    f"{remote_hash[:16] + '…' if remote_hash else '(empty)'}"
                )
            msg = (
                f"Content written to '{file_path}' "
                f"({len(content_bytes)} bytes, sha256 verified)."
            )
        except Exception as e:
            msg = f"Error writing to '{file_path}': {e}"

        return await self._log_entry(
            "shell_write_content_to_file", "global", msg,
            file_path=file_path, bytes_written=len(content.encode()),
        )

    async def shell_image_read(self, image_path: str) -> Any:
        """
        Read an image file from inside the runtime and return as base64 data URI.

        Args:
            image_path (str): Path to the image inside the runtime.

        Returns:
            ToolResult: Contains text description and base64 data URI image.
        """
        result = await self.runtime.exec(f"test -f {shlex.quote(image_path)}")
        if result.return_code != 0:
            return ToolResult(text=f"Error: File '{image_path}' does not exist in runtime.")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"

        result = await self.runtime.exec(f"base64 -w0 {shlex.quote(image_path)}")
        if result.return_code != 0:
            return ToolResult(text=f"Error reading image '{image_path}': {result.stderr}")

        b64_data = (result.stdout or "").strip()
        data_uri = f"data:{mime_type};base64,{b64_data}"
        return ToolResult(
            text=f"Image read from '{image_path}' ({mime_type}).",
            images=[data_uri],
        )

    async def shell_ask_user_for_help(
        self,
        id: str,
        prompt: str,
        timeout: Optional[float] = None,
    ) -> str:
        r"""
        Pause and ask a human for input. Shows current session output and prompt,
        reads user input via input(), then forwards it to the process.

        Args:
            id (str): The unique session ID of the non-blocking process.
            prompt (str): The question or issue the agent needs help with.
            timeout (Optional[float]): Unused; reserved for future async input support.

        Returns:
            str: Output collected after forwarding the user's response.
        """
        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        current_output = await self._shell_view(id)
        await self._log_entry("shell_ask_user_for_help", id, current_output, prompt=prompt)
        print(f"\n[Session '{id}' current output]:\n{current_output}")
        print(f"\n[Agent needs help]: {prompt}")
        user_input = input("Your input: ")
        return await self.shell_write_to_process(id, user_input)

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
        in the toolkit.

        Each underlying coroutine is wrapped in a **sync** function so that
        CAMEL's FunctionTool.async_call dispatches it to its thread-pool
        executor (function_tool.py:717-731). The sync wrapper then schedules
        the original coroutine on this toolkit's captured event loop via
        ``asyncio.run_coroutine_threadsafe(coro, loop).result()`` — so:

          - The coroutine still runs on the per-step asyncio loop (where its
            ``self.runtime`` async resources were created), preserving
            correctness.
          - The agent's ``await tool()`` from inside the per-step loop now
            yields immediately to the executor instead of awaiting inline;
            other coroutines on the same loop can interleave while the tool
            runs.
          - If a tool ends up doing CPU-bound sync work (e.g. a stuck Daytona
            SDK parse), it blocks the executor's worker thread — NOT the
            per-step loop. Combined with env_service's per-step thread+loop
            isolation, this means one stuck tool only affects its own /step.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        import functools
        import inspect as _inspect

        # Capture the per-step loop at get_tools() time. terminal_env calls
        # this after the runtime is reset, which guarantees we're inside the
        # per-step loop's context.
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.get_running_loop()

        def _make_sync(async_fn):
            """Return a sync wrapper that schedules ``async_fn`` on self._loop."""
            @functools.wraps(async_fn)
            def wrapper(*args, **kwargs):
                coro = async_fn(*args, **kwargs)
                # If the underlying loop is the current running loop (e.g.
                # called from inside an event loop), run_coroutine_threadsafe
                # would deadlock — fall back to direct await via a nested
                # event loop. In our deployment this branch never fires
                # because FunctionTool always dispatches to an executor
                # thread which has no running loop.
                try:
                    running = asyncio.get_running_loop()
                except RuntimeError:
                    running = None
                if running is self._loop:
                    raise RuntimeError(
                        f"TerminalToolkit sync wrapper for "
                        f"{async_fn.__name__} called inside its own event "
                        f"loop — FunctionTool must dispatch to a worker "
                        f"thread first."
                    )
                fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
                return fut.result()

            # Preserve async function's signature so FunctionTool's schema
            # introspection still produces the right parameter list. Hide
            # ``self`` (FunctionTool gets a bound method's signature, not a
            # plain function's, so this is already handled by functools.wraps
            # via __wrapped__).
            wrapper.__signature__ = _inspect.signature(async_fn)
            return wrapper

        return [
            FunctionTool(_make_sync(self.shell_exec)),
            FunctionTool(_make_sync(self.shell_view)),
            FunctionTool(_make_sync(self.shell_wait)),
            FunctionTool(_make_sync(self.shell_write_to_process)),
            FunctionTool(_make_sync(self.shell_kill_process)),
            FunctionTool(_make_sync(self.shell_write_content_to_file)),
            FunctionTool(_make_sync(self.shell_image_read)),
            FunctionTool(_make_sync(self.shell_ask_user_for_help)),
        ]
