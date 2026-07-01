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
import atexit
import os
import re
import subprocess
import threading
import time
import uuid
import shlex
import select
from datetime import datetime
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from camel.logger import get_logger
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer

logger = get_logger(__name__)

# Try to import docker, but don't make it a hard requirement
try:
    import docker
    from docker.errors import NotFound, APIError
    from docker.models.containers import Container
except ImportError:
    docker = None
    NotFound = None
    APIError = None
    Container = None

# Output truncation constants — keep in sync with terminal_toolkit.py
TRUNCATION_THRESHOLD = 8000  # chars (~2-3K tokens)
TRUNCATION_HEAD = 4000
TRUNCATION_TAIL = 4000


@MCPServer()
class TerminalToolkit(BaseToolkit):
    r"""A toolkit for LLM agents to execute and interact with terminal
    commands in a sandboxed Docker environment.

    All commands are executed inside a Docker container via the Docker API.

    Args:
        timeout (float): The default timeout in seconds for blocking
            commands. Defaults to 20.0.
        docker_container_name (str): The name of the Docker container to use.
        working_directory (Optional[str]): The working directory inside the
            container for exec calls. If None, uses the container's default.
        session_logs_dir (Optional[str]): The directory to store session
            logs on the host. Defaults to a 'terminal_logs' subfolder in
            working_directory.
        safe_mode (bool): Whether to apply security checks. Defaults to
            False (Docker is already sandboxed).
    """

    # ANSI escape stripping — kept in sync with terminal_toolkit.py
    # Each introducer with its own tail (CSI, OSC, SS3, DCS/SOS/PM/APC) MUST
    # be matched before the generic two-char fallback, otherwise the fallback
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
            r'|\][^\x07\x1B]*(?:\x07|\x1B\\)?'      # OSC (BEL or ST terminator)
            r'|O[@-~]'                              # SS3
            r'|[PX^_][^\x1B\x07]*(?:\x07|\x1B\\)?'  # DCS / SOS / PM / APC
            r'|[ -~]'                               # generic 2-byte fallback
        r')'
    )
    # Orphan tail when the leading bytes were consumed by a prior read.
    # Adds OSC orphan (^digits;…BEL) and MULTILINE so ^ matches line starts.
    _ANSI_ORPHAN = re.compile(
        r'^\[[0-?]*[ -/]*[@-~]'
        r'|^[0-?]+[ -/]*[@-~]'
        r'|^\d+;[^\x07]*\x07',
        re.MULTILINE,
    )

    def __init__(
        self,
        timeout: Optional[float] = 20.0,
        docker_container_name: Optional[str] = None,
        working_directory: Optional[str] = None,
        session_logs_dir: Optional[str] = None,
        safe_mode: bool = False,
        # Legacy parameters accepted but ignored for backward compat
        use_docker_backend: bool = True,
        shell_sessions: Optional[Dict[str, Any]] = None,
        need_terminal: bool = False,
    ):
        super().__init__(timeout=timeout)
        self.docker_container_name = docker_container_name
        self.timeout = timeout
        self.shell_sessions: Dict[str, Dict[str, Any]] = {}
        self.working_dir = os.path.abspath(working_directory) if working_directory else None
        self.safe_mode = safe_mode

        atexit.register(self._cleanup)

        self.log_dir = os.path.abspath(
            session_logs_dir or os.path.join(self.working_dir or ".", "terminal_logs")
        )
        # Unified log file — all tool calls go here (matches terminal_toolkit.py)
        self._log_file = os.path.join(self.log_dir, "terminal.log")
        self._log_lock = threading.Lock()
        os.makedirs(self.log_dir, exist_ok=True)

        if docker is None:
            raise ImportError(
                "The 'docker' library is required. "
                "Please install it with 'pip install docker'."
            )
        if not docker_container_name:
            raise ValueError("docker_container_name must be provided.")
        try:
            self.docker_api_client = docker.APIClient(
                base_url='unix://var/run/docker.sock', timeout=self.timeout
            )
            self.docker_client = docker.from_env()
            self.container = self.docker_client.containers.get(docker_container_name)
            print(f"Successfully attached to Docker container '{docker_container_name}'.")
        except NotFound:
            raise RuntimeError(f"Docker container '{docker_container_name}' not found.")
        except APIError as e:
            raise RuntimeError(f"Failed to connect to Docker daemon: {e}")

        # Pre-create /tmp in the container so truncated-output saves work
        # regardless of the image's WORKDIR.  /tmp should always exist but
        # this is a belt-and-suspenders safeguard.
        try:
            mkdir_res = subprocess.run(
                ['docker', 'exec', self.docker_container_name,
                 'mkdir', '-p', '/tmp'],
                capture_output=True, text=True,
            )
            if mkdir_res.returncode != 0:
                logger.warning(
                    "Failed to ensure /tmp in container %s: %s",
                    self.docker_container_name, mkdir_res.stderr.strip(),
                )
        except Exception as e:
            logger.warning(
                "Failed to ensure /tmp in container %s: %s",
                self.docker_container_name, e,
            )

    # ------------------------------------------------------------------
    # ANSI stripping — matches terminal_toolkit.py
    # ------------------------------------------------------------------

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences from raw PTY output."""
        text = self._ANSI_ESCAPE.sub('', text)
        text = self._ANSI_ORPHAN.sub('', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        return text

    # ------------------------------------------------------------------
    # Output truncation helper
    # ------------------------------------------------------------------

    def _truncate_output(self, output: str, session_id: str = "global") -> str:
        r"""Truncate output if it exceeds TRUNCATION_THRESHOLD.

        Saves the full output inside the container at
        ``/tmp/full_output_{session_id}_{timestamp}.txt`` and returns
        head + tail with a pointer to the saved file.
        """
        if len(output) <= TRUNCATION_THRESHOLD:
            return output

        timestamp = int(time.time())
        remote_dir = "/tmp"
        remote_path = f"{remote_dir}/full_output_{session_id}_{timestamp}.txt"

        # Best-effort save of full output inside container. On any failure we
        # log the explicit reason so debugging is possible and fall back to a
        # placeholder path in the agent-visible message.
        temp_host = os.path.join(self.log_dir, f"_full_{uuid.uuid4().hex}.txt")
        try:
            with open(temp_host, "w", encoding="utf-8") as f:
                f.write(output)
            # Ensure target dir exists in the container before copying
            mkdir_res = subprocess.run(
                ['docker', 'exec', self.docker_container_name,
                 'mkdir', '-p', remote_dir],
                capture_output=True, text=True,
            )
            if mkdir_res.returncode != 0:
                raise RuntimeError(
                    f"docker exec mkdir failed (rc={mkdir_res.returncode}): "
                    f"stderr={mkdir_res.stderr.strip()}"
                )
            cp_res = subprocess.run(
                ['docker', 'cp', temp_host,
                 f"{self.docker_container_name}:{remote_path}"],
                capture_output=True, text=True,
            )
            if cp_res.returncode != 0:
                raise RuntimeError(
                    f"docker cp failed (rc={cp_res.returncode}): "
                    f"stderr={cp_res.stderr.strip()}"
                )
        except Exception as e:
            logger.warning(
                "Failed to save full output to container %s at %s: %s",
                self.docker_container_name, remote_path, e,
            )
            remote_path = "(failed to save full output)"
        finally:
            if os.path.exists(temp_host):
                try:
                    os.remove(temp_host)
                except OSError:
                    pass

        return (
            output[:TRUNCATION_HEAD]
            + f"\n... [Output truncated. Full output saved at: {remote_path}] ...\n"
            + output[-TRUNCATION_TAIL:]
        )

    # ------------------------------------------------------------------
    # Unified structured logging — matches terminal_toolkit.py format
    # ------------------------------------------------------------------

    def _log_entry(self, method_name: str, session_id: str, output: str, **kwargs) -> str:
        """Truncate output if needed, write a structured entry to terminal.log,
        and return the agent-facing (possibly truncated) output string."""
        output = self._truncate_output(output, session_id)

        ts = datetime.now().isoformat()
        entry = f"\n{'=' * 64}\n[{ts}] {method_name} | session: {session_id}\n{'-' * 64}\n"
        for key, value in kwargs.items():
            if value is not None:
                entry += f"{key}: {value}\n"
        entry += f"\nOUTPUT:\n{output}\n{'=' * 64}\n"

        with self._log_lock:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(entry)
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_output_reader_thread(self, session_id: str):
        r"""Starts a daemon thread to read stdout from a Docker exec socket."""
        session = self.shell_sessions[session_id]
        # Internal per-session stream log for raw capture
        stream_log = os.path.join(self.log_dir, f"_stream_{session_id}.log")

        def reader():
            try:
                socket = session["process"]._sock
                while True:
                    if socket.fileno() == -1:
                        break
                    ready, _, _ = select.select([socket], [], [], 0.1)
                    if ready:
                        data = socket.recv(4096)
                        if not data:
                            break
                        decoded_data = data.decode('utf-8', errors='ignore')
                        session["output_stream"].put(decoded_data)
                        with open(stream_log, "a", encoding="utf-8") as f:
                            f.write(decoded_data)
                    if not self.docker_api_client.exec_inspect(session["exec_id"])['Running']:
                        break
            except Exception:
                pass
            finally:
                session["running"] = False

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

    def _collect_output_until_idle(
        self, id: str, idle_duration: float = 0.5,
        check_interval: float = 0.1, max_wait: float = 5.0
    ) -> str:
        r"""Collects output from a session until it's idle or max_wait is reached.

        Args:
            id: The session ID.
            idle_duration: How long the stream must be empty to be
                considered idle.
            check_interval: Sleep time between checks.
            max_wait: Maximum total time to wait.

        Returns:
            The collected output.
        """
        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        output_parts = []
        idle_time = 0
        start_time = time.time()

        while time.time() - start_time < max_wait:
            new_output = self._drain_queue(id)

            if new_output is None:
                # session gone
                return f"Error: No session '{id}'."

            if new_output:
                output_parts.append(new_output)
                idle_time = 0
            else:
                idle_time += check_interval
                if idle_time >= idle_duration:
                    return self._strip_ansi("".join(output_parts))
            time.sleep(check_interval)

        final_output = self._drain_queue(id)
        if final_output:
            output_parts.append(final_output)

        return self._strip_ansi("".join(output_parts))

    def _drain_queue(self, id: str) -> Optional[str]:
        r"""Drain the output queue for a session. Returns None if session
        doesn't exist, empty string if nothing new."""
        if id not in self.shell_sessions:
            return None
        session = self.shell_sessions[id]
        parts = []
        try:
            while True:
                parts.append(session["output_stream"].get_nowait())
        except Empty:
            pass
        return "".join(parts)

    def _convert_to_session(self, session_id: str, exec_id: str,
                            command: str, chunks: list) -> None:
        r"""Convert a timed-out blocking exec into a tracked non-blocking
        session so the agent can monitor it via shell_view/shell_wait."""
        output_queue: Queue = Queue()
        # Feed already-collected chunks into the queue
        for chunk in chunks:
            output_queue.put(chunk)

        self.shell_sessions[session_id] = {
            "id": session_id,
            "process": None,  # no socket for stream-based exec
            "output_stream": output_queue,
            "command_history": [command],
            "running": True,
            "exec_id": exec_id,
            "timeout_converted": True,
        }

        # Start a thread that continues reading from the docker exec stream
        def continue_reading():
            try:
                while True:
                    inspect = self.docker_api_client.exec_inspect(exec_id)
                    if not inspect.get('Running', False):
                        break
                    time.sleep(0.5)
            except Exception:
                pass
            finally:
                # The session may have been removed (shell_kill_process /
                # _cleanup) or — worse — the same id may have been recycled
                # by a fresh exec. Only mark `running=False` if the entry
                # still belongs to *our* exec.
                sess = self.shell_sessions.get(session_id)
                if sess is not None and sess.get("exec_id") == exec_id:
                    sess["running"] = False

        thread = threading.Thread(target=continue_reading, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Public tools
    # ------------------------------------------------------------------

    def shell_exec(self, id: str, command: str, block: bool = True) -> str:
        r"""Execute a shell command inside the Docker container.

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
        docker_command = ['bash', '-c', command]
        session_id = id

        if block:
            # --- BLOCKING EXECUTION ---
            try:
                exec_instance = self.docker_api_client.exec_create(
                    self.container.id, docker_command, workdir=self.working_dir
                )
                exec_id = exec_instance["Id"]
                chunks: list[str] = []

                def stream_output():
                    try:
                        for chunk in self.docker_api_client.exec_start(exec_id, stream=True):
                            if chunk:
                                chunks.append(chunk.decode("utf-8", errors="ignore"))
                    except Exception:
                        pass

                stream_thread = threading.Thread(target=stream_output, daemon=True)
                stream_thread.start()
                stream_thread.join(timeout=self.timeout)

                if stream_thread.is_alive():
                    # --- TIMEOUT: convert to non-blocking session ---
                    self._convert_to_session(session_id, exec_id, command, chunks)

                    partial = self._strip_ansi("".join(chunks))
                    if partial:
                        partial = self._truncate_output(partial, session_id)

                    timeout_msg = (
                        f"Command did not complete within {self.timeout} seconds. "
                        f"Session '{session_id}' is still running.\n\n"
                        f"You can use:\n"
                        f"  - shell_view('{session_id}') - get current output\n"
                        f"  - shell_wait('{session_id}', wait_seconds=30) - wait for completion\n"
                        f"  - shell_kill_process('{session_id}') - terminate"
                    )
                    if partial:
                        timeout_msg += f"\n\n[Partial output so far]:\n{partial}"

                    return self._log_entry(
                        "shell_exec", session_id, timeout_msg,
                        command=command, block=True,
                    )

                output = self._strip_ansi("".join(chunks))
                return self._log_entry(
                    "shell_exec", session_id, output,
                    command=command, block=True,
                )

            except Exception as e:
                if "Read timed out" in str(e):
                    error_msg = f"Error: Command timed out after {self.timeout} seconds."
                else:
                    error_msg = f"Error executing command: {e}"
                return self._log_entry(
                    "shell_exec", session_id, error_msg,
                    command=command, block=True,
                )
        else:
            # --- NON-BLOCKING EXECUTION ---
            self.shell_sessions[session_id] = {
                "id": session_id, "process": None, "output_stream": Queue(),
                "command_history": [command], "running": True,
            }

            try:
                exec_instance = self.docker_api_client.exec_create(
                    self.container.id, docker_command, stdin=True, tty=True,
                    workdir=self.working_dir
                )
                exec_id = exec_instance['Id']
                exec_socket = self.docker_api_client.exec_start(
                    exec_id, tty=True, stream=True, socket=True
                )
                self.shell_sessions[session_id]["process"] = exec_socket
                self.shell_sessions[session_id]["exec_id"] = exec_id

                self._start_output_reader_thread(session_id)
                initial_output = self._collect_output_until_idle(session_id)

                msg = f"Session '{session_id}' started.\n\n[Initial Output]:\n{initial_output}"
                return self._log_entry(
                    "shell_exec", session_id, msg,
                    command=command, block=False,
                )

            except Exception as e:
                self.shell_sessions[session_id]["running"] = False
                error_msg = f"Error starting non-blocking command: {e}"
                return self._log_entry(
                    "shell_exec", session_id, error_msg,
                    command=command, block=False,
                )

    def shell_write_to_process(self, id: str, command: str) -> str:
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
        if id not in self.shell_sessions or not self.shell_sessions[id]["running"]:
            return f"Error: No active session '{id}'."

        # Flush any lingering output from previous commands.
        self._collect_output_until_idle(id, idle_duration=0.3, max_wait=2.0)

        session = self.shell_sessions[id]
        session["command_history"].append(command)

        try:
            socket = session["process"]._sock
            socket.sendall((command + '\n').encode('utf-8'))
            output = self._collect_output_until_idle(id)
            return self._log_entry("shell_write_to_process", id, output, command=command)
        except Exception as e:
            error_msg = f"Error writing to session '{id}': {e}"
            return self._log_entry("shell_write_to_process", id, error_msg, command=command)

    def _shell_view(self, id: str) -> str:
        r"""Raw drain of a session's output queue with ANSI stripping.
        Returns new output since the last call (empty string if none)."""
        session = self.shell_sessions[id]
        parts = []
        try:
            while True:
                parts.append(session["output_stream"].get_nowait())
        except Empty:
            pass
        return self._strip_ansi("".join(parts))

    def shell_view(self, id: str) -> str:
        r"""Retrieve new output from a non-blocking session since the last call.

        If the process has terminated, drains the output queue and appends a
        ``[completed]`` marker. If the process is still running, returns any
        new output (empty string if none).

        Args:
            id (str): The session ID created with shell_exec(block=False)
                or converted from a timed-out blocking exec.

        Returns:
            str: New output, or empty string if nothing new.
        """
        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        output = self._shell_view(id)

        if not self.shell_sessions[id]["running"]:
            output += "\n[completed]"

        return self._log_entry("shell_view", id, output)

    def shell_wait(self, id: str, wait_seconds: float = 5.0) -> str:
        r"""Wait for a non-blocking process to produce more output or terminate.

        Polls the session every 0.5 seconds for the specified duration and
        collects all output produced during the wait. Uses wall-clock time.

        Args:
            id (str): The session ID created with shell_exec(block=False).
            wait_seconds (float): Maximum seconds to wait (capped at 10.0).
                Defaults to 5.0.

        Returns:
            str: All output collected during the wait period.
        """
        wait_seconds = min(wait_seconds, 10.0)

        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        session = self.shell_sessions[id]
        if not session["running"]:
            return "Session is no longer running. Use shell_view to get final output."

        output_collected = []
        end_time = time.time() + wait_seconds
        while time.time() < end_time and session["running"]:
            new_output = self._shell_view(id)
            if new_output:
                output_collected.append(new_output)
            time.sleep(0.5)

        # Final drain — pick up output that arrived after the process exited
        final = self._shell_view(id)
        if final:
            output_collected.append(final)

        output = "".join(output_collected)
        return self._log_entry("shell_wait", id, output, wait_seconds=wait_seconds)

    def shell_kill_process(self, id: str) -> str:
        r"""Terminate a running non-blocking session.

        Closes the Docker exec socket which terminates the process, and
        cleans up session state.

        Args:
            id (str): The session ID to terminate.

        Returns:
            str: Confirmation message.
        """
        if id not in self.shell_sessions or not self.shell_sessions[id]["running"]:
            return f"Error: No active session '{id}'."

        session = self.shell_sessions.pop(id)
        try:
            if session.get("process"):
                session["process"].close()
            session["running"] = False
            msg = f"Session '{id}' terminated."
            return self._log_entry("shell_kill_process", id, msg)
        except Exception as e:
            msg = f"Error killing session '{id}': {e}"
            return self._log_entry("shell_kill_process", id, msg)

    def shell_ask_user_for_help(self, id: str, prompt: str) -> str:
        r"""Pause and ask a human for help with an interactive session.

        Displays the current session output and the agent's prompt to the
        user, waits for user input, sends it to the session, and returns
        the resulting output.

        Args:
            id (str): The session ID of the interactive process.
            prompt (str): The question or instruction from the agent.

        Returns:
            str: Output collected after forwarding the user's response.
        """
        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        last_output = self._collect_output_until_idle(id)
        self._log_entry("shell_ask_user_for_help", id, last_output, prompt=prompt)

        print("\n" + "=" * 60)
        print("LLM Agent needs your help!")
        print(f"SESSION ID: {id}")
        print(f"PROMPT: {prompt}")
        print("--- LAST OUTPUT ---")
        print(last_output.strip())
        print("-------------------")

        try:
            user_input = input("Your input: ")
        except EOFError:
            user_input = ""

        return self.shell_write_to_process(id, user_input)

    def shell_write_content_to_file(self, content: str, file_path: str) -> str:
        r"""Write content to a file inside the Docker container.

        Uses ``docker cp`` to transfer a temporary host file into the
        container at the specified path.

        Args:
            content (str): The content to write.
            file_path (str): Destination path inside the container.

        Returns:
            str: Success or error message.
        """
        try:
            temp_host_path = os.path.join(self.log_dir, f"temp_{uuid.uuid4().hex}.txt")
            with open(temp_host_path, "w", encoding="utf-8") as f:
                f.write(content)
            subprocess.run(
                ['docker', 'cp', temp_host_path,
                 f"{self.docker_container_name}:{file_path}"],
                check=True, capture_output=True, text=True,
            )
            os.remove(temp_host_path)
            msg = f"Content written to '{file_path}'."
        except subprocess.CalledProcessError as e:
            msg = f"Error writing to '{file_path}': {e.stderr}"
        except Exception as e:
            msg = f"Error writing to '{file_path}': {e}"

        return self._log_entry("shell_write_content_to_file", "global", msg, file_path=file_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _cleanup(self):
        """Tear down all session sockets and the docker clients.

        Idempotent — safe to call from both ``atexit`` and ``__del__``.
        After it runs, the toolkit instance is no longer usable.

        Three layers of resources to release, in order:
          1. per-session non-blocking exec sockets (``session["process"]``).
             Each non-blocking ``shell_exec`` opens a persistent unix
             socket via ``exec_start(..., socket=True)``; if the reader
             thread already flipped ``running=False`` when the process
             exited, ``shell_kill_process`` would skip it and the socket
             would leak. Close every session's socket directly.
          2. the shared ``docker_api_client`` and ``docker_client`` —
             each holds a requests session and a unix-socket connection
             pool that leaks fds across many task instances if not closed.
          3. the cached ``container`` reference (it back-references the
             high-level client and would keep it alive otherwise).
        """
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True

        # 1. Close every session's exec socket. We deliberately do NOT
        #    gate on ``running`` — a session whose reader thread already
        #    flipped ``running=False`` may still own an unclosed socket.
        for session_id in list(self.shell_sessions.keys()):
            session = self.shell_sessions.get(session_id) or {}
            sock = session.get("process")
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            session["running"] = False
        self.shell_sessions.clear()

        # 2. Tear down shared docker clients.
        for attr in ("docker_api_client", "docker_client"):
            client = getattr(self, attr, None)
            if client is None:
                continue
            try:
                client.close()
            except Exception:
                pass
            setattr(self, attr, None)

        # 3. Drop container reference so the high-level client can be GC'd.
        self.container = None

    def __del__(self):
        try:
            self._cleanup()
        except Exception:
            pass

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
        in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        return [
            FunctionTool(self.shell_exec),
            FunctionTool(self.shell_view),
            FunctionTool(self.shell_wait),
            FunctionTool(self.shell_write_to_process),
            FunctionTool(self.shell_kill_process),
            FunctionTool(self.shell_write_content_to_file),
            FunctionTool(self.shell_ask_user_for_help),
        ]
