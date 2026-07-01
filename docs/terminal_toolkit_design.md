# Terminal Toolkit: Detailed Design and Implementation Plan

This document outlines the detailed design for the TerminalToolkit, a versatile tool for LLM agents to execute and interact with terminal commands in both local and Docker environments. This design is consistent with the latest stable implementation.


## 1. Core Concepts and Structure


### TerminalToolkit Class

The main class is initialized with the desired backend (local or Docker) and global settings like timeout. The mode of execution (blocking vs. non-blocking) is determined per command.

```python
class TerminalToolkit: \
    def __init__( \
        self, \
        use_docker_backend: bool = False, \
        docker_container_name: Optional[str] = None, \
        working_dir: str = "./workspace", \
        session_logs_dir: Optional[str] = None, \
        timeout: int = 60, \
    ): \
        # ... initialization logic ... \
```



* **use_docker_backend**: A boolean flag to select the execution environment.
* **docker_container_name**: Required if the Docker backend is used.
* **working_dir**: The sandboxed directory for local execution.
* **session_logs_dir**: A dedicated directory for logs, separate from the working directory.
* **timeout**: A global timeout for all blocking commands.


### shell_sessions Dictionary

This dictionary is the central component for managing the state of all non-blocking sessions.
```python
self.shell_sessions[id] = { \
    "id": str, \
    "process": Popen_object | Docker_socket, \
    "output_stream": Queue(), \
    "command_history": List[str], \
    "running": bool, \
    "log_file": str, \
    "backend": "local" | "docker", \
    "exec_id": str # Docker only \
} \
```



* **process**: Holds the subprocess.Popen object for local sessions or the Docker socket for Docker sessions.
* **output_stream**: A thread-safe Queue that holds real-time output from the process, populated by a background reader thread.
* **log_file**: The path to the raw, interleaved log file for the session.


## 2. Logging Strategy

The logging is designed to be robust and provide a complete audit trail.



* **Blocking Commands**: All blocking commands and their full output are appended to a single, shared log file: session_logs_dir/blocking_commands.log.
* **Non-Blocking Sessions**: Each non-blocking session gets its own dedicated log file: session_logs_dir/session_{id}.log. This file is a **raw, interleaved transcript** of the entire session, including all commands (prefixed with > ) and all process output, captured in chronological order.


## 3. Core Helper Functions


### _start_output_reader_thread(id)

For every non-blocking session, this function spawns a dedicated background thread. This thread's sole purpose is to continuously read from the process's output stream (stdout or Docker socket) and put the data into the session's output_stream queue and the raw log file. This prevents the main program from blocking and ensures no output is missed.


### _collect_output_until_idle(id, max_wait)

This is the intelligent waiting mechanism. Instead of a fixed time.sleep(), this function actively monitors the session's output_stream. It collects output until the stream has been empty for a stable period (idle_duration). It also includes a max_wait timeout to prevent it from waiting forever on a noisy process. If the timeout is hit, it returns the collected output with a warning. It is also designed to detect if a session has terminated and will return gracefully.


## 4. Tool Implementation Details


### shell_exec(command, block=True, id=None)

This is the primary entry point for executing any command.



1. **Sanitization (Local Only)**: If using the local backend, the command is first passed through _sanitize_command to block dangerous commands and prevent directory traversal outside the working_dir.
2. **Docker Command Wrapping**: If using the Docker backend, the command is automatically wrapped in bash -c "..." to ensure shell features like pipes (|) and redirection (>) work correctly.
3. **Blocking Mode (block=True)**:
    * **Local**: Uses subprocess.run() with the global timeout.
    * **Docker**: Uses docker_api_client.exec_start(), which respects the timeout set during initialization.
    * Logs the command and its full output to blocking_commands.log and returns the output.
4. **Non-Blocking Mode (block=False)**:
    * Creates a new entry in the shell_sessions dictionary.
    * **Local**: Starts the process using subprocess.Popen().
    * **Docker**: Creates and starts a new exec instance, obtaining a socket.
    * Starts the background reader thread using _start_output_reader_thread().
    * Calls _collect_output_until_idle() to wait for and capture the command's initial output.
    * Returns a formatted string containing the new session ID and this initial output.


### shell_write_to_process(id, command)

Sends input to a running non-blocking process.



1. Checks if the session id is valid and running.
2. Calls _collect_output_until_idle() to ensure the process is idle and to capture any lingering output from the previous command.
3. Logs the new command to the session's log file.
4. Writes the command (with a newline appended) to the process's stdin or Docker socket.
5. Calls _collect_output_until_idle() again to capture the result of the new command.
6. Returns the new output.


### shell_view(id)

Retrieves new output from a non-blocking session without waiting.



1. Checks if the session id exists.
2. If the session is no longer marked as running, it performs one final drain of the output_stream queue and returns the content along with a --- SESSION TERMINATED --- message.
3. If the session is running, it simply drains the queue of any content and returns it.


### shell_wait(id, wait_seconds)

Waits for a running process for a specified duration, collecting any output.



1. Checks if the session id is valid and running.
2. Enters a loop for wait_seconds, periodically calling shell_view() to accumulate output.
3. Returns the total output collected during the wait.


### shell_kill_process(id)

Forcibly terminates a non-blocking session.



1. Checks if the session id is valid and running.
2. **Local**: Calls process.terminate() and then process.kill() if necessary.
3. **Docker**: Closes the exec socket, which terminates the process.
4. Sets the session's running flag to False.


### shell_ask_user_for_help(id, prompt)

Engages a human for assistance.



1. Calls _collect_output_until_idle() to get the latest screen state.
2. Prints the latest output and the LLM's prompt to the console.
3. Waits for the human to type a command.
4. Calls shell_write_to_process() with the human's input and returns its result.


### __del__()

A cleanup method registered with atexit to ensure that any running non-blocking sessions are automatically killed when the program exits.