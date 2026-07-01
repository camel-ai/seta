"""
Plan 04 — TerminalToolkit connected to a live Docker runtime.
Run directly: python test/test_terminal_toolkit.py

Uses: terminal-bench-2.0/nginx-request-logging (prebuilt image, fast start).
One container shared across all tests.
"""
import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
from seta_env.toolkits.terminal_toolkit import TerminalToolkit

_REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = str(_REPO_ROOT / "dataset/terminal-bench-2.0/nginx-request-logging")
TRIAL_ROOT = str(_REPO_ROOT / "test/output/trials")

PASS = 0
FAIL = 0

def sid(prefix="tk"):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")

def fail(name, reason):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}: {reason}")


# ---------------------------------------------------------------------------
# Init tests (no container needed)
# ---------------------------------------------------------------------------

def run_init_tests(log_dir):
    from seta_env.toolkits.terminal_toolkit import TerminalToolkit

    class FakeRuntime:
        pass

    # runtime=None raises ValueError
    try:
        TerminalToolkit(working_directory="/workdir", session_logs_dir=log_dir, runtime=None)
        fail("init_no_runtime", "expected ValueError")
    except ValueError:
        ok("init_no_runtime")

    # working_directory=None raises AssertionError
    try:
        TerminalToolkit(working_directory=None, session_logs_dir=log_dir, runtime=FakeRuntime())
        fail("init_no_working_dir", "expected AssertionError")
    except AssertionError:
        ok("init_no_working_dir")

    # log dir created on init
    tk = TerminalToolkit(working_directory="/workdir", session_logs_dir=log_dir, runtime=FakeRuntime())
    if tk._log_file.parent.exists():
        ok("init_log_dir_created")
    else:
        fail("init_log_dir_created", "log dir not created")

    # get_tools returns 8
    tools = tk.get_tools()
    names = [t.get_function_name() for t in tools]
    if len(tools) == 8 and "shell_exec" in names and "shell_image_read" in names:
        ok("get_tools_returns_8")
    else:
        fail("get_tools_returns_8", f"got {len(tools)} tools: {names}")


# ---------------------------------------------------------------------------
# Live tests (need container)
# ---------------------------------------------------------------------------

async def run_live_tests(rt, log_dir):
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=log_dir, runtime=rt)

    # --- blocking shell_exec ---

    out = await tk.shell_exec(sid(), "echo hello")
    if "hello" in out:
        ok("shell_exec_block_basic")
    else:
        fail("shell_exec_block_basic", repr(out))

    out = await tk.shell_exec(sid(), "echo err >&2")
    if "err" in out:
        ok("shell_exec_block_stderr_merged")
    else:
        fail("shell_exec_block_stderr_merged", repr(out))

    out = await tk.shell_exec(sid(), "false")
    if isinstance(out, str):
        ok("shell_exec_block_nonzero_no_raise")
    else:
        fail("shell_exec_block_nonzero_no_raise", "raised exception")

    # safe_mode cwd
    tk_safe = await TerminalToolkit(working_directory="/workdir", session_logs_dir=log_dir,
                                    safe_mode=True, runtime=rt)
    out = await tk_safe.shell_exec(sid(), "pwd")
    if "/workdir" in out:
        ok("shell_exec_block_safe_mode_cwd")
    else:
        fail("shell_exec_block_safe_mode_cwd", repr(out))

    # log file written after exec
    await tk.shell_exec(sid(), "echo logged_line")
    if tk._log_file.exists() and "logged_line" in tk._log_file.read_text():
        ok("shell_exec_block_logs_output")
    else:
        fail("shell_exec_block_logs_output", "not in log")

    # --- non-blocking shell_exec ---

    session_id = sid("nb")
    out = await tk.shell_exec(session_id, "echo nonblocking", block=False)
    if session_id in tk.shell_sessions:
        ok("shell_exec_nonblocking_session_created")
    else:
        fail("shell_exec_nonblocking_session_created", "not in shell_sessions")

    if f"Session {session_id} started" in out:
        ok("shell_exec_nonblocking_start_message")
    else:
        fail("shell_exec_nonblocking_start_message", repr(out[:80]))

    session_id = sid("nb")
    out = await tk.shell_exec(session_id, "echo quick_done", block=False)
    if "quick_done" in out:
        ok("shell_exec_nonblocking_output_visible")
    else:
        fail("shell_exec_nonblocking_output_visible", repr(out[:80]))

    if tk._tmux_checked:
        ok("shell_exec_nonblocking_tmux_checked")
    else:
        fail("shell_exec_nonblocking_tmux_checked", "_tmux_checked still False")

    # --- shell_view ---

    out = await tk.shell_view("no_such_session")
    if "Error: No session" in out:
        ok("shell_view_unknown_session")
    else:
        fail("shell_view_unknown_session", repr(out))

    session_id = sid("view")
    await tk.shell_exec(session_id, "echo view_test", block=False)
    offset_before = tk.shell_sessions[session_id]["last_offset"]
    await tk.shell_view(session_id)
    offset_after = tk.shell_sessions[session_id]["last_offset"]
    # offset only advances if there was new output; either way it should not go backwards
    if offset_after >= offset_before:
        ok("shell_view_offset_advances")
    else:
        fail("shell_view_offset_advances", f"offset went from {offset_before} to {offset_after}")

    # --- shell_write_to_process ---

    out = await tk.shell_write_to_process("no_such", "hello")
    if "Error: No active session" in out:
        ok("shell_write_unknown_session")
    else:
        fail("shell_write_unknown_session", repr(out))

    session_id = sid("py")
    await tk.shell_exec(session_id, "python3", block=False)
    await asyncio.sleep(0.5)
    out = await tk.shell_write_to_process(session_id, "print(40 + 2)")
    if "42" in out:
        ok("shell_write_to_python_repl")
    else:
        fail("shell_write_to_python_repl", repr(out[:80]))

    # --- shell_wait ---

    out = await tk.shell_wait("no_such", wait_seconds=1)
    if "Error: No session" in out:
        ok("shell_wait_unknown_session")
    else:
        fail("shell_wait_unknown_session", repr(out))

    session_id = sid("wait")
    await tk.shell_exec(session_id, "sleep 10", block=False)
    t0 = time.time()
    await tk.shell_wait(session_id, wait_seconds=2)
    elapsed = time.time() - t0
    if 1.5 < elapsed < 4.0:
        ok("shell_wait_duration_respected")
    else:
        fail("shell_wait_duration_respected", f"elapsed={elapsed:.2f}s")

    # --- shell_kill_process ---

    out = await tk.shell_kill_process("no_such")
    if "Error: No session" in out:
        ok("shell_kill_unknown_session")
    else:
        fail("shell_kill_unknown_session", repr(out))

    session_id = sid("kill")
    await tk.shell_exec(session_id, "sleep 60", block=False)
    out = await tk.shell_kill_process(session_id)
    if f"Session '{session_id}' terminated" in out:
        ok("shell_kill_live_session")
    else:
        fail("shell_kill_live_session", repr(out))

    session_id = sid("kill2")
    await tk.shell_exec(session_id, "sleep 60", block=False)
    tmux_name = tk.shell_sessions[session_id]["tmux_name"]
    await tk.shell_kill_process(session_id)
    if session_id not in tk.shell_sessions:
        ok("shell_kill_removes_from_state")
    else:
        fail("shell_kill_removes_from_state", "still in shell_sessions")
    result = await rt.harbor_env.exec("tmux ls 2>&1 || true")
    if tmux_name not in (result.stdout or ""):
        ok("shell_kill_tmux_gone")
    else:
        fail("shell_kill_tmux_gone", f"still in tmux ls: {result.stdout}")

    # --- shell_write_content_to_file ---

    content = "hello from toolkit"
    path = "/workdir/toolkit_test.txt"
    out = await tk.shell_write_content_to_file(content, path)
    result = await rt.harbor_env.exec(f"cat {path}")
    if f"Content written to '{path}'" in out and content in (result.stdout or ""):
        ok("write_content_to_file_success")
    else:
        fail("write_content_to_file_success", f"out={repr(out[:60])}, cat={repr(result.stdout)}")

    path2 = "/workdir/empty_test.txt"
    out = await tk.shell_write_content_to_file("", path2)
    result = await rt.harbor_env.exec(f"test -f {path2} && echo exists")
    if "Content written to" in out and "exists" in (result.stdout or ""):
        ok("write_content_to_file_empty")
    else:
        fail("write_content_to_file_empty", repr(out[:60]))

    # --- shell_image_read ---

    result = await tk.shell_image_read("/nonexistent/image.png")
    if "does not exist" in result.text:
        ok("image_read_nonexistent")
    else:
        fail("image_read_nonexistent", repr(result.text))

    png_path = "/workdir/test_image.png"
    make_png = (
        "python3 -c \""
        "import struct,zlib;"
        "w=lambda t,d: struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff);"
        "png=b'\\x89PNG\\r\\n\\x1a\\n'+w(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))"
        "+w(b'IDAT',zlib.compress(b'\\x00\\xff\\xff\\xff'))+w(b'IEND',b'');"
        "open('/workdir/test_image.png','wb').write(png)\""
    )
    await rt.harbor_env.exec(make_png)
    result = await tk.shell_image_read(png_path)
    if hasattr(result, "images") and result.images and "data:image/png;base64," in result.images[0]:
        ok("image_read_valid_png")
    else:
        fail("image_read_valid_png", repr(getattr(result, "text", result)))

    # --- truncation ---

    out = await tk.shell_exec(sid(), "python3 -c \"print('X' * 2000, end='')\"")
    if "[Output truncated" not in out:
        ok("no_truncation_at_2000")
    else:
        fail("no_truncation_at_2000", "was truncated")

    out = await tk.shell_exec(sid(), "python3 -c \"print('Y' * 2001, end='')\"")
    if "[Output truncated" in out:
        ok("truncation_above_2000")
    else:
        fail("truncation_above_2000", "not truncated")


async def main():
    import tempfile
    log_dir = str(_REPO_ROOT / "test/output/toolkit_logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print("\n=== Init tests (no container) ===")
    run_init_tests(log_dir)

    print("\n=== Live tests (shared container) ===")
    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid("main"),
        environment_type="docker",
    )
    t0 = time.time()
    await rt.reset()
    print(f"  container ready in {time.time()-t0:.1f}s\n")

    try:
        await run_live_tests(rt, log_dir)
    finally:
        await rt.stop(delete=True)

    print(f"\n{'='*40}")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
