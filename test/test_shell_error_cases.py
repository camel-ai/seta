"""
Tests targeting specific shell_* error cases observed in evaluation logs.

Covers:
  1. tmux init: all install methods tried / fallback chain / fail-fast on init
  2. shell_view called with wrong kwarg 'command'
  3. shell_wait silent/empty exception from dead runtime connection
  4. shell_exec blocking timeout capped at 25s
  5. shell_wait when tmux session has disappeared
  6. shell_exec non-blocking fails when tmux session start fails

Run: python test/test_shell_error_cases.py
"""
import asyncio
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from seta_env.toolkits.terminal_toolkit import TerminalToolkit

PASS = 0
FAIL = 0
LOG_DIR = "/tmp/test_shell_error_cases_logs"


def sid(prefix="t"):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")


def fail(name, reason):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}: {reason}")


def make_result(stdout="", stderr="", return_code=0):
    r = MagicMock()
    r.stdout = stdout
    r.stderr = stderr
    r.return_code = return_code
    return r


def make_runtime(exec_side_effect=None, exec_return=None):
    """Return a mock runtime whose exec() is controlled by the caller."""
    runtime = MagicMock()
    if exec_side_effect is not None:
        runtime.exec = AsyncMock(side_effect=exec_side_effect)
    else:
        runtime.exec = AsyncMock(return_value=exec_return or make_result())
    return runtime


def tmux_ok_exec(cmd, **_kw):
    """Default exec: tmux present, mkdir succeeds, everything else succeeds."""
    return make_result()


# ---------------------------------------------------------------------------
# 1. tmux init — fail-fast at TerminalToolkit construction
# ---------------------------------------------------------------------------

async def test_tmux_not_available_fails_init():
    """
    await TerminalToolkit(...) must raise RuntimeError immediately when tmux
    cannot be installed — no tools are returned to the agent.
    Matches: Error executing async tool 'shell_exec': tmux is not available...
    """
    async def exec_fn(cmd, **_kw):
        return make_result(return_code=1, stderr="not found")

    rt = make_runtime(exec_side_effect=exec_fn)
    try:
        await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)
        fail("tmux_not_available_fails_init", "expected RuntimeError")
    except RuntimeError as e:
        if "tmux is not available" in str(e):
            ok("tmux_not_available_fails_init")
        else:
            fail("tmux_not_available_fails_init", f"unexpected message: {e}")


async def test_tmux_all_pkg_managers_tried():
    """
    _ensure_tmux must try all four package managers before giving up.
    Note: apk uses 'add', not 'install'.
    """
    tried = []

    async def exec_fn(cmd, **_kw):
        if "apt-get" in cmd and "install" in cmd:
            tried.append("apt-get")
        elif "yum" in cmd and "install" in cmd:
            tried.append("yum")
        elif "dnf" in cmd and "install" in cmd:
            tried.append("dnf")
        elif "apk" in cmd and "add" in cmd:
            tried.append("apk")
        return make_result(return_code=1, stderr="not found")

    rt = make_runtime(exec_side_effect=exec_fn)
    try:
        await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)
    except RuntimeError:
        pass

    if tried == ["apt-get", "yum", "dnf", "apk"]:
        ok("tmux_all_pkg_managers_tried")
    else:
        fail("tmux_all_pkg_managers_tried", f"tried={tried}")


async def test_tmux_install_succeeds_via_apt():
    """
    apt-get success must stop the install chain. No yum/dnf/apk attempted.
    Toolkit construction must succeed.
    """
    tried = []
    apt_installed = False

    async def exec_fn(cmd, **_kw):
        nonlocal apt_installed
        if "which tmux" in cmd:
            return make_result(return_code=0 if apt_installed else 1)
        if "apt-get" in cmd and "install" in cmd:
            tried.append("apt-get")
            apt_installed = True
            return make_result(return_code=0)
        for pm in ("yum", "dnf", "apk"):
            if pm in cmd:
                tried.append(pm)
        return make_result(return_code=0)

    rt = make_runtime(exec_side_effect=exec_fn)
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    if tried == ["apt-get"] and tk is not None:
        ok("tmux_install_succeeds_via_apt")
    else:
        fail("tmux_install_succeeds_via_apt", f"tried={tried}")


async def test_tmux_install_apt_fails_falls_back_to_apk():
    """
    apt/yum/dnf fail → apk (Alpine) succeeds. Construction must succeed.
    apk uses 'apk add --no-cache tmux', not 'apk install'.
    """
    tried = []
    apk_installed = False

    async def exec_fn(cmd, **_kw):
        nonlocal apk_installed
        if "which tmux" in cmd:
            return make_result(return_code=0 if apk_installed else 1)
        if "apt-get" in cmd and "install" in cmd:
            tried.append("apt-get")
            return make_result(return_code=1)
        if "yum" in cmd and "install" in cmd:
            tried.append("yum")
            return make_result(return_code=1)
        if "dnf" in cmd and "install" in cmd:
            tried.append("dnf")
            return make_result(return_code=1)
        if "apk" in cmd and "add" in cmd:
            tried.append("apk")
            apk_installed = True
            return make_result(return_code=0)
        return make_result(return_code=0)

    rt = make_runtime(exec_side_effect=exec_fn)
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    if tried == ["apt-get", "yum", "dnf", "apk"] and tk is not None:
        ok("tmux_install_apt_fails_falls_back_to_apk")
    else:
        fail("tmux_install_apt_fails_falls_back_to_apk", f"tried={tried}")


async def test_tmux_install_ok_but_which_fails():
    """
    Install returns rc=0 but 'which tmux' still fails after → still raises.
    Catches silent failure: package manager exits 0 but tmux ends up out of PATH.
    """
    async def exec_fn(cmd, **_kw):
        return make_result(return_code=1 if "which tmux" in cmd else 0)

    rt = make_runtime(exec_side_effect=exec_fn)
    try:
        await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)
        fail("tmux_install_ok_but_which_fails", "expected RuntimeError")
    except RuntimeError as e:
        if "tmux is not available" in str(e):
            ok("tmux_install_ok_but_which_fails")
        else:
            fail("tmux_install_ok_but_which_fails", f"unexpected: {e}")


async def test_tmux_already_present_no_install():
    """tmux present from the start → no install attempted, construction succeeds."""
    install_called = False

    async def exec_fn(cmd, **_kw):
        nonlocal install_called
        if any(pm in cmd for pm in ("apt-get", "yum", "dnf", "apk")):
            install_called = True
        return make_result(return_code=0, stdout="/usr/bin/tmux")

    rt = make_runtime(exec_side_effect=exec_fn)
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    if not install_called and tk is not None:
        ok("tmux_already_present_no_install")
    else:
        fail("tmux_already_present_no_install", f"install_called={install_called}")


# ---------------------------------------------------------------------------
# 2. shell_view called with wrong keyword argument 'command'
# ---------------------------------------------------------------------------

async def test_shell_view_wrong_kwarg():
    """
    shell_view(command="...") must raise TypeError.
    Matches: TerminalToolkit.shell_view() got an unexpected keyword argument 'command'
    """
    rt = make_runtime()
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    try:
        await tk.shell_view(command="echo hi")  # type: ignore[call-arg]
        fail("shell_view_wrong_kwarg", "expected TypeError")
    except TypeError as e:
        if "command" in str(e):
            ok("shell_view_wrong_kwarg")
        else:
            fail("shell_view_wrong_kwarg", f"unexpected: {e}")


async def test_shell_view_correct_signature():
    """shell_view(id=...) with unknown session must return error string, not raise."""
    rt = make_runtime()
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    out = await tk.shell_view(id="no_such_session")
    if "Error: No session" in out:
        ok("shell_view_correct_signature")
    else:
        fail("shell_view_correct_signature", repr(out))


# ---------------------------------------------------------------------------
# 3. shell_wait silent/empty exception
# ---------------------------------------------------------------------------

async def test_shell_wait_exception_propagates():
    """
    If runtime.exec raises Exception("") inside shell_wait, it propagates
    unhandled — the empty-message error seen in logs.
    Matches: Error executing async tool 'shell_wait':  (empty message)
    """
    async def exec_fn(cmd, **_kw):
        if "tail" in cmd:
            raise Exception("")  # empty-message exception
        return make_result()

    rt = make_runtime(exec_side_effect=exec_fn)
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    s = sid("dead")
    tk.shell_sessions[s] = {
        "tmux_name": f"terminal_{s}",
        "log_path": f"/workdir/session_{s}.log",
        "last_offset": 0,
    }

    try:
        await tk.shell_wait(s, wait_seconds=1.0)
        fail("shell_wait_empty_exception", "expected exception to propagate")
    except Exception as e:
        if str(e) == "":
            ok("shell_wait_empty_exception_propagates_empty_msg")
        else:
            ok("shell_wait_exception_propagates")


async def test_shell_wait_unknown_session_returns_error():
    """shell_wait with unknown id must return error string, not raise."""
    rt = make_runtime()
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    out = await tk.shell_wait("nonexistent", wait_seconds=1)
    if "Error: No session" in out:
        ok("shell_wait_unknown_session")
    else:
        fail("shell_wait_unknown_session", repr(out))


# ---------------------------------------------------------------------------
# 4. shell_exec blocking — timeout-to-background fallback
# ---------------------------------------------------------------------------

async def test_shell_exec_block_completes_returns_output():
    """
    block=True: command finishes before timeout → full output returned,
    session cleaned up (not in shell_sessions).
    """
    async def exec_fn(cmd, **_kw):
        if "display-message" in cmd:
            return make_result(stdout="bash")   # pane_current_command = shell → done
        if "tail" in cmd:
            return make_result(stdout="hello world\n")
        return make_result()

    rt = make_runtime(exec_side_effect=exec_fn)
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)
    s = sid("blk")
    out = await tk.shell_exec(s, "echo hello world", block=True)

    if s not in tk.shell_sessions and "hello world" in out:
        ok("shell_exec_block_completes_returns_output")
    else:
        fail("shell_exec_block_completes_returns_output",
             f"in_sessions={s in tk.shell_sessions}, out={repr(out[:80])}")


async def test_shell_exec_block_timeout_session_survives():
    """
    block=True: command exceeds self.timeout → session left alive in
    shell_sessions, return message contains session ID and instructions.
    Uses timeout=0.6s so one poll cycle (0.5s) fires before deadline.
    """
    async def exec_fn(cmd, **_kw):
        if "display-message" in cmd:
            return make_result(stdout="sleep")  # always running
        return make_result()

    rt = make_runtime(exec_side_effect=exec_fn)
    # tiny timeout so the test completes quickly
    tk = await TerminalToolkit(timeout=0.6, working_directory="/workdir",
                               session_logs_dir=LOG_DIR, runtime=rt)
    s = sid("tmo")
    out = await tk.shell_exec(s, "sleep 300", block=True)

    if s in tk.shell_sessions and "still running" in out and s in out:
        ok("shell_exec_block_timeout_session_survives")
    else:
        fail("shell_exec_block_timeout_session_survives",
             f"in_sessions={s in tk.shell_sessions}, out={repr(out[:120])}")


# ---------------------------------------------------------------------------
# 5. shell_wait when tmux session has vanished
# ---------------------------------------------------------------------------

async def test_shell_wait_tmux_session_gone():
    """
    shell_wait on a session whose pipe-pane log is gone must not crash.
    (tail returns rc=1, no exception — just empty output.)
    """
    async def exec_fn(cmd, **_kw):
        if "tail" in cmd:
            return make_result(stdout="", return_code=1)
        return make_result()

    rt = make_runtime(exec_side_effect=exec_fn)
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    s = sid("gone")
    tk.shell_sessions[s] = {
        "tmux_name": f"terminal_{s}",
        "log_path": f"/workdir/session_{s}.log",
        "last_offset": 0,
    }

    try:
        await tk.shell_wait(s, wait_seconds=1.0)
        ok("shell_wait_tmux_gone_no_crash")
    except Exception as e:
        fail("shell_wait_tmux_gone_no_crash", f"raised: {e}")


# ---------------------------------------------------------------------------
# 6. shell_exec non-blocking — tmux session start fails
# ---------------------------------------------------------------------------

async def test_shell_exec_nonblocking_tmux_start_failure():
    """
    If tmux new-session fails after init, shell_exec(block=False) must raise.
    """
    session_start_called = False

    async def exec_fn(cmd, **_kw):
        nonlocal session_start_called
        if "script -qc" in cmd:
            session_start_called = True
            return make_result(return_code=1, stderr="session creation failed")
        return make_result()

    rt = make_runtime(exec_side_effect=exec_fn)
    tk = await TerminalToolkit(working_directory="/workdir", session_logs_dir=LOG_DIR, runtime=rt)

    try:
        await tk.shell_exec(sid(), "echo hi", block=False)
        fail("nonblocking_tmux_start_failure", "expected RuntimeError")
    except RuntimeError as e:
        if session_start_called and "Failed to start tmux session" in str(e):
            ok("nonblocking_tmux_start_failure")
        else:
            fail("nonblocking_tmux_start_failure",
                 f"session_start_called={session_start_called}, err={e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    tests = [
        # tmux init — fail-fast
        ("tmux unavailable fails init",                      test_tmux_not_available_fails_init),
        ("all 4 pkg managers tried before giving up",        test_tmux_all_pkg_managers_tried),
        ("apt-get success stops install chain",              test_tmux_install_succeeds_via_apt),
        ("apt/yum/dnf fail → apk fallback succeeds",        test_tmux_install_apt_fails_falls_back_to_apk),
        ("install rc=0 but which fails → still raises",     test_tmux_install_ok_but_which_fails),
        ("tmux already present → no install attempted",     test_tmux_already_present_no_install),
        # shell_view wrong kwarg
        ("shell_view wrong kwarg raises TypeError",         test_shell_view_wrong_kwarg),
        ("shell_view correct signature works",              test_shell_view_correct_signature),
        # shell_wait silent error
        ("shell_wait empty exception propagates",           test_shell_wait_exception_propagates),
        ("shell_wait unknown session returns error string", test_shell_wait_unknown_session_returns_error),
        # blocking timeout-to-background
        ("shell_exec block completes returns output",        test_shell_exec_block_completes_returns_output),
        ("shell_exec block timeout session survives",        test_shell_exec_block_timeout_session_survives),
        # tmux session gone
        ("shell_wait tmux gone no crash",                   test_shell_wait_tmux_session_gone),
        # nonblocking start failure
        ("shell_exec nonblocking tmux start failure",       test_shell_exec_nonblocking_tmux_start_failure),
    ]

    print("\n=== Shell error case tests (unit, no container needed) ===\n")
    for desc, fn in tests:
        print(f"  [{desc}]")
        try:
            await fn()
        except Exception as e:
            fail(desc, f"uncaught: {e}")

    print(f"\n{'=' * 50}")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
