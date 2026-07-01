"""P0 leak-cure: run TerminalEnvironment.step() in a KILLABLE subprocess.

Why: Python threads cannot be cancelled. A step that hangs in an uninterruptible
Daytona/toolkit call leaves a zombie thread even after STEP_TIMEOUT releases its
semaphore slot (env_service.py documents this). Over a long run these accumulate
(observed: 1800+ threads / 44 GB) until uvicorn's accept loop starves -> hang.

Fix: run each step in a forkserver child. On timeout we ``kill()`` the child and
ALL its sub-threads/clients die with it -> zero zombies in env_service. The only
thread env_service uses is a ``p.join(timeout)`` waiter, which ALWAYS unwinds
after ``timeout`` (it just waits, it isn't blocked on the hung work).

forkserver preloads the heavy imports (camel/harbor via terminal_env) ONCE at
forkserver start, so per-step fork is cheap. The (run_info, reward) result comes
back via a pickle FILE (not a pipe) to avoid pipe-buffer deadlock on large
run_info. Gated by STEP_USE_SUBPROCESS in env_service.py — default OFF.
"""
import asyncio
import multiprocessing as mp
import os
import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor

# ── DEDICATED p.join executor (CRITICAL) ─────────────────────────────────────
# The p.join waiter below MUST NOT use the asyncio DEFAULT executor. The default
# ThreadPoolExecutor is capped at min(32, cpu+4)=32 threads, and harbor's Daytona
# create-semaphore acquires via ``asyncio.to_thread`` (== default executor). If
# the p.joins (up to MAX_SLOTS of them, each BLOCKING for up to STEP_TIMEOUT=3000s)
# ran on the default executor they would pin all 32 threads, and EVERY build's
# ``to_thread(create_sem.acquire)`` would queue behind them forever — the env_service
# would report building_active>0 while ZERO sandboxes build on Daytona (observed).
# A dedicated pool sized for the full slot budget keeps p.join off the default
# executor (freeing it for builds) AND uncaps P0 trajectory concurrency (was 32).
_JOIN_WORKERS = int(os.environ.get("JOIN_EXECUTOR_WORKERS", "0")) or (
    int(os.environ.get("MAX_SLOTS", "160")) + 64
)
_JOIN_EXECUTOR = ThreadPoolExecutor(
    max_workers=_JOIN_WORKERS, thread_name_prefix="p0-join"
)

# forkserver: child is forked from a clean single-threaded forkserver process
# (NOT from the multi-threaded uvicorn), avoiding inherited-lock hazards.
_ctx = mp.get_context("forkserver")
try:
    # Import the heavy deps ONCE in the forkserver, so per-step forks are fast.
    _ctx.set_forkserver_preload(["seta_env.environments.terminal_env"])
except Exception:
    pass


def _worker(result_path: str, te, task: dict, uid: str, traj_i: int) -> None:
    """Runs in the child: execute the step, pickle the result to result_path."""
    try:
        res = asyncio.run(te.step(task, uid=uid, traj_i=traj_i))
        payload = ("ok", res)
    except BaseException as e:  # noqa: BLE001 - report anything, incl. SystemExit
        payload = ("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    try:
        with open(result_path, "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        try:
            with open(result_path, "wb") as f:
                pickle.dump(("err", "step result not picklable"), f)
        except Exception:
            pass


async def run_step_killable(te, task: dict, uid: str, traj_i: int,
                            timeout: float, result_path: str):
    """Run te.step() in a killable subprocess; return (run_info, reward).

    Raises asyncio.TimeoutError (after killing the child) on timeout, matching
    the existing wait_for behavior so the /step handler's except-blocks are unchanged.
    """
    try:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        if os.path.exists(result_path):
            os.remove(result_path)
    except Exception:
        pass

    p = _ctx.Process(target=_worker, args=(result_path, te, task, uid, traj_i),
                     name=f"env-step-proc-{uid}", daemon=False)
    p.start()
    loop = asyncio.get_event_loop()
    # Wait for the child, bounded by `timeout`, in a thread that ALWAYS unwinds.
    # MUST use _JOIN_EXECUTOR (not None/default) — see module docstring above.
    await loop.run_in_executor(_JOIN_EXECUTOR, p.join, timeout)
    if p.is_alive():
        p.kill()                                  # SIGKILL the hung step (+ its threads)
        await loop.run_in_executor(_JOIN_EXECUTOR, p.join, 15)
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
        raise asyncio.TimeoutError(
            f"step exceeded {timeout:.0f}s (subprocess killed; no thread leaked)")

    try:
        with open(result_path, "rb") as f:
            status, payload = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"step subprocess exited (code={p.exitcode}) without writing a result")
    finally:
        try:
            os.remove(result_path)
        except Exception:
            pass

    if status == "ok":
        return payload
    raise RuntimeError(f"step subprocess error: {payload}")
