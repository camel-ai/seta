"""Debug DockerHarborRuntime: start container, print actual docker names, get tools, exec.

Usage:
    python scripts/evaluation/test_docker_runtime.py
"""
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime

TASK       = "break-filter-js-from-html"
TASK_DIR   = f"/root/terminal_agent/dataset/terminal-bench-2.0/{TASK}"
TRIAL_ROOT = "/tmp/test_docker_runtime"
SESSION_ID = f"{TASK}_test"


async def main():
    runtime = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=SESSION_ID,
        environment_type="docker",
    )

    print(f"\n[1] reset()  session_id={SESSION_ID}")
    await runtime.reset()

    running = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    ).stdout.strip()
    print(f"[docker ps] containers:\n{running}\n")

    print(f"[2] get_tools()  docker_container_name={SESSION_ID}")
    try:
        await runtime.get_tools()
        print(f"[3] tools ok: {[t.func.__name__ for t in runtime.tools]}")
        result = runtime.terminal_toolkit.shell_exec("echo hello", "s0")
        print(f"[4] shell_exec: {result}")
    except Exception as e:
        print(f"[ERROR] {e}")

    print("\n[5] stop()")
    await runtime.stop(delete=True)


asyncio.run(main())
