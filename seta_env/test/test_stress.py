"""Stress & Concurrency Tests for Slot Pool Service

Tests the node manager, RemoteDockerEnvironment, and terminal toolkit under
concurrent load on real remote nodes. No agent loop — purely runtime/toolkit level.

Run:
    cd /root/terminal_agent
    NODE_MANAGER_API_KEY=harbor-node-dev-key \
      /root/miniforge3/envs/terminal_agent/bin/python seta_env/test/test_stress.py [--group N]

Prerequisites:
    - Both nodes running node_manager (deploy via start.sh)
    - Scheduler running at localhost:8000
    - Dataset tbench-tasks_migrated active on all nodes
"""

import argparse
import asyncio
import io
import os
import sys
import tarfile
import time
import uuid
from pathlib import Path

import httpx

# ── Config ───────────────────────────────────────────────────────────────────

NODE_URLS = [
    "http://95.133.253.167:8001",
    "http://95.133.253.138:8001",
]
API_KEY = os.environ["NODE_MANAGER_API_KEY"]
SCHEDULER_URL = "http://localhost:8000"
TASK_NAME = "hello-world"
DATASET_NAME = "tbench-tasks_migrated"

# ── Helpers ──────────────────────────────────────────────────────────────────

def sid(prefix: str = "stress") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def make_client(node_url: str, timeout: float = 300) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=node_url,
        headers={"X-API-Key": API_KEY},
        timeout=timeout,
    )


async def build_and_poll(client: httpx.AsyncClient, task_name: str, poll_timeout: float = 300) -> dict:
    """Fire a build and poll until done. Returns the final status dict."""
    r = await client.post("/build", json={"task_name": task_name})
    assert r.status_code == 200, f"build POST failed: {r.status_code} {r.text}"
    data = r.json()
    job_id = data["job_id"]

    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        await asyncio.sleep(2.0)
        r = await client.get(f"/build/{job_id}")
        assert r.status_code == 200, f"build status failed: {r.status_code}"
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
    raise TimeoutError(f"Build {job_id} did not complete within {poll_timeout}s")


async def ensure_built(node_url: str, task_name: str = TASK_NAME) -> None:
    """Build task on node if not already cached."""
    async with make_client(node_url, timeout=30) as c:
        data = await build_and_poll(c, task_name, poll_timeout=600)
        assert data.get("success") is True, f"Build failed on {node_url}: {data}"


async def compose_up(client: httpx.AsyncClient, session_id: str, task_name: str = TASK_NAME) -> dict:
    r = await client.post("/compose/up", json={
        "session_id": session_id,
        "task_name": task_name,
        "env_vars": {"MAIN_IMAGE_NAME": f"hb__{task_name}"},
    })
    assert r.status_code == 200, f"compose up failed: {r.status_code} {r.text}"
    data = r.json()
    assert data["success"] is True, f"compose up failed: {data}"
    return data


async def compose_down(client: httpx.AsyncClient, session_id: str, delete: bool = True) -> None:
    r = await client.post("/compose/down", json={"session_id": session_id, "delete": delete})
    # Don't assert — may already be cleaned up


async def exec_cmd(client: httpx.AsyncClient, session_id: str, command: str, **kwargs) -> dict:
    r = await client.post("/exec", json={
        "session_id": session_id,
        "command": command,
        **kwargs,
    })
    assert r.status_code == 200, f"exec failed: {r.status_code} {r.text}"
    return r.json()


# ── Cleanup ──────────────────────────────────────────────────────────────────

async def cleanup_all():
    """Clean orphan containers on all nodes via scheduler."""
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{SCHEDULER_URL}/cleanup")
            if r.status_code == 200:
                data = r.json()
                print(f"  Cleanup: stopped {sum(r.get('body', {}).get('stopped_containers', 0) for r in data.get('results', []))} containers")
    except Exception as e:
        print(f"  Cleanup warning: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1: Concurrent Builds
# ══════════════════════════════════════════════════════════════════════════════

async def test_1_1_concurrent_build_both_nodes():
    """Build same task on both nodes concurrently."""
    results = await asyncio.gather(*[
        ensure_built(url, TASK_NAME) for url in NODE_URLS
    ])
    # Build again — should use cache (fast)
    t0 = time.time()
    await asyncio.gather(*[ensure_built(url, TASK_NAME) for url in NODE_URLS])
    elapsed = time.time() - t0
    print(f"PASS test_1_1_concurrent_build_both_nodes (cache rebuild: {elapsed:.1f}s)")


async def test_1_2_concurrent_build_4_tasks():
    """Build 4 different tasks concurrently on node 1."""
    tasks = ["hello-world", "broken-python", "solve-sudoku", "gcode-to-text"]
    await asyncio.gather(*[ensure_built(NODE_URLS[0], t) for t in tasks])
    print(f"PASS test_1_2_concurrent_build_4_tasks")


async def test_1_3_build_poll_resilience():
    """Build via polling — verify the poll loop works end-to-end."""
    async with make_client(NODE_URLS[0], timeout=30) as c:
        data = await build_and_poll(c, TASK_NAME, poll_timeout=600)
        assert data["status"] == "done"
        assert data.get("success") is True
    print("PASS test_1_3_build_poll_resilience")


async def test_1_4_build_status_not_found():
    """GET /build/nonexistent → 404."""
    async with make_client(NODE_URLS[0], timeout=10) as c:
        r = await c.get("/build/nonexistent-job-id")
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("PASS test_1_4_build_status_not_found")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2: Concurrent Container Lifecycle
# ══════════════════════════════════════════════════════════════════════════════

async def test_2_1_concurrent_compose_up_16():
    """Start 16 containers (8 per node), exec on all, then tear down."""
    sessions_per_node = []
    for node_url in NODE_URLS:
        sessions = [sid(f"g2n{NODE_URLS.index(node_url)}") for _ in range(8)]
        sessions_per_node.append((node_url, sessions))

    # Compose up all 16 concurrently
    async def up_one(node_url, session_id):
        async with make_client(node_url) as c:
            await compose_up(c, session_id)

    await asyncio.gather(*[
        up_one(node_url, s)
        for node_url, sessions in sessions_per_node
        for s in sessions
    ])

    # Exec on all 16 concurrently
    async def exec_one(node_url, session_id):
        async with make_client(node_url) as c:
            data = await exec_cmd(c, session_id, f"echo alive-{session_id}")
            assert f"alive-{session_id}" in data["stdout"], f"Bad stdout: {data['stdout']}"

    await asyncio.gather(*[
        exec_one(node_url, s)
        for node_url, sessions in sessions_per_node
        for s in sessions
    ])

    # Tear down all 16
    async def down_one(node_url, session_id):
        async with make_client(node_url) as c:
            await compose_down(c, session_id)

    await asyncio.gather(*[
        down_one(node_url, s)
        for node_url, sessions in sessions_per_node
        for s in sessions
    ])

    print("PASS test_2_1_concurrent_compose_up_16")


async def test_2_2_rapid_create_destroy():
    """Rapid sequential create-exec-destroy cycle — 8 iterations."""
    node_url = NODE_URLS[0]
    for i in range(8):
        session_id = sid(f"rapid{i}")
        async with make_client(node_url) as c:
            await compose_up(c, session_id)
            data = await exec_cmd(c, session_id, f"echo iter-{i}")
            assert f"iter-{i}" in data["stdout"]
            await compose_down(c, session_id)
    print("PASS test_2_2_rapid_create_destroy (8 cycles)")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3: Concurrent Exec (heavy load)
# ══════════════════════════════════════════════════════════════════════════════

async def test_3_1_16_concurrent_exec():
    """16 concurrent exec calls on same container."""
    session_id = sid("exec16")
    node_url = NODE_URLS[0]
    async with make_client(node_url) as c:
        await compose_up(c, session_id)
        try:
            async def run_one(i):
                data = await exec_cmd(c, session_id, f"echo output-{i}")
                assert f"output-{i}" in data["stdout"], f"Exec {i} bad stdout: {data['stdout']}"
                return i

            results = await asyncio.gather(*[run_one(i) for i in range(16)])
            assert len(results) == 16
        finally:
            await compose_down(c, session_id)
    print("PASS test_3_1_16_concurrent_exec")


async def test_3_2_exec_idempotency():
    """Same request_id returns cached result, not re-executed command."""
    session_id = sid("idemp")
    node_url = NODE_URLS[0]
    request_id = str(uuid.uuid4())

    async with make_client(node_url) as c:
        await compose_up(c, session_id)
        try:
            # First call
            r1 = await c.post("/exec", json={
                "session_id": session_id,
                "command": "echo hello-first",
                "request_id": request_id,
            })
            data1 = r1.json()
            assert "hello-first" in data1["stdout"], f"First call bad: {data1}"

            # Second call with same request_id but different command
            r2 = await c.post("/exec", json={
                "session_id": session_id,
                "command": "echo different-command",
                "request_id": request_id,
            })
            data2 = r2.json()
            # Should return cached result from first call
            assert "hello-first" in data2["stdout"], \
                f"Expected cached 'hello-first', got: {data2['stdout']}"
            assert "different-command" not in data2["stdout"], \
                f"Command was re-executed! Got: {data2['stdout']}"
        finally:
            await compose_down(c, session_id)
    print("PASS test_3_2_exec_idempotency")


async def test_3_3_rapid_exec_burst():
    """50 sequential execs as fast as possible. Measures latency."""
    session_id = sid("burst")
    node_url = NODE_URLS[0]

    async with make_client(node_url) as c:
        await compose_up(c, session_id)
        try:
            latencies = []
            for i in range(50):
                t0 = time.time()
                data = await exec_cmd(c, session_id, f"echo burst-{i}")
                latencies.append(time.time() - t0)
                assert f"burst-{i}" in data["stdout"]

            avg = sum(latencies) / len(latencies)
            p99 = sorted(latencies)[int(0.99 * len(latencies))]
            total = sum(latencies)
            print(f"PASS test_3_3_rapid_exec_burst (50 execs: avg={avg:.3f}s p99={p99:.3f}s total={total:.1f}s)")
        finally:
            await compose_down(c, session_id)


async def test_3_4_exec_timeout():
    """Exec with timeout_sec — command should be killed."""
    session_id = sid("timeout")
    node_url = NODE_URLS[0]

    async with make_client(node_url) as c:
        await compose_up(c, session_id)
        try:
            data = await exec_cmd(c, session_id, "sleep 30", timeout_sec=3)
            # Should have been killed by timeout
            assert data["return_code"] != 0 or "timed out" in data.get("stderr", "").lower(), \
                f"Expected timeout error, got rc={data['return_code']}"
        finally:
            await compose_down(c, session_id)
    print("PASS test_3_4_exec_timeout")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4: Concurrent Toolkit Operations (tmux via DockerHarborRuntime)
# ══════════════════════════════════════════════════════════════════════════════

async def _make_runtime_with_toolkit(node_url: str, session_id: str):
    """Create a DockerHarborRuntime with tmux toolkit against a remote container."""
    from harbor.environments.docker.remote_docker_environment import RemoteDockerEnvironment
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.task.task import Task
    from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime

    task_dir = Path(__file__).resolve().parent.parent.parent / "dataset" / DATASET_NAME / TASK_NAME
    task = Task(task_dir)
    trial_root = Path(__file__).resolve().parent.parent / "test" / "output" / "stress_trials"
    trial_paths = TrialPaths(trial_dir=trial_root / session_id)
    trial_paths.mkdir()

    env = RemoteDockerEnvironment(
        node_manager_url=node_url,
        api_key=API_KEY,
        environment_name=TASK_NAME,
        session_id=session_id,
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
    )
    runtime = DockerHarborRuntime(environment=env)
    return runtime


async def test_4_1_concurrent_shell_exec_4_containers():
    """4 containers on node 1, concurrent shell_exec via tmux toolkit."""
    runtimes = []
    try:
        # Start 4 containers
        for i in range(4):
            session_id = sid(f"tk4c{i}")
            rt = await _make_runtime_with_toolkit(NODE_URLS[0], session_id)
            await rt.reset()
            await rt.get_tools(toolkit="tmux")
            runtimes.append(rt)

        # Concurrent shell_exec
        async def exec_one(rt, i):
            out = await rt.terminal_toolkit.shell_exec(id=f"t{i}", command=f"echo toolkit-{i}", block=True)
            assert f"toolkit-{i}" in out, f"Bad output for {i}: {out}"

        await asyncio.gather(*[exec_one(rt, i) for i, rt in enumerate(runtimes)])
        print("PASS test_4_1_concurrent_shell_exec_4_containers")
    finally:
        for rt in runtimes:
            try:
                await rt.stop(delete=True)
            except Exception:
                pass


async def test_4_2_concurrent_nonblocking_sessions():
    """4 non-blocking sessions in same container — no cross-contamination."""
    session_id = sid("tknb")
    rt = await _make_runtime_with_toolkit(NODE_URLS[0], session_id)
    try:
        await rt.reset()
        await rt.get_tools(toolkit="tmux")
        tk = rt.terminal_toolkit

        # Start 4 non-blocking sessions
        for i in range(4):
            out = await tk.shell_exec(id=f"bg_{i}", command=f"sleep 3 && echo done_{i}", block=False)
            assert f"Session 'bg_{i}' started." in out, f"Bad start msg: {out}"

        # Wait for completion
        await asyncio.sleep(5)

        # Check all 4
        for i in range(4):
            out = await tk.shell_view(id=f"bg_{i}")
            assert f"done_{i}" in out, f"Session bg_{i} missing output: {out}"
            assert "[completed]" in out, f"Session bg_{i} not completed: {out}"

        print("PASS test_4_2_concurrent_nonblocking_sessions")
    finally:
        await rt.stop(delete=True)


async def test_4_3_blocking_timeout_conversion():
    """Blocking exec timeout converts to non-blocking session."""
    session_id = sid("tktimeout")
    rt = await _make_runtime_with_toolkit(NODE_URLS[0], session_id)
    try:
        await rt.reset()
        await rt.get_tools(toolkit="tmux")
        tk = rt.terminal_toolkit

        # Fast command should return immediately
        fast_out = await tk.shell_exec(id="fast", command="echo quick", block=True)
        assert "quick" in fast_out, f"Fast command bad: {fast_out}"

        # Slow command with short timeout
        original_timeout = tk.timeout
        tk.timeout = 5.0
        try:
            slow_out = await tk.shell_exec(id="slow", command="sleep 60", block=True)
            assert "did not complete within" in slow_out, f"No timeout msg: {slow_out}"
        finally:
            tk.timeout = original_timeout

        # Session should still be accessible
        view_out = await tk.shell_view(id="slow")
        assert isinstance(view_out, str)

        # Cleanup
        kill_out = await tk.shell_kill_process(id="slow")
        assert "terminated" in kill_out.lower(), f"Kill failed: {kill_out}"

        print("PASS test_4_3_blocking_timeout_conversion")
    finally:
        await rt.stop(delete=True)


async def test_4_4_rapid_repl_interaction():
    """Send 20 commands to Python REPL rapidly."""
    session_id = sid("tkrepl")
    rt = await _make_runtime_with_toolkit(NODE_URLS[0], session_id)
    try:
        await rt.reset()
        await rt.get_tools(toolkit="tmux")
        tk = rt.terminal_toolkit

        await tk.shell_exec(id="py", command="python3", block=False)
        await asyncio.sleep(1.5)

        for i in range(20):
            out = await tk.shell_write_to_process(id="py", command=f"print({i}*{i})")
            expected = str(i * i)
            assert expected in out, f"Step {i}: expected {expected} in output, got: {out}"

        await tk.shell_write_to_process(id="py", command="exit()")
        print("PASS test_4_4_rapid_repl_interaction (20 REPL commands)")
    finally:
        await rt.stop(delete=True)


async def test_4_5_concurrent_file_writes():
    """8 concurrent file writes to different paths."""
    session_id = sid("tkfiles")
    rt = await _make_runtime_with_toolkit(NODE_URLS[0], session_id)
    try:
        await rt.reset()
        await rt.get_tools(toolkit="tmux")
        tk = rt.terminal_toolkit

        # Write 8 files concurrently
        async def write_one(i):
            return await tk.shell_write_content_to_file(
                content=f"content_{i}_data",
                file_path=f"/tmp/stress_file_{i}.txt",
            )

        results = await asyncio.gather(*[write_one(i) for i in range(8)])
        for r in results:
            assert "Content written" in r or "written" in r.lower(), f"Write failed: {r}"

        # Verify all 8
        for i in range(8):
            out = await tk.shell_exec(id=f"verify_{i}", command=f"cat /tmp/stress_file_{i}.txt", block=True)
            assert f"content_{i}_data" in out, f"File {i} content mismatch: {out}"

        print("PASS test_4_5_concurrent_file_writes (8 files)")
    finally:
        await rt.stop(delete=True)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5: Full Stack (Scheduler + Runtime + Toolkit)
# ══════════════════════════════════════════════════════════════════════════════

async def test_5_1_scheduler_runtime_lifecycle():
    """Allocate 8 slots → build → start → toolkit exec → stop → release."""
    task_id = sid("fullstack")
    async with httpx.AsyncClient(base_url=SCHEDULER_URL, timeout=30) as sc:
        # Allocate
        r = await sc.post("/allocate_group", json={"task_id": task_id, "n_slots": 8})
        assert r.status_code == 200, f"Allocate failed: {r.text}"
        assignments = r.json()["assignments"]
        assert len(assignments) == 8

        runtimes = []
        try:
            # Build on unique nodes
            unique_nodes = list({a["node_url"] for a in assignments})
            await asyncio.gather(*[ensure_built(url) for url in unique_nodes])

            # Start all containers + init toolkit
            for a in assignments:
                session_id = sid("fs")
                rt = await _make_runtime_with_toolkit(a["node_url"], session_id)
                await rt.reset()
                await rt.get_tools(toolkit="tmux")
                runtimes.append(rt)

            # Concurrent exec on all
            async def exec_one(rt, i):
                out = await rt.terminal_toolkit.shell_exec(id=f"fs{i}", command=f"echo slot-{i}", block=True)
                assert f"slot-{i}" in out

            await asyncio.gather(*[exec_one(rt, i) for i, rt in enumerate(runtimes)])

        finally:
            # Stop all
            for rt in runtimes:
                try:
                    await rt.stop(delete=True)
                except Exception:
                    pass
            # Release slots
            await sc.post("/release_group", json={"task_id": task_id})

    # Verify all slots freed
    async with httpx.AsyncClient(base_url=SCHEDULER_URL, timeout=10) as sc:
        r = await sc.get("/status")
        status = r.json()
        total_used = sum(n["total_slots"] - n["free_slots"] for n in status["nodes"])
        assert total_used == 0, f"Slots not freed: {total_used} still used"

    print("PASS test_5_1_scheduler_runtime_lifecycle (8 slots)")


async def test_5_2_max_concurrency_16():
    """16 slots across both nodes — full concurrent exec."""
    task_id = sid("max16")
    async with httpx.AsyncClient(base_url=SCHEDULER_URL, timeout=30) as sc:
        r = await sc.post("/allocate_group", json={"task_id": task_id, "n_slots": 16})
        assert r.status_code == 200, f"Allocate failed: {r.text}"
        assignments = r.json()["assignments"]
        assert len(assignments) == 16

        runtimes = []
        try:
            unique_nodes = list({a["node_url"] for a in assignments})
            await asyncio.gather(*[ensure_built(url) for url in unique_nodes])

            # Start all 16 concurrently
            async def start_one(a):
                session_id = sid("mx")
                rt = await _make_runtime_with_toolkit(a["node_url"], session_id)
                await rt.reset()
                await rt.get_tools(toolkit="tmux")
                return rt

            runtimes = await asyncio.gather(*[start_one(a) for a in assignments])

            # Exec on all 16 concurrently
            async def exec_one(rt, i):
                out = await rt.terminal_toolkit.shell_exec(id=f"mx{i}", command="echo ok", block=True)
                assert "ok" in out

            await asyncio.gather(*[exec_one(rt, i) for i, rt in enumerate(runtimes)])

        finally:
            await asyncio.gather(*[
                rt.stop(delete=True) for rt in runtimes
            ], return_exceptions=True)
            await sc.post("/release_group", json={"task_id": task_id})

    print("PASS test_5_2_max_concurrency_16")


async def test_5_3_allocate_release_cycling():
    """Rapid allocate-release cycling — 10 iterations."""
    async with httpx.AsyncClient(base_url=SCHEDULER_URL, timeout=30) as sc:
        for i in range(10):
            task_id = f"cycle-{i}-{uuid.uuid4().hex[:6]}"
            r = await sc.post("/allocate_group", json={"task_id": task_id, "n_slots": 4})
            assert r.status_code == 200, f"Allocate failed at iter {i}: {r.text}"
            r = await sc.post("/release_group", json={"task_id": task_id})
            assert r.status_code == 200

        # Verify all free
        r = await sc.get("/status")
        status = r.json()
        for n in status["nodes"]:
            assert n["free_slots"] == n["total_slots"], \
                f"Node {n['url']}: {n['free_slots']}/{n['total_slots']} free"

    print("PASS test_5_3_allocate_release_cycling (10 cycles)")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6: Error Recovery
# ══════════════════════════════════════════════════════════════════════════════

async def test_6_1_exec_unknown_session():
    """Exec on nonexistent session → 404."""
    async with make_client(NODE_URLS[0], timeout=10) as c:
        r = await c.post("/exec", json={"session_id": "nonexistent", "command": "echo hi"})
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("PASS test_6_1_exec_unknown_session")


async def test_6_2_compose_down_unknown():
    """Compose down on nonexistent session → 404."""
    async with make_client(NODE_URLS[0], timeout=10) as c:
        r = await c.post("/compose/down", json={"session_id": "nonexistent"})
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("PASS test_6_2_compose_down_unknown")


async def test_6_3_double_compose_down():
    """Start → down → down again. First succeeds, second 404."""
    session_id = sid("dbldown")
    async with make_client(NODE_URLS[0]) as c:
        await compose_up(c, session_id)
        r = await c.post("/compose/down", json={"session_id": session_id, "delete": True})
        assert r.json().get("success") is True

        r = await c.post("/compose/down", json={"session_id": session_id})
        assert r.status_code == 404, f"Expected 404 on second down, got {r.status_code}"
    print("PASS test_6_3_double_compose_down")


async def test_6_4_build_nonexistent_task():
    """Build nonexistent task → 400."""
    async with make_client(NODE_URLS[0], timeout=10) as c:
        r = await c.post("/build", json={"task_name": "nonexistent_task_xyz_999"})
        assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"
    print("PASS test_6_4_build_nonexistent_task")


async def test_6_5_exec_after_down():
    """Exec after compose down → 404."""
    session_id = sid("afterdown")
    async with make_client(NODE_URLS[0]) as c:
        await compose_up(c, session_id)
        data = await exec_cmd(c, session_id, "echo alive")
        assert "alive" in data["stdout"]

        await compose_down(c, session_id)
        r = await c.post("/exec", json={"session_id": session_id, "command": "echo gone"})
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("PASS test_6_5_exec_after_down")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

GROUPS = {
    1: ("Concurrent Builds", [
        test_1_1_concurrent_build_both_nodes,
        test_1_2_concurrent_build_4_tasks,
        test_1_3_build_poll_resilience,
        test_1_4_build_status_not_found,
    ]),
    2: ("Concurrent Container Lifecycle", [
        test_2_1_concurrent_compose_up_16,
        test_2_2_rapid_create_destroy,
    ]),
    3: ("Concurrent Exec", [
        test_3_1_16_concurrent_exec,
        test_3_2_exec_idempotency,
        test_3_3_rapid_exec_burst,
        test_3_4_exec_timeout,
    ]),
    4: ("Concurrent Toolkit (tmux)", [
        test_4_1_concurrent_shell_exec_4_containers,
        test_4_2_concurrent_nonblocking_sessions,
        test_4_3_blocking_timeout_conversion,
        test_4_4_rapid_repl_interaction,
        test_4_5_concurrent_file_writes,
    ]),
    5: ("Full Stack Stress", [
        test_5_1_scheduler_runtime_lifecycle,
        test_5_2_max_concurrency_16,
        test_5_3_allocate_release_cycling,
    ]),
    6: ("Error Recovery", [
        test_6_1_exec_unknown_session,
        test_6_2_compose_down_unknown,
        test_6_3_double_compose_down,
        test_6_4_build_nonexistent_task,
        test_6_5_exec_after_down,
    ]),
}


async def run_group(group_num: int) -> bool:
    name, tests = GROUPS[group_num]
    print(f"\n{'='*60}")
    print(f"Group {group_num}: {name} ({len(tests)} tests)")
    print(f"{'='*60}")
    failed = 0
    for test in tests:
        try:
            await test()
        except Exception as e:
            print(f"FAIL {test.__name__}: {e}")
            failed += 1
    return failed == 0


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, default=None, help="Run specific group (1-6)")
    args = parser.parse_args()

    print(f"Stress tests: nodes={NODE_URLS}")
    print(f"Scheduler: {SCHEDULER_URL}")

    # Pre-flight: check health
    for url in NODE_URLS:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{url}/health")
            assert r.status_code == 200, f"Node {url} unhealthy: {r.text}"
            print(f"  Node {url}: OK (dataset={r.json().get('active_dataset')})")

    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{SCHEDULER_URL}/health")
        assert r.status_code == 200, f"Scheduler unhealthy: {r.text}"
        print(f"  Scheduler: OK")

    # Cleanup before testing
    print("\nCleaning up orphan containers...")
    await cleanup_all()

    groups_to_run = [args.group] if args.group else sorted(GROUPS.keys())
    all_passed = True
    for g in groups_to_run:
        if not await run_group(g):
            all_passed = False

    # Final cleanup
    print("\nFinal cleanup...")
    await cleanup_all()

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL STRESS TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
