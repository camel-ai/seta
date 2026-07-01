"""Test: Node Manager HTTP API
Run: python seta_env/test/test_node_manager.py
Env: NODE_MANAGER_URL, NODE_MANAGER_API_KEY
"""
import asyncio
import io
import os
import sys
import tarfile
import uuid
from pathlib import Path

import httpx

NODE_URL = os.environ.get("NODE_MANAGER_URL", "http://95.133.253.67:8001")
API_KEY  = os.environ["NODE_MANAGER_API_KEY"]
_REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = _REPO_ROOT / "dataset/seta-env-harbor/0"


def make_client(timeout: float = 300) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=NODE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=timeout,
    )


def sid(prefix: str = "nm") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ── Basic tests ───────────────────────────────────────────────────────────────

async def test_health():
    async with httpx.AsyncClient(base_url=NODE_URL, timeout=10) as c:
        r = await c.get("/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert "node" in body
    print("PASS test_health")


async def test_auth_rejected():
    async with httpx.AsyncClient(base_url=NODE_URL, timeout=10) as c:
        r = await c.post(
            "/build",
            json={"task_name": "0"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.text}"
    print("PASS test_auth_rejected")


# ── Dataset setup ─────────────────────────────────────────────────────────────

async def test_setup():
    """Set active dataset (downloads if missing). May take a while on first run."""
    async with make_client(timeout=600) as c:
        r = await c.post("/setup", json={"dataset_name": "seta-env-harbor"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["success"] is True
        assert "already_present" in data
    print(f"PASS test_setup  (already_present={data['already_present']})")


async def test_health_shows_active_dataset():
    async with httpx.AsyncClient(base_url=NODE_URL, timeout=10) as c:
        r = await c.get("/health")
        body = r.json()
        assert body["active_dataset"] == "seta-env-harbor", \
            f"Expected seta-env-harbor, got {body['active_dataset']}"
        assert "seta-env-harbor" in body.get("datasets", []), \
            f"seta-env-harbor not in datasets: {body.get('datasets')}"
    print("PASS test_health_shows_active_dataset")


# ── Build ─────────────────────────────────────────────────────────────────────

async def test_build_happy_path():
    """Build task 0 — slow on first run, fast on cache hit."""
    async with make_client(timeout=600) as c:
        r = await c.post("/build", json={"task_name": "0"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["success"] is True, f"Build failed: {data.get('logs', '')}"
        assert data["image_name"] == "hb__0"
    print("PASS test_build_happy_path")


async def test_build_idempotent():
    """Building the same task twice should both succeed (second uses cache)."""
    async with make_client(timeout=600) as c:
        for i in range(2):
            r = await c.post("/build", json={"task_name": "0"})
            assert r.json()["success"] is True, f"Build {i} failed: {r.json()}"
    print("PASS test_build_idempotent")


# ── Compose lifecycle ─────────────────────────────────────────────────────────

async def test_compose_lifecycle():
    """up → exec → down."""
    session = sid("lifecycle")
    async with make_client(timeout=300) as c:
        # up
        r = await c.post("/compose/up", json={
            "session_id": session,
            "task_name": "0",
            "env_vars": {"MAIN_IMAGE_NAME": "hb__0"},
        })
        assert r.json()["success"] is True, f"compose up failed: {r.json()}"

        # exec
        r = await c.post("/exec", json={"session_id": session, "command": "echo hello"})
        data = r.json()
        assert data["return_code"] == 0, f"exec failed: {data}"
        assert "hello" in data["stdout"], f"unexpected stdout: {data['stdout']!r}"

        # exec with cwd
        r = await c.post("/exec", json={
            "session_id": session, "command": "pwd", "cwd": "/workdir",
        })
        assert "/workdir" in r.json()["stdout"]

        # exec with env
        r = await c.post("/exec", json={
            "session_id": session, "command": "echo $FOO", "env": {"FOO": "bar"},
        })
        assert "bar" in r.json()["stdout"]

        # exec failing command
        r = await c.post("/exec", json={"session_id": session, "command": "exit 42"})
        assert r.json()["return_code"] != 0

        # down with delete
        r = await c.post("/compose/down", json={"session_id": session, "delete": True})
        assert r.json()["success"] is True, f"compose down failed: {r.json()}"
    print("PASS test_compose_lifecycle")


# ── File transfer ─────────────────────────────────────────────────────────────

async def test_upload_and_download():
    """Upload tests dir → write reward.txt → download verifier dir → check content."""
    session = sid("fileio")
    async with make_client(timeout=300) as c:
        r = await c.post("/compose/up", json={
            "session_id": session,
            "task_name": "0",
            "env_vars": {"MAIN_IMAGE_NAME": "hb__0"},
        })
        assert r.json()["success"] is True

        try:
            # Upload the tests directory as a tarball
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(TASK_DIR / "tests", arcname=".")
            buf.seek(0)
            r = await c.post(
                "/upload",
                data={"session_id": session, "target_path": "/tests"},
                files={"content_tar": ("tests.tar.gz", buf.read(), "application/gzip")},
            )
            assert r.json()["success"] is True, f"upload failed: {r.json()}"

            # Verify upload landed
            r = await c.post("/exec", json={
                "session_id": session, "command": "ls /tests",
            })
            assert r.json()["return_code"] == 0

            # Write reward.txt and download verifier dir
            await c.post("/exec", json={
                "session_id": session,
                "command": "mkdir -p /logs/verifier && echo 0.75 > /logs/verifier/reward.txt",
            })

            r = await c.get(f"/download/{session}",
                            params={"source_path": "/logs/verifier"})
            assert r.status_code == 200, f"download failed: {r.status_code} {r.text}"

            # Untar and verify reward.txt
            buf2 = io.BytesIO(r.content)
            with tarfile.open(fileobj=buf2, mode="r:gz") as tar:
                names = tar.getnames()
                reward_member = next((n for n in names if "reward.txt" in n), None)
                assert reward_member is not None, f"reward.txt not found in: {names}"
                content = tar.extractfile(reward_member).read().decode()
            assert "0.75" in content, f"unexpected reward content: {content!r}"

        finally:
            await c.post("/compose/down", json={"session_id": session, "delete": True})

    print("PASS test_upload_and_download")


# ── Concurrency ───────────────────────────────────────────────────────────────

async def test_concurrent_builds():
    """4 concurrent builds of the same task — all should succeed (semaphore allows ≤16)."""
    async def build_one() -> bool:
        async with make_client(timeout=600) as c:
            r = await c.post("/build", json={"task_name": "0"})
            return r.json()["success"]

    results = await asyncio.gather(*[build_one() for _ in range(4)])
    assert all(results), f"Some builds failed: {results}"
    print("PASS test_concurrent_builds (4 concurrent)")


async def test_concurrent_compose_up():
    """4 concurrent compose-up sessions — each isolated by session_id."""
    sessions = [sid("concurrent") for _ in range(4)]

    async def up_exec_down(s: str) -> bool:
        async with make_client(timeout=300) as c:
            r = await c.post("/compose/up", json={
                "session_id": s, "task_name": "0",
                "env_vars": {"MAIN_IMAGE_NAME": "hb__0"},
            })
            if not r.json()["success"]:
                return False
            r = await c.post("/exec", json={"session_id": s, "command": "echo ok"})
            ok = "ok" in r.json()["stdout"]
            await c.post("/compose/down", json={"session_id": s, "delete": True})
            return ok

    results = await asyncio.gather(*[up_exec_down(s) for s in sessions])
    assert all(results), f"Some concurrent runs failed: {results}"
    print("PASS test_concurrent_compose_up (4 concurrent sessions)")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print(f"Testing node manager at {NODE_URL}\n")

    # Fast checks — no build needed
    await test_health()
    await test_auth_rejected()

    # Dataset setup — downloads if missing (may be slow on first run)
    print("Setting up dataset (may download on first run)...")
    await test_setup()
    await test_health_shows_active_dataset()

    # Build — slow on first run, fast on cache hit
    print("Building task 0 (slow on first run)...")
    await test_build_happy_path()
    await test_build_idempotent()

    # Compose / exec / file transfer
    await test_compose_lifecycle()
    await test_upload_and_download()

    # Concurrency
    print("Running concurrent tests...")
    await test_concurrent_builds()
    await test_concurrent_compose_up()

    print("\nAll node manager tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
