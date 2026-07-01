"""Tests for env_scheduler — routing, affinity, load balancing, and HTTP endpoints."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from seta_env.services.env_scheduler import NodeState, Scheduler


# ── Unit tests: Scheduler routing logic ─────────────────────────────────────


@pytest.mark.asyncio
async def test_affinity_same_task():
    """8 requests for same task_id → all routed to same node."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=16),
        NodeState(url="http://node2:8002", total_slots=16),
    ])

    first = await scheduler.pick_node("task_42")
    first_url = first.url
    await scheduler.release(first)

    for _ in range(7):
        node = await scheduler.pick_node("task_42")
        assert node.url == first_url, "Affinity should route to same node"
        await scheduler.release(node)


@pytest.mark.asyncio
async def test_load_balance_no_affinity():
    """No affinity → pick node with best free_ratio."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=16, active_slots=14),
        NodeState(url="http://node2:8002", total_slots=16, active_slots=0),
    ])

    node = await scheduler.pick_node("new_task")
    assert node.url == "http://node2:8002"
    await scheduler.release(node)


@pytest.mark.asyncio
async def test_different_tasks_distributed():
    """Different task_ids can go to different nodes."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=16),
        NodeState(url="http://node2:8002", total_slots=16),
    ])

    n1 = await scheduler.pick_node("task_a")
    n2 = await scheduler.pick_node("task_b")

    # With both nodes empty, first goes to either; second should go to the other
    # (since first node now has 1 active, second has 0 → better free_ratio)
    assert n1.url != n2.url, "Different tasks should spread across nodes"
    await scheduler.release(n1)
    await scheduler.release(n2)


@pytest.mark.asyncio
async def test_capacity_exhaustion():
    """All nodes full → HTTPException 503."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=1, active_slots=1),
        NodeState(url="http://node2:8002", total_slots=1, active_slots=1),
    ])

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await scheduler.pick_node("any_task")
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_affinity_expires():
    """After affinity window, task can go to a different node."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=16),
        NodeState(url="http://node2:8002", total_slots=16),
    ])

    # Set affinity with an old timestamp
    n1 = await scheduler.pick_node("old_task")
    n1.task_affinity["old_task"] = time.monotonic() - 200  # expired
    await scheduler.release(n1)

    # Now should pick based on free_ratio, not affinity
    # Both nodes equal → could go to either (no assertion on which)
    n2 = await scheduler.pick_node("old_task")
    await scheduler.release(n2)
    # Just verify it didn't crash


@pytest.mark.asyncio
async def test_release_decrements():
    """release() decrements active_slots."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=16),
    ])

    node = await scheduler.pick_node("task_x")
    assert node.active_slots == 1
    await scheduler.release(node)
    assert node.active_slots == 0


@pytest.mark.asyncio
async def test_affinity_prefers_node_with_capacity():
    """Affinity node full → falls through to load balance."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=2, active_slots=0),
        NodeState(url="http://node2:8002", total_slots=2, active_slots=0),
    ])

    # Build affinity to node1
    n = await scheduler.pick_node("task_fill")
    assert n.url == "http://node1:8002" or n.url == "http://node2:8002"
    affinity_url = n.url

    # Fill up the affinity node
    n2 = await scheduler.pick_node("task_fill")
    assert n2.url == affinity_url  # affinity
    # Now affinity node is full (2/2)

    # Next request should go to the other node
    n3 = await scheduler.pick_node("task_fill")
    other_url = "http://node2:8002" if affinity_url == "http://node1:8002" else "http://node1:8002"
    assert n3.url == other_url

    await scheduler.release(n)
    await scheduler.release(n2)
    await scheduler.release(n3)


def test_status():
    """status() returns per-node breakdown."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=16, active_slots=3),
        NodeState(url="http://node2:8002", total_slots=8, active_slots=0),
    ])
    s = scheduler.status()
    assert len(s) == 2
    assert s[0]["active_slots"] == 3
    assert s[1]["free_ratio"] == 1.0


def test_cleanup_affinity():
    """cleanup_affinity removes expired entries."""
    scheduler = Scheduler([
        NodeState(url="http://node1:8002", total_slots=16),
    ])
    # Add old affinity
    scheduler._nodes[0].task_affinity["old"] = time.monotonic() - 500
    scheduler._nodes[0].task_affinity["recent"] = time.monotonic()

    removed = scheduler.cleanup_affinity()
    assert removed == 1
    assert "old" not in scheduler._nodes[0].task_affinity
    assert "recent" in scheduler._nodes[0].task_affinity


# ── HTTP endpoint tests ─────────────────────────────────────────────────────


@pytest.fixture
def nodes_yaml(tmp_path):
    """Write a test nodes.yaml."""
    cfg = {
        "nodes": [
            {"url": "http://fake-node1:8002", "slots": 4},
            {"url": "http://fake-node2:8002", "slots": 4},
        ]
    }
    p = tmp_path / "nodes.yaml"
    p.write_text(json.dumps(cfg))
    return str(p)


@pytest.fixture
def client(nodes_yaml):
    os.environ["NODES_YAML"] = nodes_yaml
    from fastapi.testclient import TestClient
    from seta_env.services.env_scheduler import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    os.environ.pop("NODES_YAML", None)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["nodes"] == 2
    assert data["total_slots"] == 8


def test_status_endpoint(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 2


def test_step_node_unreachable(client):
    """POST /step returns error when node is unreachable (no mock needed)."""
    resp = client.post("/step", json={
        "task": {"task_name": "test_task"},
        "uid": "sess_1",
        "agent_config": {},
        "llm_config": {},
    })

    assert resp.status_code == 200
    data = resp.json()
    # Node is fake, so we expect a connection error
    assert data["error"] is not None
    assert data["run_info"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
