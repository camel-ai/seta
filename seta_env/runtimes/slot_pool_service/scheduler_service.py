"""Scheduler Service — FastAPI service running locally on the training machine.

Tracks slot availability across all remote node managers and allocates groups
of N slots atomically for GRPO rollouts.

Start:
    cd seta_env/runtimes/slot_pool_service
    uvicorn scheduler_service:app --host 127.0.0.1 --port 8000

Or via module path from project root:
    uvicorn seta_env.runtimes.slot_pool_service.scheduler_service:app \
        --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────

MAX_GROUP_SIZE = 16
_HERE = Path(__file__).parent

app = FastAPI(title="Scheduler Service")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NodeConfig:
    url: str
    slots: int
    api_key: str = "harbor-node-dev-key"


@dataclass
class SlotAssignment:
    node_url: str
    slot_id: int


@dataclass
class NodeState:
    url: str
    total_slots: int
    slots: dict[int, str | None] = field(default_factory=dict)

    def __post_init__(self):
        if not self.slots:
            self.slots = {i: None for i in range(self.total_slots)}

    @property
    def free_count(self) -> int:
        return sum(1 for v in self.slots.values() if v is None)

    def allocate(self, n: int, task_id: str) -> list[int]:
        """Allocate n free slots for task_id; returns slot IDs or raises on shortfall."""
        allocated = []
        for slot_id, owner in self.slots.items():
            if owner is None:
                self.slots[slot_id] = task_id
                allocated.append(slot_id)
                if len(allocated) == n:
                    return allocated
        # Should not reach here if caller checked free_count first, but guard anyway.
        for slot_id in allocated:
            self.slots[slot_id] = None
        raise RuntimeError(f"Allocation shortfall on {self.url}: need {n}, have {self.free_count}")

    def release(self, task_id: str) -> int:
        """Release all slots owned by task_id; returns count released."""
        count = 0
        for slot_id in list(self.slots):
            if self.slots[slot_id] == task_id:
                self.slots[slot_id] = None
                count += 1
        return count


# ── Scheduler ─────────────────────────────────────────────────────────────────

class Scheduler:
    def __init__(self, nodes: list[NodeConfig]):
        self._nodes: list[NodeState] = [
            NodeState(url=n.url, total_slots=n.slots) for n in nodes
        ]
        self._lock = asyncio.Lock()
        self._groups: dict[str, list[SlotAssignment]] = {}
        self._group_allocated_at: dict[str, float] = {}  # task_id -> monotonic timestamp

    async def allocate_group(
        self, task_id: str, n_slots: int
    ) -> list[SlotAssignment]:
        """Atomically allocate n_slots for task_id across nodes.

        Strategy: sort nodes by free_count descending, fill greedily from
        the most-available node first (keeps groups co-located when possible).
        """
        if n_slots > MAX_GROUP_SIZE:
            raise HTTPException(
                400,
                f"n_slots={n_slots} exceeds max group size {MAX_GROUP_SIZE}",
            )

        async with self._lock:
            if task_id in self._groups:
                raise HTTPException(400, f"task_id {task_id!r} already allocated")

            total_free = sum(n.free_count for n in self._nodes)
            if total_free < n_slots:
                raise HTTPException(
                    503,
                    f"Not enough free slots: need {n_slots}, total free {total_free}",
                )

            # Sort by free_ratio (free/total) descending so load stays
            # proportional to capacity across nodes of different sizes.
            sorted_nodes = sorted(
                self._nodes,
                key=lambda n: n.free_count / n.total_slots,
                reverse=True,
            )

            assignments: list[SlotAssignment] = []
            remaining = n_slots
            for node in sorted_nodes:
                if remaining == 0:
                    break
                take = min(remaining, node.free_count)
                if take == 0:
                    continue
                slot_ids = node.allocate(take, task_id)
                assignments.extend(
                    SlotAssignment(node_url=node.url, slot_id=s) for s in slot_ids
                )
                remaining -= take

            self._groups[task_id] = assignments
            self._group_allocated_at[task_id] = time.monotonic()
            return assignments

    async def release_group(self, task_id: str) -> int:
        """Release all slots for task_id. Returns count released."""
        async with self._lock:
            if task_id not in self._groups:
                return 0
            count = sum(n.release(task_id) for n in self._nodes)
            del self._groups[task_id]
            self._group_allocated_at.pop(task_id, None)
            return count

    def status(self) -> dict:
        return {
            "nodes": [
                {
                    "url": n.url,
                    "total_slots": n.total_slots,
                    "free_slots": n.free_count,
                    "slots": {str(k): v for k, v in n.slots.items()},
                }
                for n in self._nodes
            ],
            "active_groups": {
                task_id: {
                    "allocated_secs": round(time.monotonic() - self._group_allocated_at.get(task_id, now), 1),
                    "slots": [
                        {"node_url": a.node_url, "slot_id": a.slot_id}
                        for a in assignments
                    ],
                }
                for now in [time.monotonic()]
                for task_id, assignments in self._groups.items()
            },
        }


# ── Startup: load nodes.yaml ──────────────────────────────────────────────────

def _load_nodes() -> list[NodeConfig]:
    cfg_path = _HERE / "nodes.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    default_key = os.environ.get("NODE_MANAGER_API_KEY", "harbor-node-dev-key")
    nodes = []
    for n in raw["nodes"]:
        deploy = n.get("deploy") or {}
        api_key = n.get("api_key") or deploy.get("api_key") or default_key
        nodes.append(NodeConfig(url=n["url"], slots=n["slots"], api_key=api_key))
    return nodes


_scheduler: Scheduler | None = None
_nodes: list[NodeConfig] = []


@app.on_event("startup")
async def _startup():
    global _scheduler, _nodes
    _nodes = _load_nodes()
    _scheduler = Scheduler(_nodes)


def _get_scheduler() -> Scheduler:
    if _scheduler is None:
        raise HTTPException(503, "Scheduler not initialised")
    return _scheduler


# ── Request / Response Models ─────────────────────────────────────────────────

class AllocateRequest(BaseModel):
    task_id: str
    n_slots: int


class ReleaseRequest(BaseModel):
    task_id: str


class SetupDatasetRequest(BaseModel):
    dataset_name: str
    hf_token: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/allocate_group")
async def allocate_group(req: AllocateRequest):
    s = _get_scheduler()
    assignments = await s.allocate_group(req.task_id, req.n_slots)
    return {
        "task_id": req.task_id,
        "assignments": [
            {"node_url": a.node_url, "slot_id": a.slot_id} for a in assignments
        ],
    }


@app.post("/release_group")
async def release_group(req: ReleaseRequest):
    s = _get_scheduler()
    released = await s.release_group(req.task_id)
    return {"task_id": req.task_id, "released_slots": released}


@app.get("/status")
async def status():
    return _get_scheduler().status()


@app.post("/setup_dataset")
async def setup_dataset(req: SetupDatasetRequest):
    """Fan out POST /setup to all nodes in parallel. Returns per-node results."""

    async def _setup_one(node: NodeConfig) -> dict:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                payload: dict = {"dataset_name": req.dataset_name}
                token = req.hf_token or os.environ.get("HF_TOKEN", "")
                if token:
                    payload["hf_token"] = token
                resp = await client.post(
                    f"{node.url}/setup",
                    json=payload,
                    headers={"X-API-Key": node.api_key},
                )
            return {"node": node.url, "status": resp.status_code, "body": resp.json()}
        except Exception as e:
            return {"node": node.url, "status": "error", "body": f"{type(e).__name__}: {e}"}

    results = await asyncio.gather(*[_setup_one(n) for n in _nodes])
    failed = [r for r in results if r["status"] != 200]
    return {
        "dataset_name": req.dataset_name,
        "results": results,
        "success": len(failed) == 0,
        "failed_nodes": [r["node"] for r in failed],
    }


@app.post("/cleanup")
async def cleanup():
    """Fan out POST /cleanup to all nodes in parallel.

    Stops and removes ALL containers on every node and clears session state.
    Use this to fully reset all nodes between eval runs.
    """

    async def _cleanup_one(node: NodeConfig) -> dict:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{node.url}/cleanup",
                    headers={"X-API-Key": node.api_key},
                )
            return {"node": node.url, "status": resp.status_code, "body": resp.json()}
        except Exception as e:
            return {"node": node.url, "status": "error", "body": f"{type(e).__name__}: {e}"}

    results = await asyncio.gather(*[_cleanup_one(n) for n in _nodes])
    failed = [r for r in results if r["status"] != 200]

    s = _get_scheduler()
    async with s._lock:
        released_groups = list(s._groups.keys())
        for node in s._nodes:
            node.slots = {i: None for i in range(node.total_slots)}
        s._groups.clear()
        s._group_allocated_at.clear()

    return {
        "results": results,
        "success": len(failed) == 0,
        "failed_nodes": [r["node"] for r in failed],
        "released_groups": released_groups,
    }
