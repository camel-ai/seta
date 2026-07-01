"""Env Scheduler — routes step requests to env_service nodes with affinity and load balancing.

Runs on the local/GPU machine. Transparent proxy: picks the best node,
forwards the StepRequest, returns the StepResponse.

Usage:
    NODES_YAML=nodes.yaml ENV_SERVICE_API_KEY=dev-key \
        uvicorn seta_env.services.env_scheduler:app --host 127.0.0.1 --port 8003
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────

AFFINITY_WINDOW = float(os.environ.get("AFFINITY_WINDOW", "120.0"))  # seconds
NODE_API_KEY = os.environ.get("ENV_SERVICE_API_KEY", "")
NODE_API_KEY = os.environ.get("ENV_SERVICE_API_KEY", "")
STEP_TIMEOUT = float(os.environ.get("STEP_TIMEOUT", "900.0"))  # 15 min


# ── Node state ──────────────────────────────────────────────────────────────


@dataclass
class NodeState:
    url: str
    total_slots: int
    active_slots: int = 0
    task_affinity: dict[str, float] = field(default_factory=dict)

    @property
    def free_slots(self) -> int:
        return max(0, self.total_slots - self.active_slots)

    @property
    def free_ratio(self) -> float:
        return self.free_slots / self.total_slots if self.total_slots > 0 else 0.0


# ── Scheduler ───────────────────────────────────────────────────────────────


class Scheduler:
    def __init__(self, nodes: list[NodeState]):
        self._nodes = nodes
        self._lock = asyncio.Lock()

    async def pick_node(self, task_id: str) -> NodeState:
        async with self._lock:
            now = time.monotonic()

            # 1. Affinity: task_id recently sent to a node with capacity
            for node in self._nodes:
                ts = node.task_affinity.get(task_id)
                if ts and (now - ts) < AFFINITY_WINDOW and node.free_slots > 0:
                    node.active_slots += 1
                    node.task_affinity[task_id] = now
                    return node

            # 2. No affinity → best free_ratio
            candidates = [n for n in self._nodes if n.free_slots > 0]
            if not candidates:
                raise HTTPException(503, "All nodes are at capacity")

            best = max(candidates, key=lambda n: n.free_ratio)
            best.active_slots += 1
            best.task_affinity[task_id] = now
            return best

    async def release(self, node: NodeState):
        async with self._lock:
            node.active_slots = max(0, node.active_slots - 1)

    def status(self) -> list[dict]:
        now = time.monotonic()
        return [
            {
                "url": n.url,
                "total_slots": n.total_slots,
                "active_slots": n.active_slots,
                "free_ratio": round(n.free_ratio, 3),
                "task_affinity": {
                    k: round(now - v, 1)
                    for k, v in n.task_affinity.items()
                    if now - v < AFFINITY_WINDOW
                },
            }
            for n in self._nodes
        ]

    def cleanup_affinity(self):
        now = time.monotonic()
        removed = 0
        for node in self._nodes:
            expired = [
                k for k, v in node.task_affinity.items()
                if now - v > AFFINITY_WINDOW * 2
            ]
            for k in expired:
                del node.task_affinity[k]
                removed += 1
        return removed


# ── Global state ────────────────────────────────────────────────────────────

scheduler: Scheduler | None = None


# ── Request models ──────────────────────────────────────────────────────────


class SetupRequest(BaseModel):
    dataset_name: str
    hf_token: str = ""


class ConfigUpdateRequest(BaseModel):
    config_path: str | None = None
    config: dict | None = None


# ── App lifecycle ───────────────────────────────────────────────────────────


_url_rewrite: dict[str, str] = {}


def _load_config() -> tuple[list[NodeState], dict[str, str], str]:
    nodes_yaml = os.environ.get(
        "NODES_YAML", str(Path(__file__).parent / "nodes.yaml")
    )
    data = yaml.safe_load(open(nodes_yaml))
    nodes = [
        NodeState(url=n["url"], total_slots=n["slots"])
        for n in data["nodes"]
    ]
    url_rewrite = data.get("url_rewrite") or {}
    api_key = data.get("api_key") or os.environ.get("ENV_SERVICE_API_KEY", "")
    return nodes, url_rewrite, api_key


def _rewrite_url(url: str) -> str:
    """Apply url_rewrite mapping. Longest prefix match."""
    for src, dst in _url_rewrite.items():
        if url.startswith(src):
            return dst + url[len(src):]
    return url


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, _url_rewrite, NODE_API_KEY
    nodes, _url_rewrite, NODE_API_KEY = _load_config()
    scheduler = Scheduler(nodes)
    if _url_rewrite:
        logger.info("URL rewrite rules: %s", _url_rewrite)
    logger.info(
        "env_scheduler started: %d nodes, %d total slots",
        len(nodes),
        sum(n.total_slots for n in nodes),
    )

    affinity_task = asyncio.create_task(_affinity_cleanup_loop())
    yield
    affinity_task.cancel()


app = FastAPI(title="Env Scheduler", lifespan=lifespan)


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "nodes": len(scheduler._nodes),
        "total_slots": sum(n.total_slots for n in scheduler._nodes),
        "active_slots": sum(n.active_slots for n in scheduler._nodes),
    }


@app.get("/status")
async def status():
    return {"nodes": scheduler.status()}


@app.post("/step")
async def step(request: dict, x_api_key: str = Header("")):
    """Transparent proxy: pick best node, forward request, return response."""
    task_id = (
        request.get("task", {}).get("task_id")
        or request.get("task", {}).get("task_name")
        or request.get("uid", "unknown")
    )

    # Rewrite model_url if url_rewrite rules are configured
    if _url_rewrite and "model_url" in request:
        request["model_url"] = _rewrite_url(request["model_url"])

    node = await scheduler.pick_node(task_id)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(STEP_TIMEOUT, connect=30.0)
        ) as client:
            resp = await client.post(
                f"{node.url}/step",
                json=request,
                headers={"X-API-Key": NODE_API_KEY},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error("Node %s returned %d: %s", node.url, e.response.status_code, e.response.text)
        return {"run_info": None, "reward": None, "error": f"Node error: {e.response.status_code}"}
    except Exception as e:
        logger.error("Failed to reach node %s: %s", node.url, e)
        return {"run_info": None, "reward": None, "error": str(e)}
    finally:
        await scheduler.release(node)


@app.post("/setup_dataset")
async def setup_dataset(req: SetupRequest, x_api_key: str = Header("")):
    """Fan out dataset setup to all env_service nodes in parallel."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        tasks = [
            client.post(
                f"{n.url}/setup",
                json=req.model_dump(),
                headers={"X-API-Key": NODE_API_KEY},
            )
            for n in scheduler._nodes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "results": [
            r.json() if isinstance(r, httpx.Response) else {"error": str(r)}
            for r in results
        ]
    }


@app.post("/url_rewrite")
async def update_url_rewrite(rewrite_map: dict, x_api_key: str = Header("")):
    """Update url_rewrite rules at runtime. Replaces existing rules."""
    global _url_rewrite
    _url_rewrite = rewrite_map
    logger.info("URL rewrite updated: %s", _url_rewrite)
    return {"status": "ok", "url_rewrite": _url_rewrite}


@app.get("/url_rewrite")
async def get_url_rewrite():
    return {"url_rewrite": _url_rewrite}


@app.post("/config")
async def update_config(req: ConfigUpdateRequest, x_api_key: str = Header("")):
    """Fan out config update to all env_service nodes.

    Accepts either:
      - ``config_path``: path to a YAML file on the scheduler machine
      - ``config``: inline JSON config dict
    If both are provided, ``config_path`` takes precedence.
    """
    if req.config_path:
        p = Path(req.config_path)
        if not p.exists():
            raise HTTPException(400, f"Config file not found: {req.config_path}")
        try:
            new_config = yaml.safe_load(p.read_text())
        except Exception as e:
            raise HTTPException(400, f"Failed to parse {req.config_path}: {e}")
    elif req.config:
        new_config = req.config
    else:
        raise HTTPException(400, "Provide either config_path or config")

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            client.post(
                f"{n.url}/config",
                json=new_config,
                headers={"X-API-Key": NODE_API_KEY},
            )
            for n in scheduler._nodes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "results": [
            r.json() if isinstance(r, httpx.Response) else {"error": str(r)}
            for r in results
        ]
    }


@app.post("/cleanup")
async def cleanup(x_api_key: str = Header("")):
    """Fan out full Docker cleanup to all nodes."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            client.post(
                f"{n.url}/cleanup",
                headers={"X-API-Key": NODE_API_KEY},
            )
            for n in scheduler._nodes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return {
        "status": "ok",
        "results": [
            r.json() if isinstance(r, httpx.Response) else {"error": str(r)}
            for r in results
        ],
    }


# ── Background tasks ────────────────────────────────────────────────────────


async def _affinity_cleanup_loop():
    while True:
        await asyncio.sleep(60)
        try:
            removed = scheduler.cleanup_affinity()
            if removed:
                logger.info("Cleaned up %d expired affinity entries", removed)
        except Exception as e:
            logger.warning("Affinity cleanup error: %s", e)
