import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.environments.docker.remote_docker_environment import RemoteDockerEnvironment
from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
from seta_env.environments.terminal_env import TerminalEnvironment
from seta_env.utils.configs import (
    TerminalEnvConfig,
    build_agent_config, build_env_config, build_model_config,
)

logger = logging.getLogger(__name__)


@dataclass
class SlotAssignment:
    node_url: str
    slot_id: int


class GRPORollout:
    """Orchestrates a full GRPO group rollout for one task across any backend.

    Supported backends (runtime.env_type):
        "remote_docker"  — allocates slots from a scheduler, sets up the dataset
                           on each assigned node, builds the image remotely, then
                           runs N TerminalEnvironment.step() calls concurrently.
        "docker"         — builds the Docker image locally once, then runs N
                           TerminalEnvironment.step() calls concurrently.  No
                           scheduler required.
        "daytona"        — ensures the Daytona snapshot is active once, then runs
                           N TerminalEnvironment.step() calls concurrently.  No
                           scheduler required.
    """

    def __init__(
        self,
        cfg: TerminalEnvConfig,
        model_config_override: dict | list | Callable | None = None,
    ):
        """
        Args:
            cfg: TerminalEnvConfig with agent, model, runtime, env settings.
            model_config_override: Override for the model config. Used when
                the model is created externally (e.g. AReaL). Can be:
                - dict: ``{"model": <ModelBackend>}`` shared across all trajectories.
                - list: one dict per trajectory, indexed by traj_i.
                - callable: zero-argument factory called once per trajectory.
                - None: use cfg.model (must not be None).
        """
        self._cfg = cfg
        self._environment_type = cfg.runtime.env_type
        self._trial_root = Path(cfg.runtime.trial_root)

        scheduler_url = cfg.runtime.scheduler_url
        self._scheduler_url = scheduler_url.rstrip("/") if scheduler_url else None
        self._api_key = cfg.runtime.node_api_key

        # Build dicts for TerminalEnvironment (backward compat)
        self._agent_config = build_agent_config(cfg.agent)
        self._env_config = build_env_config(cfg.runtime, cfg.env)

        # Model config: override takes precedence, else build from cfg.model
        if model_config_override is not None:
            self._model_config = model_config_override
        else:
            self._model_config = build_model_config(cfg.model)

        self._http_timeout = 300.0

    async def run(
        self,
        task: dict,
        n_trajs: int,
        task_id: str | None = None,
    ) -> list[tuple[dict, float | None]]:
        """Run N parallel trajectories for one task.

        Args:
            task: {"task_name", "task_path", "instruction", optionally "dataset_name"}
            n_trajs: number of parallel trajectories
            task_id: unique per training step; auto-generated if None

        Returns:
            list of (run_info, reward) of length n_trajs
        """
        task_id = task_id or f"{task['task_name']}_{uuid.uuid4().hex[:8]}"

        if self._environment_type == "remote_docker":
            return await self._run_remote(task, n_trajs, task_id)
        else:
            return await self._run_local(task, n_trajs)

    # ── remote_docker path ────────────────────────────────────────────────────

    async def _run_remote(
        self, task: dict, n_trajs: int, task_id: str
    ) -> list[tuple[dict, float | None]]:
        assignments = await self._allocate(task_id, n_trajs)
        try:
            unique_nodes = list({a.node_url for a in assignments})
            await asyncio.gather(*[
                self._build_remote(node_url, task)
                for node_url in unique_nodes
            ])
            results = await asyncio.gather(*[
                self._run_one(assignments[i], task, traj_i=i)
                for i in range(n_trajs)
            ])
        finally:
            await self._release(task_id)
        return list(results)

    async def _build_remote(self, node_url: str, task: dict) -> None:
        """Build the Docker image on the allocated remote node."""
        task_obj = Task(Path(task["task_path"]))
        trial_paths = TrialPaths(trial_dir=self._trial_root / f"_build_{task['task_name']}")
        trial_paths.mkdir()
        remote_env = RemoteDockerEnvironment(
            node_manager_url=node_url,
            api_key=self._api_key,
            environment_name=task["task_name"],
            session_id=f"build_{task['task_name']}",
            trial_paths=trial_paths,
            task_env_config=task_obj.config.environment,
        )
        runtime = DockerHarborRuntime(environment=remote_env)
        await runtime.build()

    async def _allocate(self, task_id: str, n_slots: int) -> list[SlotAssignment]:
        async with httpx.AsyncClient(base_url=self._scheduler_url, timeout=30) as c:
            r = await c.post("/allocate_group", json={"task_id": task_id, "n_slots": n_slots})
            if r.status_code >= 400:
                raise RuntimeError(f"Scheduler allocate_group failed ({r.status_code}): {r.text}")
            data = r.json()
        return [SlotAssignment(**a) for a in data["assignments"]]

    async def _release(self, task_id: str) -> None:
        try:
            async with httpx.AsyncClient(base_url=self._scheduler_url, timeout=30) as c:
                await c.post("/release_group", json={"task_id": task_id})
        except Exception as e:
            logger.warning("Failed to release slots for task %s: %s", task_id, e)

    # ── local backends path (docker / daytona) ────────────────────────────────

    async def _run_local(
        self, task: dict, n_trajs: int
    ) -> list[tuple[dict, float | None]]:
        await self._build_local(task)
        results = await asyncio.gather(*[
            self._run_one(None, task, traj_i=i)
            for i in range(n_trajs)
        ])
        return list(results)

    async def _build_local(self, task: dict) -> None:
        """Build the image/snapshot once for local backends (docker, daytona)."""
        runtime = DockerHarborRuntime(
            task_dir=task["task_path"],
            trial_root=str(self._trial_root / "_build"),
            session_id=f"build_{task['task_name']}",
            environment_type=self._environment_type,
        )
        await runtime.build()

    # ── shared ────────────────────────────────────────────────────────────────

    async def _run_one(
        self,
        assignment: SlotAssignment | None,
        task: dict,
        traj_i: int,
    ) -> tuple[dict, float | None]:
        session_id = f"{task['task_name']}_t{traj_i}_{uuid.uuid4().hex[:6]}"
        if callable(self._model_config):
            model_cfg = self._model_config()
        elif isinstance(self._model_config, list):
            model_cfg = self._model_config[traj_i]
        else:
            model_cfg = self._model_config

        runtime_config: dict = {
            "task_dir":         task["task_path"],
            "trial_root":       str(self._trial_root),
            "environment_type": self._environment_type,
            "toolkit":          self._cfg.runtime.toolkit,
        }
        if self._environment_type == "remote_docker" and assignment is not None:
            runtime_config["node_manager_url"] = assignment.node_url
            runtime_config["node_api_key"]     = self._api_key

        te = TerminalEnvironment(
            agent_config=self._agent_config,
            model_config=model_cfg,
            runtime_config=runtime_config,
            env_config=self._env_config,
        )
        return await te.step(task, uid=session_id, traj_i=traj_i)
