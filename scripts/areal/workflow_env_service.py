"""AReaL workflow that runs TerminalEnvironment on remote env_service nodes.

Uses ProxyServer to capture model interactions for training. Manages
ProxySession lifecycle directly via HTTP (no ProcessPoolExecutor needed
since the agent runs remotely, not locally).
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path

import aiofiles
import aiofiles.os
import httpx

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.proxy import ProxyServer, ProxySession
from areal.utils import stats_tracker

logger = logging.getLogger(__name__)


_TERMINATION_REASON_METRICS = {
    "task_finished": "agent_termination_task_finished",
    "max_iteration_reached": "agent_termination_max_iteration_reached",
    "max_tokens_exceeded": "agent_termination_max_tokens_exceeded",
    "completion_length_exceeded": "agent_termination_completion_length_exceeded",
}


class EnvServiceRLVRWorkflow(RolloutWorkflow):
    """AReaL workflow backed by remote env_service nodes.

    For each trajectory:
      1. Start a ProxySession (HTTP to ProxyServer) → get session_id
      2. Send StepRequest to env_scheduler with proxy URL as model endpoint
      3. Agent on env_service calls model via proxy → ProxyServer captures
      4. End session with reward
      5. get_completions() returns AReaL training data
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        proxy_server: ProxyServer,
        env_scheduler_url: str,
        env_service_api_key: str = "env-service-dev-key",
        dataset_name: str = "",
        trial_name: str = "",
        local_trial_root: str = "",
        dump_dir: str | None = None,
        n_trajs: int = 1,
        step_timeout: float = 900.0,
        rollout_stat_scope: str = "rollout",
        export_style: str = "individual",
        filter_uniform_reward: bool = False,
    ):
        self.proxy_server = proxy_server
        self.n_trajs = n_trajs
        self.gconfig = gconfig
        self.env_scheduler_url = env_scheduler_url.rstrip("/")
        self.env_service_api_key = env_service_api_key
        self.dataset_name = dataset_name
        self.trial_name = trial_name
        self.local_trial_root = local_trial_root
        self.dump_dir = dump_dir
        self.step_timeout = step_timeout
        self.rollout_stat_scope = rollout_stat_scope
        self.export_style = export_style
        self.filter_uniform_reward = filter_uniform_reward

        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    async def _run_one_trajectory(
        self, data: dict, traj_i: int, proxy_addr: str,
    ) -> tuple[dict | None, str, float | None]:
        """Run a single trajectory via ProxySession + env_service.

        Returns (run_info, session_id, reward).
        """
        task_name = data.get("task_name", "unknown")
        uid = f"{task_name}_t{traj_i}_{uuid.uuid4().hex[:6]}"

        async with ProxySession(base_url=proxy_addr) as session:
            session_id = session.session_id
            model_url = f"{proxy_addr}/{session_id}"
            model_api_key = session_id

            step_request = {
                "task": data,
                "uid": uid,
                "traj_i": traj_i,
                "model_url": model_url,
                "model_api_key": model_api_key,
                "dataset_name": self.dataset_name,
                "task_name": task_name,
                "trial_name": self.trial_name,
            }

            run_info = None
            reward = None
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.step_timeout, connect=30.0)
                ) as client:
                    resp = await client.post(
                        f"{self.env_scheduler_url}/step",
                        json=step_request,
                        headers={"X-API-Key": self.env_service_api_key},
                    )
                    resp.raise_for_status()
                    result = resp.json()

                run_info = result.get("run_info")
                reward = result.get("reward")
                error = result.get("error")
                if error:
                    logger.warning("env_service error for %s traj %d: %s", task_name, traj_i, error)
            except Exception as e:
                logger.warning("Failed for %s traj %d: %s", task_name, traj_i, e)

            await session.set_reward(reward if reward is not None else 0.0)

        # Save run_info locally (same layout as eval.py)
        if self.local_trial_root and run_info:
            try:
                import json
                trial_dir = Path(self.local_trial_root) / uid
                trial_dir.mkdir(parents=True, exist_ok=True)
                (trial_dir / "run_info.json").write_text(
                    json.dumps(run_info, indent=2, default=str)
                )
            except Exception as e:
                logger.warning("Failed to save local run_info for %s: %s", uid, e)

        return run_info, session_id, reward

    async def arun_episode(self, engine: InferenceEngine, data):
        task_name = data.get("task_name", "unknown")
        logger.info("\n%s\n[EPISODE START] Task %s\n%s", "=" * 70, task_name, "=" * 70)

        proxy_addr = f"{self.proxy_server.public_addr}/v1"

        # Launch N trajectories concurrently (pure async, no processes)
        results = await asyncio.gather(*[
            self._run_one_trajectory(data, i, proxy_addr)
            for i in range(self.n_trajs)
        ])

        run_infos, session_ids, rewards = zip(*results)

        # ── Filtering ────────────────────────────────────────────────────
        if self.filter_uniform_reward:
            valid_rewards = [r for r in rewards if r is not None]
            if not valid_rewards:
                logger.warning("Task %s: all trajectories failed.", task_name)
                return None
            if all(r == valid_rewards[0] for r in valid_rewards):
                logger.warning("Task %s: uniform reward — discarding.", task_name)
                return {}

        # ── Stats ────────────────────────────────────────────────────────
        rollout_stats = stats_tracker.get(self.rollout_stat_scope)
        for run_info, reward in zip(run_infos, rewards):
            if reward is None:
                continue
            rollout_stats.scalar(reward=reward)
            if run_info:
                agent_summary = run_info.get("agent_summary") or {}
                for key in ("iteration_count", "total_tool_calls", "max_parallel_tool_call",
                            "parse_error_count", "total_tokens"):
                    val = agent_summary.get(key)
                    if val is not None and isinstance(val, (int, float)):
                        rollout_stats.scalar(**{key: float(val)})
                termination_reason = agent_summary.get("important_termination_reason")
                if termination_reason:
                    rollout_stats.scalar(**{
                        name: float(termination_reason == reason)
                        for reason, name in _TERMINATION_REASON_METRICS.items()
                    })

        rollout_stats.scalar(
            num_full_passes=sum(
                1 for ri, rw in zip(run_infos, rewards)
                if rw is not None and ri
                and all(bool(v) for v in (ri.get("evaluation") or {}).values())
            ),
            num_trajectories_failed=sum(1 for r in rewards if r is None),
        )

        # ── Completions (sessions already ended, returns immediately) ────
        completions = await self.proxy_server.get_completions(
            session_ids=list(session_ids),
            style=self.export_style,
            discount=0.9,
        )

        # ── Dump ─────────────────────────────────────────────────────────
        if self.dump_dir and completions:
            for sid, completion in completions.items():
                version = completion.model_response.output_versions[-1]
                dump_path = os.path.join(self.dump_dir, str(version))
                await aiofiles.os.makedirs(dump_path, exist_ok=True)
                qid = data.get("query_id") or data.get("id") or task_name
                async with aiofiles.open(
                    os.path.join(dump_path, f"{qid}_{sid}.txt"), "a"
                ) as f:
                    await f.write(f"completion is: {completion}\n")

        if not completions:
            logger.warning("All trajectories failed for task %s.", task_name)
            return None

        logger.info("[EPISODE END] Task %s completed.", task_name)
        return completions
