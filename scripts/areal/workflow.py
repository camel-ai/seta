"""Shared AReaL workflow and config for the seta-env agent.

Both eval.py (inference-only) and rl_train.py (RL training) import from here.
"""

import logging
import os
from dataclasses import dataclass, field

from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.camel.openai_model import AReaLOpenAICompatibleModel
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker

from seta_env.orchestrators.grpo_rollout import GRPORollout
from seta_env.utils.configs import TerminalEnvConfig


logger = logging.getLogger(__name__)


_TERMINATION_REASON_METRICS = {
    "task_finished": "agent_termination_task_finished",
    "max_iteration_reached": "agent_termination_max_iteration_reached",
    "max_tokens_exceeded": "agent_termination_max_tokens_exceeded",
    "completion_length_exceeded": "agent_termination_completion_length_exceeded",
}


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig(GRPOConfig):
    """GRPOConfig extension shared by eval and rl_train scripts.

    Adds the seta-env TerminalEnvConfig (agent, runtime, env settings)
    and the number of parallel trajectories per task.

    Model is always None in TerminalEnvConfig for AReaL — the model is
    created externally via ArealOpenAI and passed as model_config_override.
    """
    n_trajs: int = field(
        default=1,
        metadata={"help": "Number of parallel trajectories per task."},
    )
    terminal_env: TerminalEnvConfig = field(
        default_factory=TerminalEnvConfig,
        metadata={"help": "TerminalEnvironment configuration (agent, runtime, env)."},
    )


# ── Workflow ────────────────────────────────────────────────────────────────────

class CamelRLVRWorkflow(RolloutWorkflow):
    """AReaL RolloutWorkflow backed by GRPORollout.

    Each arun_episode call:
      1. Creates one ArealOpenAI client per trajectory.
      2. Builds GRPORollout with per-trajectory model configs.
      3. Runs all N trajectories concurrently.
      4. Attaches rewards to clients and exports completions for AReaL.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        terminal_env_cfg: TerminalEnvConfig,
        dump_dir: str,
        n_trajs: int = 1,
        max_tokens: int = 32768,
        rollout_stat_scope: str = "rollout",
        filter_uniform_reward: bool = False,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.terminal_env_cfg = terminal_env_cfg
        self.dump_dir = dump_dir
        self.n_trajs = n_trajs
        self.max_tokens = max_tokens
        self.rollout_stat_scope = rollout_stat_scope
        self.filter_uniform_reward = filter_uniform_reward
        os.makedirs(dump_dir, exist_ok=True)

    async def arun_episode(self, engine, data):
        task_name = data.get("task_name")
        logger.info("\n%s\n[EPISODE START] Task %s\n%s", "=" * 70, task_name, "=" * 70)

        # Build clients and model configs upfront as a list; GRPORollout indexes by traj_i.
        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer, tool_call_parser="qwen25")
            for _ in range(self.n_trajs)
        ]
        model_configs = [
            {"model": AReaLOpenAICompatibleModel(
                openai_client=client,
                tokenizer=self.tokenizer,
                model_type="areal",
                model_config_dict={
                    "max_tokens": self.max_tokens,
                    "max_completion_tokens": self.gconfig.max_new_tokens,
                },
            )}
            for client in clients
        ]

        rollout = GRPORollout(
            cfg=self.terminal_env_cfg,
            model_config_override=model_configs,
        )

        results = await rollout.run(data, n_trajs=self.n_trajs)

        logger.info("\n%s\n[EPISODE END] Task %s\n%s", "=" * 70, task_name, "=" * 70)

        rewards = [r for _, r in results]

        # Optional: discard episodes where every valid trajectory got the same reward.
        if self.filter_uniform_reward:
            valid_rewards = [r for r in rewards if r is not None]
            if not valid_rewards:
                logger.warning(
                    "[Rank %s] Task %s: all trajectories failed.",
                    os.getenv("RANK"),
                    task_name,
                )
                return None
            if all(r == valid_rewards[0] for r in valid_rewards):
                logger.warning(
                    "[Rank %s] Task %s: uniform reward across trajectories - discarding.",
                    os.getenv("RANK"),
                    task_name,
                )
                return {}

        completions_with_reward = {}
        rollout_stats = stats_tracker.get(self.rollout_stat_scope)
        for i, (run_info, reward) in enumerate(results):
            if reward is None:
                logger.warning(
                    "[Rank %s] Task %s, Trajectory %s failed.",
                    os.getenv("RANK"),
                    task_name,
                    i,
                )
                failed_dir = os.path.join(self.dump_dir, "failed_tasks")
                os.makedirs(failed_dir, exist_ok=True)
                with open(os.path.join(failed_dir, f"{task_name}_traj_{i}.txt"), "w") as fh:
                    fh.write(f"Task {task_name} trajectory {i} failed.\nrun_info: {run_info}\n")
                continue

            logger.debug(
                "[Rank %s] Task %s, Trajectory %s reward: %s",
                os.getenv("RANK"),
                task_name,
                i,
                reward,
            )
            rollout_stats.scalar(reward=reward)
            clients[i].set_final_reward(reward)
            clients[i].apply_reward_discount(turn_discount=0.9)
            interactions = clients[i].export_interactions(style="individual")
            rollout_stats.scalar(
                num_turns=len(interactions),
                num_exported_interactions=len(interactions),
            )

            agent_summary = run_info.get("agent_summary") or {}
            scalar_summary = {}
            for metric_key in (
                "iteration_count",
                "total_tool_calls",
                "max_parallel_tool_call",
                "parse_error_count",
                "total_tokens",
            ):
                value = agent_summary.get(metric_key)
                if value is not None and isinstance(value, (int, float, bool)):
                    scalar_summary[metric_key] = float(value)
            if scalar_summary:
                rollout_stats.scalar(**scalar_summary)

            termination_reason = agent_summary.get("important_termination_reason")
            if termination_reason is not None:
                termination_scalars = {
                    metric_name: float(termination_reason == reason_key)
                    for reason_key, metric_name in _TERMINATION_REASON_METRICS.items()
                }
                rollout_stats.scalar(**termination_scalars)

            completions_with_reward.update(interactions)

        rollout_stats.scalar(
            num_full_passes=sum(
                1
                for run_info, reward in results
                if reward is not None
                and (evaluation := run_info.get("evaluation"))
                and all(bool(v) for v in evaluation.values())
            )
        )
        rollout_stats.scalar(
            num_trajectories_failed=sum(1 for r in rewards if r is None)
        )

        if not completions_with_reward:
            logger.warning("All trajectories failed for task %s.", task_name)
            return None

        logger.info("[Rank %s] Task %s completed.", os.getenv("RANK"), task_name)
        return completions_with_reward
