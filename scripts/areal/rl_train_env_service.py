"""RL training script using remote env_service nodes.

Same training loop as rl_train.py but uses EnvServiceRLVRWorkflow:
- ProxyServer captures model interactions for AReaL training
- Agents run on remote env_service nodes via env_scheduler
- FRP tunnel (optional) exposes ProxyServer to remote nodes

Usage:
    python scripts/areal/rl_train_env_service.py --config scripts/areal/configs/config_train.yaml
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_REPO_ROOT = Path(__file__).resolve().parents[2]

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.proxy import ProxyServer
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import cycle_dataloader
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.network import find_free_ports
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.utils import perf_tracer
from areal.workflow.rlvr import RLVRWorkflow

from workflow_env_service import EnvServiceRLVRWorkflow

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class EnvServiceTrainConfig:
    """GRPOConfig extension for env_service-based training.

    The TerminalEnvConfig lives on the env_service nodes (deployed separately).
    This config only has training + env_service connection settings.
    """

    # env_service connection
    env_scheduler_url: str = field(
        default="http://127.0.0.1:8003",
        metadata={"help": "URL of the env_scheduler service."},
    )
    env_service_api_key: str = field(
        default="env-service-dev-key",
        metadata={"help": "API key for env_service nodes."},
    )
    dataset_name: str = field(
        default="seta-env-harbor",
        metadata={"help": "Dataset name (must be set up on env_service nodes via /setup)."},
    )

    # Training
    n_trajs: int = field(
        default=4,
        metadata={"help": "Number of parallel trajectories per task."},
    )
    step_timeout: float = field(
        default=900.0,
        metadata={"help": "Timeout (seconds) for each env_service step request."},
    )
    filter_uniform_reward: bool = field(
        default=False,
        metadata={"help": "Discard episodes where all trajectories have the same reward."},
    )
    export_style: str = field(
        default="individual",
        metadata={"help": "Export style for completions: 'individual' or 'concat'."},
    )
    tool_call_parser: str = field(
        default="qwen25",
        metadata={"help": "Tool call parser for ArealOpenAI client."},
    )


# Compose with GRPOConfig
from areal.api.cli_args import GRPOConfig


@dataclass
class TrainConfig(GRPOConfig, EnvServiceTrainConfig):
    pass


# ── Reward function (for eval workflow, not env_service) ──────────────────────

def _pass_ratio_reward(prompt, completions, prompt_ids, completion_ids, **kwargs):
    """Placeholder eval reward. Real reward is computed by env_service."""
    return 0.0


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config, _ = load_expr_config(args, TrainConfig)
    config: TrainConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # ── Train engine ──────────────────────────────────────────────────────
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    perf_tracer.configure(config.perf_tracer, rank=rank)

    # ── Dataset ───────────────────────────────────────────────────────────
    from datasets import load_dataset as _load_dataset
    dataset_path = Path(config.train_dataset.path)
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    if "parquet" in dataset_path.suffix:
        dataset = _load_dataset(path="parquet", split="train", data_files=[str(dataset_path)])
    elif dataset_path.is_dir():
        from seta_env.dataset import load_harbor_dataset
        dataset = load_harbor_dataset(dataset_path)
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    train_dataloader = create_dataloader(
        dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # ── Inference engine ──────────────────────────────────────────────────
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    # ── Stop tokens ───────────────────────────────────────────────────────
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    # ── ProxyServer (captures model interactions for training) ────────────
    client = ArealOpenAI(
        engine=rollout,
        tokenizer=tokenizer,
        tool_call_parser=config.tool_call_parser,
        chat_template_type="concat" if config.export_style == "concat" else "hf",
    )

    free_port = find_free_ports(1)[0]
    proxy_server = ProxyServer(port=free_port, client=client)
    proxy_server.start(wait_until_ready=True)

    # Gather all proxy addresses across data-parallel ranks
    all_addresses = [None for _ in range(actor.data_parallel_world_size)]
    dist.all_gather_object(
        all_addresses, proxy_server.public_addr, group=actor.data_parallel_group
    )
    logger.info("Found %d proxy servers: %s", len(all_addresses), all_addresses)
    dist.barrier(group=actor.cpu_group)

    # ── Auto-setup FRP tunnel + url_rewrite (head rank only) ─────────────
    if actor.is_data_parallel_head():
        from seta_env.services.proxy_setup import setup_proxy_tunnels
        setup_proxy_tunnels(
            proxy_addresses=all_addresses,
            scheduler_url=config.env_scheduler_url,
        )
    dist.barrier(group=actor.cpu_group)

    dump_dir = os.path.join(StatsLogger.get_log_path(config.stats_logger), "generated")
    trial_root = os.path.abspath(
        f"{config.stats_logger.fileroot}/{config.stats_logger.experiment_name}"
        f"/{config.stats_logger.trial_name}/trials"
    )

    # ── Workflow ──────────────────────────────────────────────────────────
    workflow = EnvServiceRLVRWorkflow(
        gconfig=config.gconfig,
        proxy_server=proxy_server,
        env_scheduler_url=config.env_scheduler_url,
        env_service_api_key=config.env_service_api_key,
        dataset_name=config.dataset_name,
        trial_name=config.trial_name,
        local_trial_root=trial_root,
        dump_dir=dump_dir,
        n_trajs=config.n_trajs,
        step_timeout=config.step_timeout,
        filter_uniform_reward=config.filter_uniform_reward,
        export_style=config.export_style,
    )

    # ── Training utilities ────────────────────────────────────────────────
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor, saver, evaluator, stats_logger, train_dataloader,
        inference_engine=rollout, weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step if recover_info is not None else 0
    )

    # ── Training loop ─────────────────────────────────────────────────────
    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    logger.info(
        "Training: %s steps (%s epochs x %s steps/epoch, batch=%s, n_trajs=%s)",
        max_steps, total_epochs, steps_per_epoch,
        config.train_dataset.batch_size, config.n_trajs,
    )

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step, epoch=epoch,
            epoch_step=step, steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=1,
                workflow=workflow,
                should_accept_fn=lambda x: (x is not None) and (len(x) > 0),
            )
            perf_tracer.save(step=global_step, force=False)

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        perf_tracer.save(force=True)
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor, step_info, saver, evaluator, stats_logger,
                train_dataloader, tokenizer=tokenizer,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        rollout.resume()

    stats_logger.close()
    proxy_server.close()
    rollout.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
