"""RL training script for the seta-env agent using the AReaL training stack.

Usage:
    python scripts/areal/rl_train.py --config scripts/areal/configs/config_train.yaml
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_REPO_ROOT = Path(__file__).resolve().parents[2]

import torch.distributed as dist

# ── AReaL imports ──────────────────────────────────────────────────────────────
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import cycle_dataloader
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.utils import perf_tracer

# ── Shared workflow ────────────────────────────────────────────────────────────
from workflow import CamelRLVRWorkflow, EvalConfig


logger = logging.getLogger(__name__)


# ── Training-specific config ───────────────────────────────────────────────────

@dataclass
class TrainConfig(EvalConfig):
    """EvalConfig extension with RL-training-specific options."""

    filter_uniform_reward: bool = field(
        default=False,
        metadata={
            "help": (
                "Discard episodes where all non-failed trajectories share the same reward. "
                "Prevents trivial gradient updates on tasks that are always solved or never solved."
            ),
        },
    )
    async_training: bool = field(
        default=True,
        metadata={
            "help": (
                "Use actor.prepare_batch() (overlap rollout with previous weight update) "
                "instead of actor.rollout_batch() (synchronous)."
            ),
        },
    )


# ── Entry point ────────────────────────────────────────────────────────────────

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
    assert parallel_strategy is not None, "allocation_mode must include a train strategy"

    # ── Train engine ───────────────────────────────────────────────────────────
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    perf_tracer.configure(config.perf_tracer, rank=rank)

    # ── Dataset ────────────────────────────────────────────────────────────────
    from datasets import load_dataset as _load_dataset
    dataset_path = Path(config.train_dataset.path)
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    if "parquet" in dataset_path.suffix:
        dataset = _load_dataset(
            path="parquet",
            split="train",
            data_files=[str(dataset_path)],
        )
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

    # ── Inference engine ───────────────────────────────────────────────────────
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    # ── Stop token ids ─────────────────────────────────────────────────────────
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    # ── Terminal env config ────────────────────────────────────────────────────
    te_cfg = config.terminal_env
    te_cfg.model = None  # AReaL builds models externally
    trial_root = os.path.abspath(
        f"{config.stats_logger.fileroot}/{config.stats_logger.experiment_name}"
        f"/{config.stats_logger.trial_name}/trials"
    )
    te_cfg.runtime.trial_root = trial_root

    dump_dir = os.path.join(StatsLogger.get_log_path(config.stats_logger), "generated")
    logger.info("Dump dir:   %s", dump_dir)
    logger.info("Trial root: %s", trial_root)

    # ── Workflow ───────────────────────────────────────────────────────────────
    workflow = CamelRLVRWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        terminal_env_cfg=te_cfg,
        dump_dir=dump_dir,
        n_trajs=config.n_trajs,
        max_tokens=te_cfg.agent.max_total_tokens,
        filter_uniform_reward=config.filter_uniform_reward,
    )

    # ── Training utilities ─────────────────────────────────────────────────────
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch
    data_generator = cycle_dataloader(train_dataloader)

    logger.info(
        "Training: %s steps (%s epoch(s) x %s steps/epoch, batch=%s)",
        max_steps,
        total_epochs,
        steps_per_epoch,
        config.train_dataset.batch_size,
    )

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        logger.debug(
            "\n%s\n[Rank %s] epoch %s  step %s\n%s",
            "=" * 50,
            rank,
            epoch,
            step,
            "=" * 50,
        )

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = actor.prepare_batch(
                    train_dataloader,
                    workflow=workflow,
                    should_accept_fn=lambda x: (x is not None) and (len(x) > 0),
                )
            else:
                batch = actor.rollout_batch(
                    next(data_generator),
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

        # Pause inference for weight sync, checkpoint, and eval.
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
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        rollout.resume()

    stats_logger.close()
    rollout.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
