"""Evaluation script for the seta-env agent using the AReaL inference stack.

Usage:
    python scripts/areal/eval.py --config scripts/areal/configs/config_eval.yaml
"""

import logging
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# ── AReaL imports ──────────────────────────────────────────────────────────────
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.utils import perf_tracer

# ── Shared workflow ────────────────────────────────────────────────────────────
from workflow import CamelRLVRWorkflow, EvalConfig


logger = logging.getLogger(__name__)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config, _ = load_expr_config(args, EvalConfig)
    config: EvalConfig

    rank = int(os.getenv("RANK") or "0")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    dump_dir = os.path.abspath(
        f"{config.stats_logger.fileroot}/{config.stats_logger.experiment_name}"
        f"/{config.stats_logger.trial_name}/logs/generated"
    )
    trial_root = os.path.abspath(
        f"{config.stats_logger.fileroot}/{config.stats_logger.experiment_name}"
        f"/{config.stats_logger.trial_name}/trials"
    )
    logger.info("Dump dir:   %s", dump_dir)
    logger.info("Trial root: %s", trial_root)

    seeding.set_random_seed(config.seed, key="eval")
    AllocationMode.from_str(config.allocation_mode)

    if config.perf_tracer is not None:
        perf_tracer.configure(config.perf_tracer, rank=rank)

    # ── Terminal env config ────────────────────────────────────────────────────
    te_cfg = config.terminal_env
    te_cfg.model = None  # AReaL builds models externally
    te_cfg.runtime.trial_root = trial_root

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

    eval_dataloader = create_dataloader(
        dataset,
        rank=rank,
        world_size=1,
        dataset_config=config.train_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(eval_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    stats_logger = StatsLogger(config, ft_spec)

    # ── Inference engine (no weight updates) ───────────────────────────────────
    rollout_engine = RemoteSGLangEngine(config.rollout)
    rollout_engine.initialize()

    # ── Workflow ───────────────────────────────────────────────────────────────
    workflow = CamelRLVRWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        terminal_env_cfg=te_cfg,
        dump_dir=dump_dir,
        n_trajs=config.n_trajs,
        max_tokens=te_cfg.agent.max_total_tokens,
    )

    # ── Eval loop ──────────────────────────────────────────────────────────────
    total_epochs = config.total_train_epochs
    steps_per_epoch = len(eval_dataloader)
    max_steps = total_epochs * steps_per_epoch

    logger.info(
        "Eval: %s steps (%s epoch(s) x %s steps/epoch, batch=%s)",
        max_steps,
        total_epochs,
        steps_per_epoch,
        config.train_dataset.batch_size,
    )

    for global_step in range(max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        start_time = time.time()

        logger.debug(
            "\n%s\n[Rank %s] epoch %s  step %s\n%s",
            "=" * 50,
            rank,
            epoch,
            step,
            "=" * 50,
        )

        with stats_tracker.record_timing("rollout"):
            _batch = rollout_engine.prepare_batch(eval_dataloader, workflow=workflow)

        rollout_engine.set_version(global_step + 1)
        perf_tracer.save(step=global_step)

        elapsed = time.time() - start_time
        logger.info(
            "[Rank %s] epoch %s step %s done in %.2f min",
            rank,
            epoch,
            step,
            elapsed / 60,
        )

        stats = stats_tracker.export_all()
        stats_logger.commit(epoch, step, global_step, stats)

    stats_logger.close()
    rollout_engine.destroy()
    perf_tracer.save(force=True)


if __name__ == "__main__":
    main(sys.argv[1:])
