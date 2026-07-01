"""Evaluation script using remote env_service nodes.

Same structure as eval.py but uses EnvServiceRLVRWorkflow:
- ProxyServer captures model interactions
- Agents run on remote env_service nodes via env_scheduler
- No training, just inference + eval metrics

Usage:
    python scripts/areal/eval_env_service.py --config scripts/areal/configs/config_eval.yaml
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.proxy import ProxyServer
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.network import find_free_ports
from areal.utils.stats_logger import StatsLogger
from areal.utils import perf_tracer

from workflow_env_service import EnvServiceRLVRWorkflow

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class EvalEnvServiceConfig(GRPOConfig):
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

    # Eval
    n_trajs: int = field(
        default=1,
        metadata={"help": "Number of parallel trajectories per task."},
    )
    step_timeout: float = field(
        default=900.0,
        metadata={"help": "Timeout (seconds) for each env_service step request."},
    )
    export_style: str = field(
        default="individual",
        metadata={"help": "Export style for completions."},
    )
    tool_call_parser: str = field(
        default="qwen25",
        metadata={"help": "Tool call parser for ArealOpenAI client."},
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config, _ = load_expr_config(args, EvalEnvServiceConfig)
    config: EvalEnvServiceConfig

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

    # ── Dataset ───────────────────────────────────────────────────────────
    from datasets import load_dataset as _load_dataset
    dataset_path = Path(config.train_dataset.path)
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    if "parquet" in dataset_path.suffix:
        dataset = _load_dataset(
            path="parquet", split="train", data_files=[str(dataset_path)],
        )
    elif dataset_path.is_dir():
        from seta_env.dataset import load_harbor_dataset
        dataset = load_harbor_dataset(dataset_path)
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    eval_dataloader = create_dataloader(
        dataset, rank=rank, world_size=1, dataset_config=config.train_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(eval_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    stats_logger_inst = StatsLogger(config, ft_spec)

    # ── Inference engine ──────────────────────────────────────────────────
    rollout_engine = RemoteSGLangEngine(config.rollout)
    rollout_engine.initialize()

    # ── ProxyServer ───────────────────────────────────────────────────────
    client = ArealOpenAI(
        engine=rollout_engine,
        tokenizer=tokenizer,
        tool_call_parser=config.tool_call_parser,
        chat_template_type="concat" if config.export_style == "concat" else "hf",
    )
    free_port = find_free_ports(1)[0]
    proxy_server = ProxyServer(port=free_port, client=client)
    proxy_server.start(wait_until_ready=True)
    logger.info("ProxyServer at %s", proxy_server.public_addr)

    # ── Auto-setup FRP tunnel + url_rewrite ──────────────────────────────
    from seta_env.services.proxy_setup import setup_proxy_tunnels
    setup_proxy_tunnels(
        proxy_addresses=[proxy_server.public_addr],
        scheduler_url=config.env_scheduler_url,
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
        export_style=config.export_style,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────
    total_epochs = config.total_train_epochs
    steps_per_epoch = len(eval_dataloader)
    max_steps = total_epochs * steps_per_epoch

    logger.info(
        "Eval: %s steps (%s epoch(s) x %s steps/epoch, batch=%s, n_trajs=%s)",
        max_steps, total_epochs, steps_per_epoch,
        config.train_dataset.batch_size, config.n_trajs,
    )

    for global_step in range(max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        start_time = time.time()

        with stats_tracker.record_timing("rollout"):
            _batch = rollout_engine.prepare_batch(eval_dataloader, workflow=workflow)

        rollout_engine.set_version(global_step + 1)
        perf_tracer.save(step=global_step)

        elapsed = time.time() - start_time
        logger.info(
            "[Rank %s] epoch %s step %s done in %.2f min",
            rank, epoch, step, elapsed / 60,
        )

        stats = stats_tracker.export_all()
        stats_logger_inst.commit(epoch, step, global_step, stats)

    stats_logger_inst.close()
    proxy_server.close()
    rollout_engine.destroy()
    perf_tracer.save(force=True)


if __name__ == "__main__":
    main(sys.argv[1:])
