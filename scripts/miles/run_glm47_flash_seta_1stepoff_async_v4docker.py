"""
GLM-4.7-Flash SETA 1-step-off async trainer (DeepSeek-V4 docker image).

Adapted from run_glm47_flash_aime.py (the GLM-4.7-Flash aime example) for the
seta_env camel terminal-agent task, mirroring the seta wiring of
run_deepseek_v4_seta_1stepoff_async_v4docker.py (binary reward, 1-step-off async
via train_async.py + standard single-shot rollout, group_reward_filter, env_service
generate/rm/log hooks). GLM model handling (arch args sourced from
scripts/models/glm4.7-flash.sh by execute_train, tp4 because 20 attn heads, ep8,
miles-router + rollout-routing-replay) is inherited from the aime example. EAGLE/MTP
speculative decoding from the aime base is intentionally OFF (it inflates the
train_rollout_logprob_abs_diff metric this ablation studies). Runs on the new miles
docker (transformers 5.8.1 native GLM, arguments.py moe_ffn fix upstream) — none of
the old rolled-back-image SGLang workarounds are needed.

Cluster: 8 nodes x 8 GPUs (H200). DISAGGREGATED: 4 rollout (serving) + 4 actor
(training) nodes (rollout_num_nodes=4). Launched via the companion .sh which
restarts seta_env env_service and submits a detached ray job.

Ablation vs DeepSeek-V4: SAME dataset (951 curated mid-band pass@8 1-6/8) and SAME
RL config (lr 1e-6, grpo, kl-coef 0 logged, eps-clip 0.2/0.28), different base model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-4.7-Flash"
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    hardware: Literal["H200"] = "H200"
    enable_eval: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/data/models"        # GLM lands in /data/models (was /root/models)
    save_dir: str = "/data/training_runs"  # RUN_ROOT passed via --save-dir from the .sh
    megatron_path: str = "/root/Megatron-LM"

    # DISAGGREGATED 8-node split: 4 rollout (serving) + (num_nodes-4) actor (training)
    rollout_num_nodes: int = 4

    # seta_env (camel terminal agent) — SAME dataset as the DeepSeek-V4 run
    seta_env_parquet_path: str = "/data/terminal_agent/dataset/seta-env-mid1to6of8-qc4plus-951.parquet"
    seta_env_max_response_len: int = 8192
    group_filter_min_reward_std: float = 1e-8  # drop ONLY zero-std (all-same-reward) groups
    dump_details: bool = True


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download GLM-4.7-Flash to /data/models + convert to torch_dist (run separately first)."""
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    # New docker ships transformers 5.8.1 with native glm4_moe_lite support — no
    # custom transformers pin needed (the old rolled-back image required one).
    U.exec_command(
        f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """1-step-off async seta training on 8 nodes (4 serve + 4 train)."""
    actor_num_nodes = args.num_nodes - args.rollout_num_nodes
    rollout_num_gpus = args.rollout_num_nodes * args.num_gpus_per_node
    load_save_path = f"{args.save_dir}/checkpoints"  # unified per-run layout (save_dir == RUN_ROOT)

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 50 "
        "--save-retain-interval 50 "   # retain==interval -> keep ALL checkpoints
        "--no-save-optim "
        "--no-load-optim "
    )

    # seta_env rollout: env_service applies the GLM chat template itself, so NO
    # --apply-chat-template / --rm-type here. 1-step-off = standard single-shot
    # rollout + structural staleness=1 via train_async.py pipeline.
    rollout_args = (
        "--label-key label "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 16 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {args.seta_env_parquet_path} "
        "--input-key prompt "
        f"--rollout-max-response-len {args.seta_env_max_response_len} "
        "--custom-generate-function-path core.generate_with_camel.generate "
        "--custom-rm-path core.reward_func.reward_func "
        "--custom-rollout-log-function-path core.camel_rollout_metrics.log_rollout_data "
        "--dynamic-sampling-filter-path core.group_reward_filter.filter_group "
        "--rollout-function-path miles.rollout.sglang_rollout.generate_rollout "
        "--update-weights-interval 1 "
    )

    eval_args = "--skip-eval-before-train "

    perf_args = (
        # tp=4 because GLM-4.7-Flash has 20 attention heads (tp must divide num_heads)
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "       # tp4 engines (GLM-4.7-Flash has 20 attn heads)
        "--sglang-mem-fraction-static 0.7 "      # aime-base value; MLA (kv_lora 512) keeps KV small for 60k seqs
        # EAGLE/MTP speculative decoding intentionally OFF (the aime base enables it):
        # MTP inflates train_rollout_logprob_abs_diff — the exact metric this ablation
        # studies — so we drop it to keep rollout/train logprobs aligned (also not part
        # of the training-dynamics comparison).
        # miles router for session-sticky routing (prefix-cache locality), as in the aime base.
        # R3 (--use-rollout-routing-replay) is intentionally OFF for GLM: routed_experts capture
        # is implemented ONLY in DeepSeekV4SGLangModel. GLM uses the generic SGLangModel (correct
        # GLM chat-template + tool parsing) which cannot emit routed_experts. With R3 ON, env_service
        # threads return_routed_experts into model_config_dict, the generic model forwards it as an
        # SGLang sampling param -> SamplingParams.__init__() TypeError -> every /generate returns 500.
        # Without R3 the run is fully valid; the train/rollout MoE routing mismatch simply isn't
        # replayed (it surfaces in train_rollout_logprob_abs_diff, the metric this ablation studies).
        "--use-miles-router "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        # DISAGGREGATED: actor on (num_nodes - rollout_num_nodes) nodes, rollout on the rest
        f"--actor-num-nodes {actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {rollout_num_gpus} "
        "--use-fault-tolerance "
    )
    if args.dump_details:
        misc_args += f"--dump-details {args.save_dir}/dump_details "

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    if wandb_args:
        wandb_args += f"--wandb-dir {args.save_dir}/wandb "
        wandb_args += "--wandb-team eigent_radixark_training "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    # seta_env wiring: generate_with_camel + group_reward_filter need these on the
    # ray-worker PYTHONPATH/env (mirrors run_deepseek_v4_seta_1stepoff_async_v4docker.py).
    import os

    extra_env_vars = {
        "PYTHONPATH": (
            f"{args.megatron_path}:{Path(__file__).resolve().parent}:{Path(__file__).resolve().parents[2]}:"
            f"{U.repo_base_dir / 'examples/fully_async'}:{U.repo_base_dir}"
        ),
        "GROUP_FILTER_MIN_REWARD_STD": str(args.group_filter_min_reward_std),
        "GROUP_FILTER_MAX_ENV_FAILURES": "1",  # STRICT: drop any group with >=1 env failure
        "ROLLOUT_CONCURRENCY": os.environ.get("ROLLOUT_CONCURRENCY", "12"),
        # No SGLANG_* overrides: the new docker serves GLM-4.7-Flash natively (the aime
        # base sets none). The old rolled-back image needed SGLANG_APPLY_CONFIG_BACKUP=none
        # and SGLANG_DSV4_2604_SUBMODE="" to dodge DeepSeek-V4-only code paths; reverted.
    }
    for _k in ("CAMEL_DATASET_NAME", "CAMEL_TRIAL_NAME", "CAMEL_ENV_SERVICE_URL", "MILES_PIN_NODE_IPS"):
        if os.environ.get(_k):
            extra_env_vars[_k] = os.environ[_k]

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


if __name__ == "__main__":
    app()
