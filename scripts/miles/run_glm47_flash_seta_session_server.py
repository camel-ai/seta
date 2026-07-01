"""GLM-4.7-Flash SETA trainer via the miles SESSION SERVER (not our custom backend).

WHY this exists
---------------
Our previous GLM run used a custom env_service backend (sglang_glm47_flash + the generic
SGLangModel) that hand-rolls TITO by re-applying the chat template per turn. That path had a
multi-turn formatting bug for PARALLEL tool calls: the tool results came out as empty
``<|observation|>`` markers with no ``<|assistant|>`` generation prompt, so the model emitted a
single stop token (completion_tokens=1) and the agent terminated prematurely — deflating the
solve rate.

This launcher instead uses the miles SESSION SERVER, exactly like
``run_deepseek_v4_seta_session_server.py`` but for GLM:
  * SGLang parses GLM tool calls + reasoning ENGINE-SIDE
    (``--sglang-tool-call-parser glm47`` = Glm47MoeDetector, ``--sglang-reasoning-parser glm45``).
  * The miles session server captures the EXACT TITO tokens per session
    (``--use-session-server --tito-model glm47 --tito-allowed-append-roles user tool``),
    with ``--tito-validate`` semantics owned by the server — no manual reconstruction, so the
    parallel-tool / empty-observation / content-markup bug class cannot occur.
  * The agent runs in our seta_env env_service via ``core.seta_agent_function.run`` +
    ``miles.rollout.generate_hub.agentic_tool_call.generate`` (env_service /step client).

References combined:
  * seta wiring:  scripts/miles/run_deepseek_v4_seta_session_server.py
  * GLM session-server params:
        /root/miles/examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py

Cluster: 8 nodes x 8 GPUs (H200). DISAGGREGATED: 4 rollout (serving) + 4 actor (training).
Rollout is fully-async (the proven session-server combo): a continuous worker overproduces up
to ROLLOUT_CONCURRENCY groups; each train step drains rollout_batch_size valid groups
(staleness-capped). SAME dataset (951 mid-band) + RL config (lr 1e-6, grpo, kl-coef 0 logged,
eps-clip 0.2/0.28) as the DeepSeek run; only the base model + the session-server parsers change.
R3 (routed-experts replay) stays OFF for GLM (the GLM example doesn't use it; routed_experts
capture isn't supported for the generic GLM serving path).
"""

import os
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
    model_dir: str = "/data/models"
    save_dir: str = "/data/training_runs"     # RUN_ROOT via --save-dir from the .sh
    megatron_path: str = "/root/Megatron-LM"

    # DISAGGREGATED 8-node split: 4 rollout (serving) + (num_nodes-4) actor (training)
    rollout_num_nodes: int = 7  # 8 nodes = 7 serve + 1 train. Rollout is the throughput ceiling
    # (Daytona-bound), so minimize training to 1 node (8 GPU, tp4/ep8 -> EDP1) and push GPUs to serving
    # (7 nodes = 14 tp4 engines). Train rate ~0.5 samples/s on 8 GPU; sized down toward the rollout rate.

    # session-server / TITO (GLM-specific, from the swe-agent-v2 GLM example)
    tito_model: str = "glm47"
    sglang_tool_call_parser: str = "glm47"
    sglang_reasoning_parser: str = "glm45"
    session_server_port: int = 30000
    sglang_router_port: int = 31000

    # seta_env (camel terminal agent) — SAME dataset as the DeepSeek-V4 run
    seta_env_parquet_path: str = "/data/terminal_agent/dataset/tbench-tasks-migrated.parquet"  # 241 terminal-bench migrated tasks
    seta_env_max_response_len: int = 8192
    group_filter_min_reward_std: float = 1e-8   # drop ONLY zero-std (all-same-reward) groups
    dump_details: bool = False  # OFF: the --dump-details torch.save (train_data/*.pt) crashed the
    # run (basic_ios::clear / unexpected pos) on the 99%-full /data FS and isn't needed for training.

    # fully-async continuous-worker knobs
    rollout_concurrency: int = 30  # groups in flight (30x16=480 trajectories; MAX_SLOTS 400 caps
    # active sandboxes at 400, the extra ~80 queue for slots -> keeps all 400 slots saturated)
    max_weight_staleness: int = 4


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download GLM-4.7-Flash + convert to torch_dist (run separately first)."""
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    # New docker ships transformers 5.8.1 with native glm4_moe_lite support — no pin needed.
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
    """Session-server fully-async seta training on 8 nodes (4 serve + 4 train)."""
    actor_num_nodes = args.num_nodes - args.rollout_num_nodes
    rollout_num_gpus = args.rollout_num_nodes * args.num_gpus_per_node
    load_save_path = f"{args.save_dir}/checkpoints"

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

    # seta_env rollout via the SESSION SERVER. --apply-chat-template is NOT set: SGLang owns
    # the GLM chat template (engine-side), and seta_agent_function passes the raw instruction
    # to env_service /step.
    rollout_args = (
        "--label-key label "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 16 "        # TRAIN batch: 16 groups x 16 = 256 samples/step
        "--n-samples-per-prompt 16 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {args.seta_env_parquet_path} "
        "--input-key prompt "
        f"--rollout-max-response-len {args.seta_env_max_response_len} "
        # ── session server + agentic generate + seta env_service client ──
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path core.seta_agent_function.run "
        "--use-session-server "
        f"--tito-model {args.tito_model} "
        "--tito-allowed-append-roles user tool "
        f"--session-server-port {args.session_server_port} "
        "--custom-rm-path core.reward_func.reward_func "
        "--custom-rollout-log-function-path core.camel_rollout_metrics.log_rollout_data "
        "--dynamic-sampling-filter-path core.group_reward_filter.filter_group "
        # fully-async continuous worker (proven session-server combo): overproduce up to
        # ROLLOUT_CONCURRENCY groups; train step drains rollout_batch_size valid groups.
        "--rollout-function-path core.fully_async_rollout_seta.generate_rollout_fully_async "
        f"--max-weight-staleness {args.max_weight_staleness} "
        "--update-weights-interval 1 "
        "--pause-generation-mode in_place "
        # NOTE: --over-sampling-batch-size is a NO-OP on this fully-async path — it is only read by
        # the non-async miles.rollout.sglang_rollout / inference_rollout_train (data_source(...)).
        # Our fully_async_rollout_seta worker keeps ROLLOUT_CONCURRENCY *groups* in flight
        # (fully_async_rollout_seta.py:103) and the train step drains rollout_batch_size valid groups.
        # So overproduce-and-take-fastest is controlled by ROLLOUT_CONCURRENCY (25) > rollout_batch_size (16).
        # (omitted; miles defaults over_sampling_batch_size to rollout_batch_size, satisfying its assert.)
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

    # SGLang serving for the session server: tp4 engines, GLM engine-side parsers, native SGLang router.
    # NOTE: --use-miles-router removed -> the session server proxies to SGLang's NATIVE router
    # (router_manager.py:43-44 run_sglang_router); --sglang-router-port + --router-health-* still apply.
    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "       # tp4 engines (GLM has 20 attn heads); 6 serve nodes = 12 engines
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-tool-call-parser {args.sglang_tool_call_parser} "  # glm47 = Glm47MoeDetector
        f"--sglang-reasoning-parser {args.sglang_reasoning_parser} "  # glm45 reasoning (<think>)
        f"--sglang-router-port {args.sglang_router_port} "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
        # R3 OFF for GLM (no --use-rollout-routing-replay): routed_experts capture isn't
        # supported for the generic GLM serving path (SamplingParams rejects it).
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--grad-reduce-in-bf16 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        # DISAGGREGATED: actor on (num_nodes - rollout_num_nodes) nodes, rollout on the rest
        f"--actor-num-nodes {actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {rollout_num_gpus} "
        "--use-fault-tolerance "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
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

    extra_env_vars = {
        "PYTHONPATH": (
            f"{args.megatron_path}:{Path(__file__).resolve().parent}:{Path(__file__).resolve().parents[2]}:"
            f"{U.repo_base_dir / 'examples/fully_async'}:{U.repo_base_dir}"
        ),
        # Required so arguments.py registers the agentic_tool_call.generate CLI args
        # (--custom-agent-function-path / --use-session-server / --tito-* etc.) and routes the
        # custom generate fn through the GenerateFnInput signature. Mirrors swe-agent-v2 / the
        # DeepSeek seta session-server. Also makes the session server own router affinity.
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "GROUP_FILTER_MIN_REWARD_STD": str(args.group_filter_min_reward_std),
        "GROUP_FILTER_MAX_ENV_FAILURES": "1",  # STRICT: drop any group with >=1 env failure
        "ROLLOUT_CONCURRENCY": os.environ.get("ROLLOUT_CONCURRENCY", str(args.rollout_concurrency)),
        # No SGLANG_* overrides: new docker serves GLM natively (see run_glm47_flash_aime.py).
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
