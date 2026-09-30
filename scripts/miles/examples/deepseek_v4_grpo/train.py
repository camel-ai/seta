"""DeepSeek-V4-Flash-FP8 on Miles: model preparation and GRPO on terminal tasks.

Commands
  prepare  download sgl-project/DeepSeek-V4-Flash-FP8, cast it to BF16 and convert that to
           Megatron torch_dist, under --model-root. Finished steps are skipped.
  train    GRPO. --rollout-backend selects how trajectories are produced:
             harbor       Harbor agent server + Harbor's Terminus-2 agent; sync train.py,
                          actor and SGLang engines colocated on every GPU
             env-service  seta env_service + seta's CAMEL agent; train_async.py with a
                          continuous rollout worker, SGLang on dedicated nodes

  PYTHONPATH=/root/miles python scripts/miles/examples/deepseek_v4_grpo/train.py prepare --num-nodes 8

run_harbor_terminus2.sh and run_env_service.sh set every training knob (see
README.md). The agent settings (AGENT_SERVER_URL, HARBOR_*, CAMEL_*, ...) come from the
environment those scripts export and are forwarded to the Ray workers.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

MODEL_NAME = "DeepSeek-V4-Flash-FP8"  # 43 decoder layers, FP8 weights
MODEL_REPO = f"sgl-project/{MODEL_NAME}"
MEGATRON_MODEL_TYPE = "deepseek-v4-flash"  # Miles scripts/models/deepseek-v4-flash.sh
BLACKWELL = ("B200", "B300", "GB200", "GB300")

# scripts/miles holds the common.* Miles plugins; the seta root holds seta_env.
MILES_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
SETA_ROOT = MILES_SCRIPTS_DIR.parents[1]

# Read inside the Ray rollout workers by common.harbor_agent / common.env_service_agent.
# A Ray job does not inherit the launching shell, so each one is forwarded through the
# job's runtime env; without that the workers fall back to the module defaults.
HARBOR_FORWARDED_ENV = (
    "AGENT_SERVER_URL",
    "AGENT_MODEL_NAME",
    "MILES_ROUTER_EXTERNAL_HOST",
    "MILES_HOST_IP",
    "HARBOR_AGENT_NAME",
    "HARBOR_AGENT_MAX_ITERATIONS",
    "HARBOR_MAX_SEQ_LEN",
    "HARBOR_AGENT_CALL_TIMEOUT_SEC",
)
ENV_SERVICE_FORWARDED_ENV = ("CAMEL_DATASET_NAME", "CAMEL_TRIAL_NAME", "CAMEL_ENV_SERVICE_URL")


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # <model-root>/DeepSeek-V4-Flash-FP8 (HF, FP8) and <model-root>/DeepSeek-V4-Flash-FP8_torch_dist.
    model_root: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    hardware: Literal["auto", "H100", "H200", "B200", "B300", "GB200", "GB300"] = "auto"

    # ---- train
    rollout_backend: Literal["harbor", "env-service"] = "harbor"
    # 0: colocated (every GPU trains and serves); N > 0: N nodes serve SGLang, the rest train.
    rollout_num_nodes: int = 0
    prompt_data: str = ""  # harbor: JSONL with chat prompts; env-service: plain-text prompts
    run_root: str = ""  # checkpoints/ and dump_details/ are written here
    num_rollout: int = 3000
    # The run_*.sh scripts pass the following; there is no backend-neutral default.
    rollout_batch_size: int | None = None  # prompt groups per optimizer step
    n_samples_per_prompt: int | None = None
    temperature: float | None = None
    max_response_len: int | None = None  # tokens per model call
    # Optional; unset = flag not passed to Miles.
    over_sampling_batch_size: int | None = None  # groups started per rollout, surplus aborted
    max_seq_len: int | None = None  # training sample length; longer trajectories are truncated
    max_weight_staleness: int | None = None  # env-service: drop groups this many weight updates old
    lr: str = "1e-6"  # passed to Miles verbatim
    save_interval: int = 50
    skip_saving: bool = False
    dump_details: bool = False
    # Pre-built Miles W&B flags (wandb_args in common/launcher.sh); empty = W&B off.
    wandb_args: str = ""
    # Appended last, so a flag given here overrides the one above.
    extra_args: str = ""

    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)
    rollout_num_gpus: int = field(init=False)

    def __post_init__(self):
        assert 0 <= self.rollout_num_nodes < self.num_nodes
        self.colocate = self.rollout_num_nodes == 0
        self.actor_num_nodes = self.num_nodes - self.rollout_num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node
        self.rollout_num_gpus = (self.num_nodes if self.colocate else self.rollout_num_nodes) * self.num_gpus_per_node

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_root}/{MODEL_NAME}"

    @property
    def torch_dist(self) -> str:
        return f"{self.model_root}/{MODEL_NAME}_torch_dist"


def _is_blackwell(args: ScriptArgs) -> bool:
    if args.hardware != "auto":
        return args.hardware in BLACKWELL

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Cannot auto-detect hardware because CUDA is not available. Pass --hardware explicitly.")
    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


# ---------------------------------------------------------------- prepare


def _prepare(args: ScriptArgs):
    bf16 = f"{args.model_root}/{MODEL_NAME}-bf16"
    U.exec_command(f"mkdir -p {args.model_root}")
    # Idempotent: hf skips blobs that are already downloaded.
    U.exec_command(f"hf download {MODEL_REPO} --local-dir {args.hf_checkpoint}")
    # Megatron converts from BF16; SGLang keeps serving the FP8 checkpoint.
    U.fp8_cast_bf16(path_src=args.hf_checkpoint, path_dst=f"{bf16}/")

    if Path(args.torch_dist, "latest_checkpointed_iteration.txt").exists():
        print(f"[prepare] {args.torch_dist} already exists, skipping the conversion")
        return
    # The one verified conversion layout: 8 nodes x 8 GPUs, TP1 / PP8 / EP4. torch_dist
    # checkpoints reshard on load, so training may use any layout in _parallel_args.
    if (args.num_nodes, args.num_gpus_per_node) != (8, 8):
        raise NotImplementedError("the torch_dist conversion is verified on 8 nodes x 8 GPUs only: pass --num-nodes 8")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        hf_checkpoint=bf16,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=True,
        num_nodes=args.num_nodes,
        extra_args=(
            "--expert-tensor-parallel-size 1 --context-parallel-size 1 "
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
        ),
        dir_dst=args.model_root,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download, cast to BF16 and convert to torch_dist. Run on the Ray head with the cluster up."""
    _prepare(args)


# ---------------------------------------------------------------- train


def _parallel_args(args: ScriptArgs) -> str:
    """Megatron layout for the actor nodes: TP8 x EP8 inside a node, one pipeline stage per node."""
    total_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    if args.actor_num_gpus_per_node != 8 or total_gpus not in (40, 48, 56, 64):
        raise NotImplementedError(
            f"no Megatron layout for {args.actor_num_nodes} actor nodes x "
            f"{args.actor_num_gpus_per_node} GPUs; available: 5-8 actor nodes x 8 GPUs"
        )
    # 43 decoder layers = first(4) + (PP-2) * middle + last, with an integer middle:
    # PP7 needs last=4 (middle 7); PP5, PP6 and PP8 use last=3.
    last = {64: 3, 56: 4, 48: 3, 40: 3}[total_gpus]
    return (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.actor_num_nodes} "
        "--decoder-first-pipeline-num-layers 4 "
        f"--decoder-last-pipeline-num-layers {last} "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
    )


def _harbor_args(args: ScriptArgs) -> tuple[str, str, str, dict[str, str]]:
    """Rollout, SGLang and misc flags, and Ray env, for the Harbor agent-server backend."""
    rollout_args = (
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--balance-data "
        # Rows: {"prompt": <instruction>, "metadata": {"instance_id": ...}}. The agent drives
        # the conversation, so no chat template is applied here; the reward and the exit
        # status come back from the agent server.
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        f"--rollout-max-response-len {args.max_response_len} "
        # drop_thinking=false keeps every turn's reasoning, so Terminus-2's user-role
        # observations stay append-only for DeepSeek-V4 token-in/token-out.
        """--apply-chat-template-kwargs '{"drop_thinking": false}' """
        # Stock agentic generate, plus /flush of the surplus Harbor trials when the batch is
        # full and a zero routing-replay placeholder for empty turns.
        "--custom-generate-function-path common.harbor_abort.generate "
        "--custom-agent-function-path common.harbor_agent.run "
        "--custom-rm-path common.harbor_rollout.reward_func "
        "--rollout-function-path common.harbor_rollout.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model deepseekv4 "
        "--use-session-server "
        "--session-server-port 30000 "
    )
    sglang_args = (
        # 4-GPU engines, TP4 / EP4.
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-dp-size 1 "
        "--sglang-ep-size 4 "
        # DeepSeek-V4 engine-side parsing for the session server.
        "--sglang-tool-call-parser deepseekv4 "
        "--sglang-reasoning-parser deepseek-r1 "
        # SGLang router (not the Miles router); it always starts, so it needs a port.
        "--sglang-router-port 31000 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )
    misc_args = (
        "--sglang-mem-fraction-static 0.7 "
        # Skip (don't crash on) NaN/Inf grads: disables Megatron's fatal in-backward check
        # and lets Miles skip the optimizer step instead.
        "--no-check-for-nan-in-loss-and-grad "
    )
    env = {
        "PYTHONPATH": f"{args.megatron_path}:{MILES_SCRIPTS_DIR}:{U.repo_base_dir}",
        **{k: os.environ[k] for k in HARBOR_FORWARDED_ENV if k in os.environ},
    }
    return rollout_args, sglang_args, misc_args, env


def _env_service_args(args: ScriptArgs) -> tuple[str, str, str, dict[str, str]]:
    """Rollout, SGLang and misc flags, and Ray env, for the seta env_service backend."""
    rollout_args = (
        "--label-key label "
        "--balance-data "
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        f"--rollout-max-response-len {args.max_response_len} "
        # No --apply-chat-template: the session server owns the chat encoding and the agent
        # sends the raw instruction to env_service /step.
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path common.env_service_agent.run "
        "--use-session-server "
        "--tito-model deepseekv4 "
        "--tito-allowed-append-roles tool "  # the CAMEL agent only appends tool results
        "--session-server-port 30002 "
        "--custom-rm-path common.env_service_reward.reward_func "
        "--custom-rollout-log-function-path common.env_service_metrics.log_rollout_data "
        # Continuous rollout worker: keeps up to ROLLOUT_CONCURRENCY groups in flight; each
        # train step takes rollout_batch_size finished groups.
        "--rollout-function-path common.env_service_rollout.generate_rollout_fully_async "
        "--update-weights-interval 1 "
        "--pause-generation-mode in_place "
        "--skip-eval-before-train "
    )
    tp = args.num_gpus_per_node
    sglang_args = (
        # One engine per rollout node: TP8 / EP8.
        f"--rollout-num-gpus-per-engine {tp} "
        f"--sglang-tp-size {tp} "
        "--sglang-dp-size 1 "
        "--sglang-attention-backend compressed "
        # The session server pins each session to one engine; the router runs its default policy.
        "--sglang-page-size 256 "
        "--sglang-max-running-requests 96 "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-server-concurrency 1024 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
        f"--sglang-ep-size {tp} "
        # Engine-side parsers so the session server stores structured tool calls and
        # separated reasoning: deepseek-r1 for DeepSeek-V4's implicit-open thinking,
        # deepseekv4 for its DSML tool-call format.
        "--sglang-reasoning-parser deepseek-r1 "
        "--sglang-tool-call-parser deepseekv4 "
    )
    misc_args = "--sglang-mem-fraction-static 0.84 "
    env = {
        "PYTHONPATH": (
            f"{args.megatron_path}:{MILES_SCRIPTS_DIR}:{SETA_ROOT}:"
            f"{U.repo_base_dir / 'examples/fully_async'}:{U.repo_base_dir}"
        ),
        # Groups in flight for the continuous rollout worker, independent of the train batch.
        "ROLLOUT_CONCURRENCY": os.environ.get("ROLLOUT_CONCURRENCY", "12"),
        **{k: os.environ[k] for k in ENV_SERVICE_FORWARDED_ENV if os.environ.get(k)},
    }
    return rollout_args, sglang_args, misc_args, env


def _train(args: ScriptArgs):
    for name in ("prompt_data", "run_root", "rollout_batch_size", "n_samples_per_prompt", "temperature", "max_response_len"):
        if getattr(args, name) in (None, ""):
            raise ValueError(f"--{name.replace('_', '-')} is required")
    print(
        f"[train] {args.rollout_backend}: {args.num_nodes} nodes ({args.actor_num_nodes} actor nodes x "
        f"{args.actor_num_gpus_per_node} GPUs, {args.rollout_num_gpus} rollout GPUs, colocate={args.colocate})"
    )

    ckpt_dir = f"{args.run_root}/checkpoints"  # --load == --save: rerunning a RUN_NAME resumes it
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.torch_dist} "
    if not args.skip_saving:
        ckpt_args += (
            f"--load {ckpt_dir} --save {ckpt_dir} "
            f"--save-interval {args.save_interval} --save-retain-interval {args.save_interval} "
            # weights only: don't persist optimizer state (and don't try to load it on resume)
            "--no-save-optim --no-load-optim "
        )

    rollout_args = (
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-temperature {args.temperature} "
        "--num-steps-per-rollout 1 "
    )
    if args.over_sampling_batch_size is not None:
        rollout_args += f"--over-sampling-batch-size {args.over_sampling_batch_size} "
    if args.max_seq_len is not None:
        # Training sequence budget (Miles side only). truncate_samples_by_total_tokens drops
        # everything past it, so a small cap discards most of a long trajectory.
        rollout_args += f"--max-seq-len {args.max_seq_len} "
    if args.max_weight_staleness is not None:
        rollout_args += f"--max-weight-staleness {args.max_weight_staleness} "

    backend_args = _harbor_args if args.rollout_backend == "harbor" else _env_service_args
    backend_rollout_args, sglang_args, backend_misc_args, env = backend_args(args)
    rollout_args += backend_rollout_args
    if args.rollout_backend == "harbor":
        rollout_args += "--update-weights-interval 1 "

    perf_args = _parallel_args(args) + (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "  # log KL to the base model (coef 0: not added to the loss; ref = --ref-load)
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 1e-8 "  # ~0: Miles only computes/logs entropy when the coef is non-zero
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--use-precision-aware-optimizer "
        "--overlap-cpu-optimizer-d2h-h2d "
    )

    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
        "SGLANG_OPT_FP8_WO_A_GEMM": "0",
        # Deterministic training kernels (with --deterministic-mode below).
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        # Needed for the GenerateFnInput signature and the generate function's add_arguments hook.
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        **env,
    }

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        f"{backend_misc_args}"
        "--accumulate-allreduce-grads-in-fp32 "
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        "--use-fault-tolerance "
        # R3: replay the rollout's MoE routing in the training forward pass.
        "--use-rollout-routing-replay "
        "--deterministic-mode "
        # Blockwise FP8 (128x128) training GEMMs; rollout serves the FP8 checkpoint either way.
        "--transformer-impl transformer_engine --bf16 --fp8-format e4m3 --fp8-recipe blockwise "
    )
    misc_args += "--colocate " if args.colocate else f"--rollout-num-gpus {args.rollout_num_gpus} "
    if args.dump_details:
        misc_args += f"--dump-details {args.run_root}/dump_details "
    # On Blackwell, TE emulates the blockwise recipe with MXFP8, which requires pow2 scales.
    fp32_scales = "0" if _is_blackwell(args) else "1"
    misc_args += f"""--train-env-vars '{{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"{fp32_scales}"}}' """

    # W&B flags (incl. --wandb-key) come from common/launcher.sh wandb_args.
    wandb_args = args.wandb_args

    train_args = " ".join(
        (ckpt_args, rollout_args, optimizer_args, grpo_args, wandb_args, perf_args, sglang_args, misc_args, args.extra_args)
    )
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        # harbor: roll out a full batch, train on it, repeat. env-service: rollout and
        # training overlap, the rollout worker runs ahead of the trainer.
        train_script="train.py" if args.rollout_backend == "harbor" else "train_async.py",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Submit the training job. Run prepare first."""
    _train(args)


if __name__ == "__main__":
    app()
