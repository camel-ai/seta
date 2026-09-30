"""GLM-4.7-Flash on Miles: model preparation and GRPO on terminal tasks via the seta env_service.

Commands
  prepare  download zai-org/GLM-4.7-Flash to <model-root>/GLM-4.7-Flash and convert it to
           Megatron torch_dist at <model-root>/GLM-4.7-Flash_torch_dist
  train    full-parameter GRPO. Rollouts go through the seta env_service: for every
           trajectory Miles opens a session on its session server, common.env_service_agent
           posts the task to the env_service, and seta's CAMEL agent works on it in a Daytona
           sandbox against the session URL. The session server records the exact tokens of
           every model call (token-in/token-out), so SGLang parses GLM's tool calls and
           reasoning engine-side and nothing is re-tokenized on the client.

  PYTHONPATH=/root/miles python scripts/miles/examples/glm47_flash_grpo/train.py prepare
  bash scripts/miles/examples/glm47_flash_grpo/run_env_service.sh    # train

Placement (defaults): 8 nodes x 8 GPUs, disaggregated. rollout_num_nodes = 7 nodes serve
14 SGLang engines (TP4); the remaining node trains (TP4 x EP8). Rollout (sandbox and tool
time) bounds throughput, so most GPUs serve. Miles train_async.py runs rollout and training
concurrently: a continuous worker (common.env_service_rollout) keeps ROLLOUT_CONCURRENCY
prompt groups in flight and each step trains on the next rollout_batch_size finished groups.
"""

from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer(add_completion=False)

MODEL_ORG = "zai-org"
MODEL_NAME = "GLM-4.7-Flash"
MEGATRON_MODEL_TYPE = "glm4.7-flash"  # Miles scripts/models/glm4.7-flash.sh

# Session server and engine-side parsing for GLM-4.7: the glm47 tool-call parser and the
# glm45 reasoning parser (<think> blocks). The CAMEL agent appends user and tool turns.
TITO_MODEL = "glm47"
SGLANG_TOOL_CALL_PARSER = "glm47"
SGLANG_REASONING_PARSER = "glm45"
SESSION_SERVER_PORT = 30000
SGLANG_ROUTER_PORT = 31000
# GLM-4.7-Flash has 20 attention heads; tensor parallelism must divide them.
TENSOR_PARALLEL_SIZE = 4
EXPERT_PARALLEL_SIZE = 8

# scripts/miles holds the common.* Miles plugins.
MILES_SCRIPTS_DIR = Path(__file__).resolve().parents[2]


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # <model-root>/GLM-4.7-Flash (HF) and <model-root>/GLM-4.7-Flash_torch_dist.
    model_root: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    num_nodes: int = 8
    num_gpus_per_node: int = 8
    rollout_num_nodes: int = 7

    # ---- train
    prompt_data: str = ""
    run_root: str = ""  # checkpoints/ and dump_details/ are written here
    num_rollout: int = 3000
    rollout_batch_size: int = 16  # prompt groups per optimizer step (16 x 16 = 256 samples)
    n_samples_per_prompt: int = 16
    # Prompt groups the continuous worker keeps in flight (ROLLOUT_CONCURRENCY).
    rollout_concurrency: int = 30
    max_weight_staleness: int = 4
    lr: str = "1e-6"  # passed to Miles verbatim
    save_interval: int = 50
    # Group filter (common.env_service_filters): drop groups whose rewards are all equal
    # (zero advantage) and groups with any env-failed sample.
    group_filter_min_reward_std: float = 1e-8
    group_filter_max_env_failures: int = 1
    dump_details: bool = False  # per-step rollout/train tensors under run_root/dump_details
    # env_service wiring, forwarded to the Ray rollout workers (see common/env_service.sh).
    env_service_url: str = ""
    dataset_name: str = ""
    trial_name: str = ""
    wandb_args: str = ""  # Miles W&B flags from the launcher (launcher.sh wandb_args)
    extra_args: str = ""  # appended last


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download GLM-4.7-Flash and convert it to torch_dist (skipped when already converted)."""
    hf_dir = f"{args.model_root}/{MODEL_NAME}"
    U.exec_command(f"mkdir -p {args.model_root}")
    U.exec_command(f"hf download {MODEL_ORG}/{MODEL_NAME} --local-dir {hf_dir}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_root,
        hf_checkpoint=hf_dir,
        megatron_path=args.megatron_path,
    )


def _validate(args: ScriptArgs) -> None:
    for name in ("prompt_data", "run_root", "env_service_url", "dataset_name", "trial_name"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} is required")
    if not 0 < args.rollout_num_nodes < args.num_nodes:
        raise ValueError("need at least one training node and one rollout node")
    train_gpus = (args.num_nodes - args.rollout_num_nodes) * args.num_gpus_per_node
    if train_gpus % EXPERT_PARALLEL_SIZE:
        raise ValueError(f"{train_gpus} training GPUs are not a multiple of the expert parallel size {EXPERT_PARALLEL_SIZE}")
    if (args.rollout_num_nodes * args.num_gpus_per_node) % TENSOR_PARALLEL_SIZE:
        raise ValueError(f"rollout GPUs must be a multiple of the engine size {TENSOR_PARALLEL_SIZE}")


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """GRPO with rollouts through the seta env_service (see the module docstring)."""
    _validate(args)
    actor_num_nodes = args.num_nodes - args.rollout_num_nodes
    rollout_num_gpus = args.rollout_num_nodes * args.num_gpus_per_node
    load_save_path = f"{args.run_root}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_root}/{MODEL_NAME} "
        f"--ref-load {args.model_root}/{MODEL_NAME}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {args.save_interval} "
        # retain interval = save interval: every checkpoint is kept.
        f"--save-retain-interval {args.save_interval} "
        "--no-save-optim "
        "--no-load-optim "
    )

    # No --apply-chat-template: SGLang applies the GLM chat template engine-side and the
    # agent sends the raw instruction to the env_service. The per-call sampling settings
    # (temperature, top_p, max_tokens) the agent actually uses are in env_service.yaml;
    # --rollout-temperature and --rollout-max-response-len are Miles' own defaults for
    # requests it makes itself.
    rollout_args = (
        "--label-key label "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--rollout-max-response-len 8192 "
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path common.env_service_agent.run "
        "--use-session-server "
        f"--tito-model {TITO_MODEL} "
        "--tito-allowed-append-roles user tool "
        f"--session-server-port {SESSION_SERVER_PORT} "
        "--custom-rm-path common.env_service_reward.reward_func "
        "--custom-rollout-log-function-path common.env_service_metrics.log_rollout_data "
        "--dynamic-sampling-filter-path common.env_service_filters.filter_group "
        # Continuous rollout worker: ROLLOUT_CONCURRENCY groups in flight, each step takes
        # rollout_batch_size finished groups. --over-sampling-batch-size has no effect on
        # this path, so it is not passed.
        "--rollout-function-path common.env_service_rollout.generate_rollout_fully_async "
        f"--max-weight-staleness {args.max_weight_staleness} "
        "--update-weights-interval 1 "
        # In-flight requests pause across a weight update and resume on their KV cache.
        "--pause-generation-mode in_place "
    )

    eval_args = "--skip-eval-before-train "

    perf_args = (
        f"--tensor-model-parallel-size {TENSOR_PARALLEL_SIZE} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {EXPERT_PARALLEL_SIZE} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
    )

    # GRPO with the KL term computed and logged but weighted 0.
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
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    # The session server forwards to SGLang's own router, which takes an engine out of
    # rotation only after 40 failed health checks 15 s apart and back after one success.
    sglang_args = (
        f"--rollout-num-gpus-per-engine {TENSOR_PARALLEL_SIZE} "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-tool-call-parser {SGLANG_TOOL_CALL_PARSER} "
        f"--sglang-reasoning-parser {SGLANG_REASONING_PARSER} "
        f"--sglang-router-port {SGLANG_ROUTER_PORT} "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--grad-reduce-in-bf16 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {rollout_num_gpus} "
        # Restart an SGLang engine that stops answering health checks.
        "--use-fault-tolerance "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )
    if args.dump_details:
        misc_args += f"--dump-details {args.run_root}/dump_details "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{args.wandb_args} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    # A Ray job does not inherit the launching shell: everything the plugins read in the
    # rollout workers is forwarded here.
    extra_env_vars = {
        # common.* plugins from scripts/miles.
        "PYTHONPATH": f"{args.megatron_path}:{MILES_SCRIPTS_DIR}:{U.repo_base_dir}",
        # Registers the agentic generate flags (--custom-agent-function-path,
        # --use-session-server, --tito-*) and routes the custom generate function through
        # Miles' GenerateFnInput interface; the session server then owns request routing.
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "GROUP_FILTER_MIN_REWARD_STD": str(args.group_filter_min_reward_std),
        "GROUP_FILTER_MAX_ENV_FAILURES": str(args.group_filter_max_env_failures),
        "ROLLOUT_CONCURRENCY": str(args.rollout_concurrency),
        "CAMEL_ENV_SERVICE_URL": args.env_service_url,
        "CAMEL_DATASET_NAME": args.dataset_name,
        "CAMEL_TRIAL_NAME": args.trial_name,
    }
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train_async.py",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


if __name__ == "__main__":
    app()
