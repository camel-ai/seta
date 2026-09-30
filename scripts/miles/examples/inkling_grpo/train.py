"""Miles launcher for Inkling-Small full-parameter GRPO on terminal tasks, with rollouts through the
Harbor agent server. Called by run_harbor_camel.sh and run_harbor_terminus2.sh (which set up Ray,
Harbor and the agent environment); prepare the model first with prepare_model.py.

`--agent` selects the Harbor agent; everything else (actor mesh, serving, GRPO, memory offload,
session servers) is shared:
  camel       CAMEL agent with native tool calls. The Miles session server parses Inkling's output
              (--tito-model inkling); SGLang's parsers stay off.
  terminus-2  Harbor's Terminus-2 agent (JSON commands in plain text). SGLang's Inkling tool and
              reasoning parsers are on.

The agent settings (HARBOR_AGENT_NAME, AGENT_SERVER_URL, ...) come from the environment that the
run scripts and common/launcher.sh export; `train` forwards them to the Ray workers.

  DRY_RUN=1 bash scripts/miles/examples/inkling_grpo/run_harbor_camel.sh   # prints the full train.py command
"""

import importlib.util
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer(no_args_is_help=True)

MODEL_NAME = "Inkling-Small"
MEGATRON_MODEL_TYPE = "inkling-small"  # scripts/models/inkling-small.sh in Miles

# scripts/miles: the Harbor plugins below are imported from here (it is on PYTHONPATH).
MILES_SCRIPTS_DIR = Path(os.environ.get("MILES_SCRIPTS_DIR") or Path(__file__).resolve().parents[2])
AGENT_FUNCTION = "common.harbor_agent.run"
REWARD_FUNCTION = "common.harbor_rollout.reward_func"
ROLLOUT_FUNCTION = "common.harbor_rollout.RolloutFn"
GENERATE_FUNCTION = "miles.rollout.generate_hub.agentic_tool_call.generate"
SAMPLE_FILTER = "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted"

# One pipeline stage per node. 42 decoder layers: 8 stages = 7 x 5 plus a last stage of 7;
# 7 stages = 6 layers each; 6 stages = 7 layers each.
_LAST_STAGE_LAYERS = {6: None, 7: None, 8: 7}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    agent: Literal["camel", "terminus-2"] = "camel"
    prompt_data: str = ""

    hf_checkpoint: str = f"/root/models/{MODEL_NAME}"
    torch_dist: str = f"/root/models/{MODEL_NAME}_torch_dist"
    # Actor start weights; default torch_dist (also the KL reference). An earlier run's
    # checkpoints/ directory continues from its latest weights with a fresh optimizer state.
    load: str | None = None
    megatron_path: str = "/root/Megatron-LM"
    # Keep across runs: the first actor step compiles attention/Inductor kernels for a long
    # time when this cache is cold.
    torchinductor_cache_dir: str | None = None

    num_gpus_per_node: int = 8
    rollout_num_gpus_per_engine: int = 8

    num_rollout: int = 100
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    over_sampling_batch_size: int = 20
    global_batch_size: int = 128
    lr: float = 1e-5
    max_response_len: int = 16384
    # Bound on a training sample (prompt + all turns), not the agent's context (HARBOR_MAX_SEQ_LEN).
    max_seq_len: int = 32768

    # One Miles session-server process per port in [start, end).
    session_server_port_start: int = 30000
    session_server_port_end: int = 30064

    save_interval: int = 99
    enable_r3: bool = True
    dump_details: bool = True
    # Extra Miles/Megatron/SGLang flags, appended last (the last occurrence of a flag wins).
    extra_args: str = ""

    def __post_init__(self):
        if self.load is None:
            self.load = self.torch_dist
        if self.torchinductor_cache_dir is None:
            self.torchinductor_cache_dir = f"{self.output_dir}/torchinductor_cache"


def _parallel_args(args: ScriptArgs) -> str:
    """Actor mesh: TP8 with sequence parallel, EP8, one pipeline stage per node, CP1, DP1."""
    if args.num_gpus_per_node != 8 or args.num_nodes not in _LAST_STAGE_LAYERS:
        raise NotImplementedError(
            f"no tested Inkling-Small layout for {args.num_nodes} nodes x {args.num_gpus_per_node} GPUs "
            f"(supported: {sorted(_LAST_STAGE_LAYERS)} nodes x 8 GPUs)"
        )
    parallel = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.num_nodes} "
    )
    if (last := _LAST_STAGE_LAYERS[args.num_nodes]) is not None:
        parallel += f"--decoder-last-pipeline-num-layers {last} "
    return parallel + "--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 "


def _train_args(args: ScriptArgs) -> str:
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--load {args.load} "
        f"--ref-load {args.torch_dist} "
        "--model-name inkling "
        "--megatron-to-hf-mode raw "
        "--no-load-optim --no-load-rng --finetune "
        # Weights only: with the Adam state a checkpoint of this model is several TB.
        f"--save {args.output_dir}/checkpoints "
        f"--save-interval {args.save_interval} "
        "--no-save-optim "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        f"--rollout-max-response-len {args.max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        f"--custom-generate-function-path {GENERATE_FUNCTION} "
        f"--custom-agent-function-path {AGENT_FUNCTION} "
        f"--custom-rm-path {REWARD_FUNCTION} "
        f"--rollout-function-path {ROLLOUT_FUNCTION} "
        f"--dynamic-sampling-filter-path {SAMPLE_FILTER} "
        f"--max-seq-len {args.max_seq_len} "
        # Token-in/token-out: training sees exactly the tokens the engine produced.
        "--tito-model inkling "
        "--use-session-server "
        f"--session-server-port {args.session_server_port_start} {args.session_server_port_end} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--eps-clip-c 3.0 "
        "--use-tis "
        # KL to the reference model and the entropy are logged, not added to the loss.
        "--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--observe-training-entropy "
    )
    if args.enable_r3:
        # Replay the MoE routing SGLang chose during rollout in the training forward pass.
        grpo_args += "--use-rollout-routing-replay "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-distributed-optimizer "
        "--accumulate-allreduce-grads-in-fp32 "
        "--no-check-for-nan-in-loss-and-grad "
        # Memory: training state and the optimizer live in CPU memory while SGLang serves on the
        # same GPUs; without this the Adam state does not fit.
        "--offload-train-target cpu "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    perf_args = (
        _parallel_args(args)
        + "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
    )

    # Serving on the training GPUs: one TP8/EP1 engine per node. TP4 engines leave no room for
    # the weight update and SGLang EP does not shrink the per-GPU footprint. A 1M-token KV pool
    # per engine avoids request retractions with 160 concurrent multi-turn trials. The router
    # pins each session to one engine and places new sessions on the least-loaded one.
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        "--sglang-ep-size 1 "
        "--sglang-mem-fraction-static 0.75 "
        "--sglang-max-running-requests 256 "
        "--sglang-max-total-tokens 1048576 "
        "--sglang-router-policy manual "
        "--router-assignment-mode min_load "
        "--sglang-enable-fp32-lm-head "
        "--sglang-attention-backend fa4 "
        "--sglang-moe-runner-backend triton "
        "--sglang-mamba-scheduler-strategy extra_buffer "
        "--sglang-disable-custom-all-reduce "
    )
    if args.agent == "terminus-2":
        # Terminus-2 reads plain text, so SGLang splits off Inkling's reasoning and tool blocks.
        sglang_args += "--sglang-tool-call-parser inkling --sglang-reasoning-parser inkling "

    misc_args = (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--no-bias-dropout-fusion "
        "--distributed-timeout-minutes 30 "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-miles-dashboard "
    )
    if args.dump_details:
        # Per-rollout samples and token ids, for checking the token-in/token-out match.
        misc_args += f"--dump-details {args.output_dir}/dump_details "

    return (
        f"{ckpt_args}{rollout_args}{grpo_args}{optimizer_args}{perf_args}"
        f"{sglang_args}{misc_args}{args.extra_args}"
    )


def _extra_env_vars(args: ScriptArgs) -> dict[str, str]:
    env = {
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        "SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM": "false",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "MILES_SGLANG_DUMMY_LOAD": "0",
        "SGLANG_SERVER_ENGINE_ROLLOUT_RETURN_LOGPROB": "1",
        "RAY_memory_monitor_refresh_ms": "0",
        "NCCL_MNNVL_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_RAS_ENABLE": "0",
        "TORCHINDUCTOR_CACHE_DIR": args.torchinductor_cache_dir,
        # Ray workers import the Harbor plugins (common.harbor_*) from scripts/miles.
        "PYTHONPATH": f"{args.megatron_path}:{U.repo_base_dir}:{MILES_SCRIPTS_DIR}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
    }
    for name in ("AGENT_SERVER_URL", "AGENT_MODEL_NAME", "HARBOR_AGENT_NAME",
                 "HARBOR_AGENT_MAX_ITERATIONS", "HARBOR_MAX_SEQ_LEN"):
        if not os.environ.get(name):
            raise ValueError(f"{name} must be set (the run_harbor_*.sh scripts export it)")
        env[name] = os.environ[name]
    env["MILES_ROUTER_EXTERNAL_HOST"] = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    env["MILES_HOST_IP"] = os.environ.get("MILES_HOST_IP", "")
    # Agent policy knobs are forwarded only when set; unset keeps Harbor's default.
    optional = ["HARBOR_AGENT_CALL_TIMEOUT_SEC"]
    if args.agent == "terminus-2":
        optional += ["HARBOR_TERMINUS_PARSER", "HARBOR_INTERLEAVED_THINKING", "HARBOR_TERMINUS_ENABLE_SUMMARIZE"]
    else:
        optional += ["HARBOR_CAMEL_MAX_COMPACTIONS"]
    env.update({name: os.environ[name] for name in optional if name in os.environ})
    return env


def _validate(args: ScriptArgs, train_args: str) -> None:
    if not args.prompt_data:
        raise ValueError("--prompt-data is required")
    if os.environ.get("HARBOR_AGENT_NAME") != args.agent:
        raise ValueError(f"HARBOR_AGENT_NAME={os.environ.get('HARBOR_AGENT_NAME')!r} does not match --agent {args.agent}")
    if args.over_sampling_batch_size < args.rollout_batch_size:
        raise ValueError("--over-sampling-batch-size must be at least --rollout-batch-size")
    if args.session_server_port_end <= args.session_server_port_start:
        raise ValueError("--session-server-port-end must be above --session-server-port-start")
    tokens = set(shlex.split(train_args))
    if args.agent == "camel" and tokens & {"--sglang-tool-call-parser", "--sglang-reasoning-parser"}:
        raise ValueError("camel: the Miles session server parses Inkling output; SGLang parsers must stay off")
    if tokens & {"--apply-chat-template", "--sglang-enable-multimodal"}:
        raise ValueError("agent trajectories are structured chat messages; drop --apply-chat-template/--sglang-enable-multimodal")
    if any(token.startswith("--eval-") for token in tokens):
        raise ValueError("in-training evaluation is not wired up in this script")


def _require_module(module: str, expected: Path) -> None:
    spec = importlib.util.find_spec(module)
    actual = Path(spec.origin).resolve() if spec and spec.origin else None
    if actual != expected.resolve():
        raise RuntimeError(f"{module} resolved to {actual}, expected {expected.resolve()}; check PYTHONPATH")


def _validate_plugin_paths() -> None:
    _require_module(GENERATE_FUNCTION.rsplit(".", 1)[0],
                    Path(U.repo_base_dir) / "miles/rollout/generate_hub/agentic_tool_call.py")
    for path in (AGENT_FUNCTION, REWARD_FUNCTION):
        module = path.rsplit(".", 1)[0]
        _require_module(module, MILES_SCRIPTS_DIR / f"{module.replace('.', '/')}.py")

    from miles.utils import misc

    # Without the abort hook, trials that oversampling no longer needs keep running in Harbor.
    if not hasattr(misc, "call_agent_abort_hook"):
        raise RuntimeError("this Miles checkout has no agent abort hook (miles.utils.misc.call_agent_abort_hook)")


@app.callback()
def main():
    """Inkling-Small GRPO launcher (see run_harbor_camel.sh / run_harbor_terminus2.sh)."""


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Submit Inkling-Small GRPO training to the running Ray cluster."""
    print(f"Inkling-Small full GRPO, agent {args.agent}, {args.num_nodes} nodes x {args.num_gpus_per_node} GPUs")
    train_args = _train_args(args)
    _validate(args, train_args)
    _validate_plugin_paths()
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        extra_env_vars=_extra_env_vars(args),
        megatron_path=args.megatron_path,
    )


if __name__ == "__main__":
    app()
