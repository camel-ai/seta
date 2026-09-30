"""GLM-5.2 (744B-A40B) training launcher: LoRA GRPO on terminal tasks.

``python train.py train --help`` lists every option. The run scripts in this
folder (``run_harbor_terminus2.sh``) start the Harbor agent server and pass
the options.

Rollout: Miles' stock ``agentic_tool_call.generate`` calls
``common.harbor_agent.run`` once per sample. That posts one trial to the Harbor
agent server, which starts the task's sandbox, runs Terminus-2 against a Miles
session server and verifies the result; the reward comes back with the trial.

Terminus-2 summarizes when its context fills and continues from the summary.
Session server v2 records the pre- and post-summary branches as a trajectory
tree and returns one ``Sample`` per kept leaf, all sharing the episode's rollout
id and terminal reward, with shared completions masked in every sibling but
their first owner. A long episode can therefore yield several training samples
without counting as several GRPO rollouts.

Training is colocated: Megatron (through Megatron-Bridge, TileLang DSA) and the
FP8 SGLang engines share every GPU and swap in and out around each rollout.
"""

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shlex
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


SCRIPT_DIR = Path(__file__).resolve().parent
# scripts/miles: the root of the ``common.harbor_*`` plugin modules.
MILES_SCRIPTS_DIR = Path(os.environ.get("MILES_SCRIPTS_DIR") or SCRIPT_DIR.parents[1]).resolve()
TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
)
# Topologies this recipe has been run with; expert parallelism spans all GPUs.
SUPPORTED_NUM_NODES = (4, 8)
# Terminus-2 starts summarizing this many tokens before HARBOR_MAX_SEQ_LEN.
SUMMARIZE_RESERVE_TOKENS = 8000

AGENT_FUNCTION = "common.harbor_agent.run"
REWARD_FUNCTION = "common.harbor_rollout.reward_func"
ROLLOUT_FUNCTION = "common.harbor_rollout.RolloutFn"
GENERATE_FUNCTION = "miles.rollout.generate_hub.agentic_tool_call.generate"
SAMPLING_FILTER = "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    megatron_model_type: str = "glm5.2-744B-A40B_lora"
    num_nodes: int = 4
    num_gpus_per_node: int = 8
    enable_eval: bool = False

    prompt_data: str = ""
    hf_checkpoint: str = "/root/models/GLM-5.2"
    fp8_rollout_checkpoint: str = "/root/models/GLM-5.2-FP8"
    output_dir: str = "runs"
    megatron_path: str = "/root/Megatron-LM"
    megatron_bridge_path: str = "/root/Megatron-Bridge-glm5_2"
    offload_train_disk_dir: str = "/root/miles_train_offload"

    max_seq_len: int = 49152
    max_tokens_per_gpu: int = 49152
    rollout_max_response_len: int = 16384
    sglang_context_length: int = 1048576
    num_rollout: int = 200
    over_sampling_batch_size: int = 6
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    rollout_temperature: float = 0.8
    save_interval: int = 10
    lr: str = "3e-5"

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = TARGET_MODULES
    dsa_attention_backend: Literal["tilelang"] = "tilelang"
    fp8_rollout_gpus_per_engine: int = 16
    sglang_mem_fraction_static: float = 0.85
    # SGLang's experimental LoRA forward path (SGLANG_EXPERIMENTAL_LORA_OPTI) on
    # the rollout engines; False serves LoRA through SGLang's default path.
    sglang_lora_fastpath: bool = True
    session_server_port_start: int = 30000
    session_server_port_end: int = 30064
    session_server_startup_timeout_secs: int = 600
    extra_args: str = ""


def _require_module(module: str, expected: Path) -> None:
    spec = importlib.util.find_spec(module)
    actual = Path(spec.origin).resolve() if spec and spec.origin else None
    if actual != expected.resolve():
        raise RuntimeError(f"{module} resolved to {actual}, expected {expected.resolve()}")


def _values(tokens: list[str], flag: str) -> list[str]:
    indexes = [index for index, token in enumerate(tokens) if token == flag]
    return [tokens[index + 1] for index in indexes if index + 1 < len(tokens)]


def _world_size(args: ScriptArgs) -> int:
    return args.num_nodes * args.num_gpus_per_node


def _parallel_args(args: ScriptArgs) -> str:
    return (
        f"--tensor-model-parallel-size {args.num_gpus_per_node} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {_world_size(args)} "
        "--expert-tensor-parallel-size 1 "
        "--qkv-format thd "
        "--micro-batch-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )


def _sglang_args(args: ScriptArgs) -> str:
    max_batch_size = 64
    return (
        f"--rollout-num-gpus-per-engine {args.fp8_rollout_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        # Engine-side stage timing: TTFT and time-per-output-token split a
        # request into queue, prefill and decode, which the batch-level
        # Decode/Prefill log lines cannot do.
        "--sglang-enable-metrics "
        "--sglang-enable-metrics-for-all-schedulers "
        "--sglang-enable-mfu-metrics "
        f"--sglang-ep-size {args.fp8_rollout_gpus_per_engine} "
        "--sglang-attention-backend nsa "
        "--sglang-nsa-decode-backend flashmla_kv "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-page-size 64 "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        f"--sglang-context-length {args.sglang_context_length} "
        f"--sglang-cuda-graph-max-bs {max_batch_size} "
        f"--sglang-max-running-requests {max_batch_size} "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-watchdog-timeout 3600 "
        "--sglang-moe-runner-backend triton "
        "--sglang-disable-shared-experts-fusion "
        f"--sglang-max-lora-rank {args.lora_rank} "
        # SGLang's default LoRA backend (csgmv) crashes the DSA MoE-LoRA rollout.
        "--sglang-lora-backend triton "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--sglang-router-port 31001 "
    )


def _write_sglang_config(args: ScriptArgs, checkpoint_dir: str) -> str:
    path = Path(checkpoint_dir) / "sglang_fp8_rollout.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    # The engines serve the FP8 checkpoint; weight sync pushes the trained LoRA.
    path.write_text(
        "sglang:\n"
        "  - name: default\n"
        f"    model_path: {args.fp8_rollout_checkpoint}\n"
        "    update_weights: true\n"
        "    server_groups:\n"
        "      - worker_type: regular\n"
        f"        num_gpus: {_world_size(args)}\n"
    )
    return str(path)


def _validate(train_args: str, args: ScriptArgs) -> None:
    if args.num_nodes not in SUPPORTED_NUM_NODES or args.num_gpus_per_node != 8:
        raise ValueError(
            f"this recipe runs on {' or '.join(map(str, SUPPORTED_NUM_NODES))} nodes of 8 GPUs; "
            f"got {args.num_nodes} x {args.num_gpus_per_node}"
        )
    if _world_size(args) % args.fp8_rollout_gpus_per_engine:
        raise ValueError("the rollout engines must tile every GPU")
    if not args.prompt_data:
        raise ValueError("--prompt-data is required")
    # Every resolved flag must appear once with the value this launcher computed,
    # so a flag repeated in --extra-args cannot silently change the recipe.
    tokens = shlex.split(train_args)
    expected = {
        "--over-sampling-batch-size": [str(args.over_sampling_batch_size)],
        "--rollout-batch-size": [str(args.rollout_batch_size)],
        "--n-samples-per-prompt": [str(args.n_samples_per_prompt)],
        "--global-batch-size": [str(args.global_batch_size)],
        "--tensor-model-parallel-size": [str(args.num_gpus_per_node)],
        "--pipeline-model-parallel-size": ["1"],
        "--expert-model-parallel-size": [str(_world_size(args))],
        "--context-parallel-size": ["1"],
        "--rollout-num-gpus-per-engine": [str(args.fp8_rollout_gpus_per_engine)],
        "--tito-model": ["glm47"],
        "--use-session-server": ["v2"],
        "--pause-generation-mode": ["abort"],
        "--sglang-tool-call-parser": ["glm47"],
        "--sglang-reasoning-parser": ["glm45"],
        "--max-seq-len": [str(args.max_seq_len)],
        "--session-server-port": [str(args.session_server_port_start)],
        "--session-server-startup-timeout-secs": [str(args.session_server_startup_timeout_secs)],
        "--custom-generate-function-path": [GENERATE_FUNCTION],
        "--custom-agent-function-path": [AGENT_FUNCTION],
        "--custom-rm-path": [REWARD_FUNCTION],
        "--rollout-function-path": [ROLLOUT_FUNCTION],
        "--dynamic-sampling-filter-path": [SAMPLING_FILTER],
    }
    for flag, values in expected.items():
        actual = _values(tokens, flag)
        if actual != values:
            raise ValueError(f"expected {flag} {values}; resolved {actual}")
    if args.session_server_port_end <= args.session_server_port_start:
        raise ValueError("the session-server port range [start, end) is empty")
    if os.environ.get("HARBOR_TERMINUS_ENABLE_SUMMARIZE", "").lower() in {"1", "true", "yes"}:
        # The engine rejects a request when prompt + max_tokens exceeds its
        # window, so the agent's summarization trigger must sit below that
        # ceiling or summarization never fires and every long trajectory 400s.
        trigger = int(os.environ["HARBOR_MAX_SEQ_LEN"]) - SUMMARIZE_RESERVE_TOKENS
        ceiling = args.sglang_context_length - args.rollout_max_response_len
        if trigger >= ceiling:
            raise ValueError(
                f"summarization trigger {trigger} must be below the engine ceiling "
                f"{ceiling} (= sglang_context_length {args.sglang_context_length} - "
                f"rollout_max_response_len {args.rollout_max_response_len})"
            )
    if args.over_sampling_batch_size < args.rollout_batch_size:
        raise ValueError("oversampling must fetch at least one accepted batch of prompt groups")
    if args.rollout_batch_size * args.n_samples_per_prompt != args.global_batch_size:
        raise ValueError("accepted samples (rollout_batch_size x n_samples_per_prompt) must equal global_batch_size")
    if args.enable_eval or any(token.startswith("--eval-") for token in tokens):
        raise ValueError("evaluation is not part of this recipe")
    if "--apply-chat-template" in tokens:
        raise ValueError("the session-server TITO template owns prompt rendering")
    if os.environ.get("HARBOR_AGENT_NAME") != "terminus-2":
        raise ValueError("this recipe trains the terminus-2 agent (HARBOR_AGENT_NAME=terminus-2)")


def _train(args: ScriptArgs) -> None:
    bridge_source = Path(args.megatron_bridge_path) / "src"
    bridge_kernel = bridge_source / "megatron/bridge/models/glm5/tilelang/tilelang_sparse_mla_bwd.py"
    if not bridge_kernel.is_file():
        raise RuntimeError(f"missing Megatron-Bridge GLM-5 kernel: {bridge_kernel}")
    checkpoint_dir = f"{args.output_dir}/{args.run_id}/checkpoints"
    sglang_config = _write_sglang_config(args, checkpoint_dir)
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        "--megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
        f"--save {checkpoint_dir} "
        f"--save-interval {args.save_interval} "
    )
    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
        "--lora-base-cpu-backup "
    )
    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--balance-data "
        f"--custom-generate-function-path {GENERATE_FUNCTION} "
        f"--custom-agent-function-path {AGENT_FUNCTION} "
        f"--custom-rm-path {REWARD_FUNCTION} "
        f"--rollout-function-path {ROLLOUT_FUNCTION} "
        f"--dynamic-sampling-filter-path {SAMPLING_FILTER} "
        "--tito-model glm47 "
        "--use-session-server v2 "
        # abort is the only pause mode that drains in-flight requests before the
        # weight update's flush_cache, and with session server v2 it keeps
        # rollout routing replay (R3) incremental instead of resending the full
        # payload every turn. The requests it aborts belong to oversampled
        # groups that were going to be discarded.
        "--pause-generation-mode abort "
        f"--session-server-port {args.session_server_port_start} {args.session_server_port_end} "
        f"--session-server-startup-timeout-secs {args.session_server_startup_timeout_secs} "
    )
    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    misc_args = (
        "--use-rollout-routing-replay "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        # Training state goes to node-local disk while the engines hold the GPUs.
        "--offload-train-target disk "
        f"--offload-train-disk-dir {args.offload_train_disk_dir}/{args.run_id} "
        "--offload-train-disk-chunk-mb 256 "
        "--observe-training-entropy "
        "--no-check-for-nan-in-loss-and-grad "
        "--use-prometheus "
        "--prometheus-port 9091 "
        f"--prometheus-run-name {args.run_id} "
    )
    train_args = (
        ckpt_args
        + lora_args
        + rollout_args
        + optimizer_args
        + grpo_args
        + _parallel_args(args)
        + _sglang_args(args)
        + f"--sglang-config {sglang_config} "
        + misc_args
        + args.extra_args
    )
    _validate(train_args, args)
    _require_module("common.harbor_agent", MILES_SCRIPTS_DIR / "common/harbor_agent.py")
    _require_module("common.harbor_rollout", MILES_SCRIPTS_DIR / "common/harbor_rollout.py")

    sglang_lora_env = (
        {
            "SGLANG_EXPERIMENTAL_LORA_OPTI": "1",
            # Required once the fast path is on: without it the LoRA side stream
            # allocates its output on its own schedule, a reuse hazard under
            # CUDA graph replay.
            "SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC": "1",
        }
        if args.sglang_lora_fastpath
        else {}
    )
    # Only this runtime env reaches the Ray workers, where the agent function
    # runs; variables exported in the launcher shell do not.
    extra_env_vars = {
        # Megatron-Bridge first: its GLM-5 sparse-MLA backward fix must shadow the
        # copy installed in the image.
        "PYTHONPATH": f"{bridge_source}:{args.megatron_path}:{U.repo_base_dir}:{MILES_SCRIPTS_DIR}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_SERVER_URL": os.environ["AGENT_SERVER_URL"],
        "AGENT_MODEL_NAME": os.environ["AGENT_MODEL_NAME"],
        "HARBOR_AGENT_NAME": os.environ["HARBOR_AGENT_NAME"],
        "HARBOR_AGENT_MAX_ITERATIONS": os.environ["HARBOR_AGENT_MAX_ITERATIONS"],
        "HARBOR_AGENT_CALL_TIMEOUT_SEC": os.environ["HARBOR_AGENT_CALL_TIMEOUT_SEC"],
        "HARBOR_MAX_SEQ_LEN": os.environ["HARBOR_MAX_SEQ_LEN"],
        "HARBOR_TERMINUS_PARSER": os.environ["HARBOR_TERMINUS_PARSER"],
        "HARBOR_INTERLEAVED_THINKING": os.environ["HARBOR_INTERLEAVED_THINKING"],
        "HARBOR_TERMINUS_ENABLE_SUMMARIZE": os.environ["HARBOR_TERMINUS_ENABLE_SUMMARIZE"],
        "MILES_ROUTER_EXTERNAL_HOST": os.environ.get("MILES_ROUTER_EXTERNAL_HOST", ""),
        "MILES_HOST_IP": os.environ.get("MILES_HOST_IP", ""),
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "SGLANG_NSA_FORCE_MLA": "1",
        **sglang_lora_env,
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
    }
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


app = typer.Typer(no_args_is_help=True)


@app.callback()
def _cli() -> None:
    """GLM-5.2 training launcher (the model is prepared by prepare_model.sh)."""


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """LoRA GRPO with Terminus-2 rollouts on the Harbor agent server."""
    _train(args)


if __name__ == "__main__":
    app()
