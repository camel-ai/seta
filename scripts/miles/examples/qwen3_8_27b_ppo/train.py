"""Qwen3.8-27B fully-async PPO on terminal tasks, rolled out by the Harbor agent server.

    python train.py train --prompt-data ... [--flag value ...]

Launched by ``run_harbor_terminus2.sh``, which owns the Harbor agent server,
the sandbox backend and the version pins; this file only builds the Miles
command. Defaults are the configuration the script was run with.

* **PPO with a critic.** ``--advantage-estimator ppo`` turns the critic on;
  it shares the training GPUs with the actor (Miles places both in one
  placement group) and starts from the same checkpoint with a fresh value head.
* **Fully async.** ``train_async.py`` + ``--fully-async``: a persistent
  producer keeps trajectories in flight while the trainer drains finished
  groups, so step time tends to max(rollout, train) instead of their sum.
  ``--fully-async`` needs ``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`` in the Ray
  workers (passed through ``extra_env_vars``), forbids
  ``--rollout-function-path`` and ``--colocate``, and rejects the ``abort``
  pause mode.
* **Disaggregated placement.** ``num_nodes - rollout_num_nodes`` training
  nodes and ``rollout_num_nodes`` nodes of single-GPU SGLang engines.
* **Harbor rollout.** Miles' agentic generate function calls
  ``common.harbor_agent.run``, which posts each trial to the Harbor agent
  server; the reward comes back in the sample metadata.
* **Off-policy correction.** ``--use-rollout-logprobs``: the ratio denominator
  is the engine's own sampling log-probs, i.e. the policy that actually
  produced each token. Miles refuses async + critic without one of
  ``--use-rollout-logprobs`` / ``--use-tis`` / ``--keep-old-actor``; this one
  needs no recompute and no third model copy on the training GPUs.

Placement at the defaults (4 nodes x 8 GPUs):
  training  2 nodes  TP4 x PP2 x CP1 = DP2, actor + critic
  rollout   2 nodes  16 SGLang engines, one GPU each
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

# Miles plugins from scripts/miles/common. The run script puts scripts/miles on
# PYTHONPATH, and Miles forwards the launcher's PYTHONPATH into the Ray runtime env.
AGENT_FUNCTION_PATH = "common.harbor_agent.run"
REWARD_FUNCTION_PATH = "common.harbor_rollout.reward_func"
FILTER_MODULE = "common.harbor_rollout"
DYNAMIC_SAMPLING_FILTERS = ("none", "check_no_infra_failure", "check_no_infra_failure_and_nonzero_std")

# Megatron model type qwen3.5-27B, not qwen3.8-27B: the pinned Miles has no
# qwen3.8-27B entry (upstream added it later). Qwen3.8-27B is architecturally
# identical to Qwen3.5-27B (its config.json declares model_type qwen3_5; 64
# layers, hidden 5120, 24 heads, 4 query groups, vocab 248320), so this is the
# same architecture under the name the pinned Miles knows.
MEGATRON_MODEL_TYPE = "qwen3.5-27B"
TITO_MODEL = "qwen35"
# SGLang splits reasoning out of content only when launched with a reasoning
# parser. The Qwen3.8 template puts "<think>\n" in the generation prompt, so
# without it the whole thinking block lands in content and the Terminus-2 XML
# parser warns "Extra text detected before <response> tag" on every turn.
SGLANG_REASONING_PARSER = "qwen3"
SGLANG_TOOL_CALL_PARSER = "qwen3_coder"
TENSOR_MODEL_PARALLEL_SIZE = 4
ROLLOUT_NUM_GPUS_PER_ENGINE = 1
# Model-native maximum. It must exceed the agent budget (context + response) so
# that SGLang is never the component that rejects an overshoot.
SGLANG_CONTEXT_LENGTH = 262144
SGLANG_MEM_FRACTION_STATIC = 0.8
# expandable_segments maps physical pages on demand so reserved memory tracks
# allocated memory. Without it the actor phase can fail with "Triton Error
# [CUDA]: out of memory" inside fla's l2norm_bwd autotuner: the caching
# allocator holds most of the card (fragmented by the large fp32 logits
# tensors), while Triton benchmarks a new kernel config for every sequence
# length bucket with memory outside that pool. Usable because train offload
# (torch_memory_saver) is off.
CUDA_ALLOC_CONF = "expandable_segments:True"
# Terminus-2 summarizes when the context passes HARBOR_MAX_SEQ_LEN minus this
# margin; the trigger must stay below the engine ceiling or compaction can never
# fire and every long trajectory is rejected by the engine.
SUMMARIZE_MARGIN_TOKENS = 8000


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = ""
    prompt_data: str = ""
    hf_checkpoint: str = "/root/models/Qwen3.8-27B"
    # Megatron torch_dist checkpoint the actor and the critic start from; see
    # ckpt_args below.
    ref_load: str = "/root/models/Qwen3.8-27B_torch_dist"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    num_nodes: int = 4
    rollout_num_nodes: int = 2
    output_dir: str = "runs"

    # PP2 halves the resident weights and gradients per GPU so the critic fits
    # next to the actor. max_tokens_per_gpu == max_seq_len at CP1: every sample
    # fits one micro-batch, so dynamic batching can always make the
    # micro-batch count divisible by DP2.
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 1
    max_tokens_per_gpu: int = 49152
    max_seq_len: int = 49152
    rollout_max_response_len: int = 16384

    # 64 prompts x 2 trials = 128 = one optimizer step per drained rollout.
    # 2 trials per prompt suffice: the critic, not the group, is the baseline.
    rollout_batch_size: int = 64
    n_samples_per_prompt: int = 2
    global_batch_size: int = 128
    num_rollout: int = 3000
    lr: float = 1e-6
    critic_lr: float = 1e-5
    save_interval: int = 30
    # A trajectory generated under weight version v stays usable through the
    # update to v+2, so a long step does not discard the long-trajectory tail.
    max_weight_staleness: int = 2
    # Trajectories in flight (also the agent server's --max-concurrent).
    async_max_concurrent_samples: int = 128
    # none | check_no_infra_failure | check_no_infra_failure_and_nonzero_std
    dynamic_sampling_filter: str = "check_no_infra_failure"
    session_server_port_start: int = 30000
    session_server_port_end: int = 30016
    session_server_startup_timeout_secs: int = 600
    extra_args: str = ""


def _validate_engine_parsers() -> None:
    """Resolve the parsers through Miles' own TITO binding, so a wrong or
    missing value fails here rather than in the first trajectory."""
    from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

    reasoning, tool_call = resolve_reasoning_and_tool_call_parser(
        TITO_MODEL, SGLANG_REASONING_PARSER, SGLANG_TOOL_CALL_PARSER
    )
    if reasoning != SGLANG_REASONING_PARSER or tool_call != SGLANG_TOOL_CALL_PARSER:
        raise ValueError(
            f"tito_model={TITO_MODEL} binds reasoning={reasoning!r} tool_call={tool_call!r}; "
            f"launcher passes {SGLANG_REASONING_PARSER!r}/{SGLANG_TOOL_CALL_PARSER!r}"
        )


def _validate_harbor_contract(args: ScriptArgs) -> None:
    """The Ray workers read the agent contract from env vars forwarded below.
    A missing one does not fail loudly: the agent function falls back to its
    own defaults (json parser, no summarization, 50 turns, ...) and the run
    trains on the wrong contract."""
    for key in (
        "AGENT_SERVER_URL",
        "AGENT_MODEL_NAME",
        "HARBOR_AGENT_NAME",
        "HARBOR_AGENT_MAX_ITERATIONS",
        "HARBOR_MAX_SEQ_LEN",
        "HARBOR_AGENT_CALL_TIMEOUT_SEC",
        "HARBOR_TERMINUS_PARSER",
        "HARBOR_INTERLEAVED_THINKING",
        "HARBOR_TERMINUS_ENABLE_SUMMARIZE",
    ):
        if not os.environ.get(key):
            raise ValueError(f"{key} must be exported by the run script")
    harbor_ctx = int(os.environ["HARBOR_MAX_SEQ_LEN"])
    # The agent may present up to HARBOR_MAX_SEQ_LEN of history; a shorter
    # training sequence would silently cut the tail of every long trajectory.
    if args.max_seq_len < harbor_ctx:
        raise ValueError(f"--max-seq-len {args.max_seq_len} is below the agent context {harbor_ctx}")
    trigger = harbor_ctx - SUMMARIZE_MARGIN_TOKENS
    ceiling = SGLANG_CONTEXT_LENGTH - args.rollout_max_response_len
    if trigger >= ceiling:
        raise ValueError(f"summarization trigger {trigger} must be below the engine ceiling {ceiling}")


def _validate_shape(args: ScriptArgs) -> int:
    train_nodes = args.num_nodes - args.rollout_num_nodes
    if not 0 < args.rollout_num_nodes < args.num_nodes:
        raise ValueError("need at least one training node and one rollout node (no --colocate under async)")
    train_gpus = train_nodes * args.num_gpus_per_node
    model_parallel = TENSOR_MODEL_PARALLEL_SIZE * args.pipeline_model_parallel_size * args.context_parallel_size
    if train_gpus % model_parallel:
        raise ValueError(
            f"TP{TENSOR_MODEL_PARALLEL_SIZE} x PP{args.pipeline_model_parallel_size} x "
            f"CP{args.context_parallel_size} = {model_parallel} must divide {train_gpus} training GPUs"
        )
    # If one full-length sample does not fit a micro-batch, the micro-batch
    # count can fail to divide by DP, and the memory sizing above no longer holds.
    if args.max_seq_len % args.context_parallel_size or (
        args.max_tokens_per_gpu * args.context_parallel_size < args.max_seq_len
    ):
        raise ValueError(
            f"max_seq_len {args.max_seq_len} must be divisible by CP{args.context_parallel_size} and fit "
            f"max_tokens_per_gpu {args.max_tokens_per_gpu} x CP{args.context_parallel_size}"
        )
    if args.rollout_batch_size * args.n_samples_per_prompt != args.global_batch_size:
        raise ValueError(
            f"rollout_batch_size {args.rollout_batch_size} x n_samples_per_prompt "
            f"{args.n_samples_per_prompt} must equal global_batch_size {args.global_batch_size}"
        )
    if args.dynamic_sampling_filter not in DYNAMIC_SAMPLING_FILTERS:
        raise ValueError(f"dynamic_sampling_filter must be one of {DYNAMIC_SAMPLING_FILTERS}")
    return train_nodes


def execute(args: ScriptArgs) -> None:
    train_nodes = _validate_shape(args)
    _validate_engine_parsers()
    _validate_harbor_contract(args)
    if not args.prompt_data:
        raise ValueError("--prompt-data is required")
    run_root = f"{args.output_dir}/{args.run_id}"

    # A fresh run initialises the actor from --ref-load: with no
    # latest_checkpointed_iteration.txt under --load, Miles sets
    # args.load = args.ref_load (weights only, no optimizer/rng). The same
    # checkpoint is the KL reference and the critic's starting point (its
    # scalar value head has no counterpart there and is initialised fresh). --hf-checkpoint supplies the tokenizer,
    # the config and the engines' boot weights, which the first weight sync
    # overwrites with the actor's. Relaunching with the same run id resumes from
    # --load instead.
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--load {run_root}/checkpoints "
        f"--save {run_root}/checkpoints "
        f"--critic-load {args.ref_load} "
        f"--critic-save {run_root}/checkpoints/critic "
        f"--save-interval {args.save_interval} "
        # Weights only: with optimizer state a torch_dist checkpoint is ~380 GB
        # per model (bf16 params + fp32 master + two Adam moments), params alone
        # ~54 GB. Applies to the actor and the critic.
        # A resume therefore restarts the optimizer state.
        "--no-save-optim "
        "--no-save-rng "
    )

    filter_args = ""
    if args.dynamic_sampling_filter != "none":
        # check_no_infra_failure rejects (and the driver refills) a group with
        # an aborted sample or a trial that ended in AgentError / Flushed /
        # Unknown: those never reached the verifier, and training on them as
        # reward 0 teaches the policy that its actions failed when a sandbox or
        # tunnel did. ..._and_nonzero_std also rejects groups with no reward
        # spread; refill then costs ~1/(1-d) more rollouts at a uniform-group
        # share d.
        filter_args = f"--dynamic-sampling-filter-path {FILTER_MODULE}.{args.dynamic_sampling_filter} "

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1.0 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--balance-data "
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        f"--custom-agent-function-path {AGENT_FUNCTION_PATH} "
        f"--custom-rm-path {REWARD_FUNCTION_PATH} "
        # No --rollout-function-path: --fully-async asserts it is unset. The
        # rollout stays agentic through the custom generate/agent/rm paths,
        # which FullyAsyncRolloutFn honours via generate_and_rm_group.
        "--fully-async "
        # Exact in-flight cap, not a granule: SampleBackfillSubmission submits a
        # group only while samples_in_flight + n <= cap, and each finished sample
        # frees its own slot. There is no --over-sampling-batch-size: the
        # producer keeps refilling, so a filtered group costs throughput, never
        # concurrency.
        f"--async-max-concurrent-samples {args.async_max_concurrent_samples} "
        "--rollout-submission-granularity sample "
        f"--max-weight-staleness {args.max_weight_staleness} "
        f"{filter_args}"
        # Qwen3.8-27B declares model_type qwen3_5: the Qwen3.5 wire protocol.
        f"--tito-model {TITO_MODEL} "
        "--use-session-server v2 "
        # in_place freezes in-flight requests across a weight update and resumes
        # them on the existing KV (keeps the appended-R3 path). retract recomputes
        # KV and forces full-R3 payloads; abort is rejected by --fully-async.
        # in_place conflicts only with p2p weight transfer; broadcast is the default.
        "--pause-generation-mode in_place "
        f"--session-server-port {args.session_server_port_start} {args.session_server_port_end} "
        f"--session-server-startup-timeout-secs {args.session_server_startup_timeout_secs} "
    )

    perf_args = (
        f"--tensor-model-parallel-size {TENSOR_MODEL_PARALLEL_SIZE} "
        "--sequence-parallel "
        # PP > 1 needs the pinned Miles' is_pp_last_stage guard: under
        # --use-rollout-logprobs the critic's advantage computation otherwise
        # crashes with values=None on non-last stages.
        f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
        f"--context-parallel-size {args.context_parallel_size} "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
    )

    # upstream examples/ppo/run_qwen3_4b_ppo.py, apart from the lr values. One
    # critic-only warmup rollout: the value head starts random, so the actor's
    # first step would otherwise run on noise advantages.
    ppo_args = (
        "--advantage-estimator ppo "
        f"--critic-lr {args.critic_lr} "
        "--num-critic-only-steps 1 "
        "--normalize-advantages "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    async_args = "--use-rollout-logprobs "

    # --no-offload-train is required. With offload on, Miles keeps no CPU copy
    # of the parameters (disable_param_buffers_cpu_backup), sleep() releases the
    # parameter pages and resume() only remaps addresses, so a full-weight actor
    # wakes with garbage weights and solves nothing. Only
    # --rematerialize-param-from-master-weight makes waking safe, and it
    # requires --colocate, which async training forbids.
    offload_args = "--no-offload-train "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
        # The precision-aware optimizer keeps a second gradient buffer (fp32 main
        # grads, tens of GB) besides the DDP reduce buffer; bf16 halves it.
        # Valid only with --use-precision-aware-optimizer.
        "--main-grads-dtype bf16 "
    )

    sglang_args = (
        f"--rollout-num-gpus {args.rollout_num_nodes * args.num_gpus_per_node} "
        f"--rollout-num-gpus-per-engine {ROLLOUT_NUM_GPUS_PER_ENGINE} "
        f"--sglang-mem-fraction-static {SGLANG_MEM_FRACTION_STATIC} "
        f"--sglang-context-length {SGLANG_CONTEXT_LENGTH} "
        f"--sglang-reasoning-parser {SGLANG_REASONING_PARSER} "
        f"--sglang-tool-call-parser {SGLANG_TOOL_CALL_PARSER} "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # Gradient accumulation and all-reduce in bf16: with actor and critic
        # co-resident an fp32 grad buffer does not fit next to them. Only
        # --grad-reduce-in-bf16 sets the grad buffer dtype; dropping
        # --accumulate-allreduce-grads-in-fp32 alone is inert because Megatron
        # re-enables it when main_grads_dtype is fp32.
        "--grad-reduce-in-bf16 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {train_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--dump-details {run_root}/dump_details "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{offload_args} "
        f"{ppo_args} "
        f"{async_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        megatron_path=args.megatron_path,
        # Only the runtime env reaches the Ray workers; the launcher shell's
        # exports reach the agent server but not the rollout workers, where
        # common.harbor_agent reads the agent contract.
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "AGENT_SERVER_URL": os.environ["AGENT_SERVER_URL"],
            "AGENT_MODEL_NAME": os.environ["AGENT_MODEL_NAME"],
            "HARBOR_AGENT_NAME": os.environ["HARBOR_AGENT_NAME"],
            "HARBOR_AGENT_MAX_ITERATIONS": os.environ["HARBOR_AGENT_MAX_ITERATIONS"],
            "HARBOR_MAX_SEQ_LEN": os.environ["HARBOR_MAX_SEQ_LEN"],
            "HARBOR_AGENT_CALL_TIMEOUT_SEC": os.environ["HARBOR_AGENT_CALL_TIMEOUT_SEC"],
            "HARBOR_TERMINUS_PARSER": os.environ["HARBOR_TERMINUS_PARSER"],
            "HARBOR_INTERLEAVED_THINKING": os.environ["HARBOR_INTERLEAVED_THINKING"],
            "HARBOR_TERMINUS_ENABLE_SUMMARIZE": os.environ["HARBOR_TERMINUS_ENABLE_SUMMARIZE"],
            "PYTORCH_CUDA_ALLOC_CONF": CUDA_ALLOC_CONF,
            "MILES_ROUTER_EXTERNAL_HOST": os.environ.get("MILES_ROUTER_EXTERNAL_HOST", ""),
            "MILES_HOST_IP": os.environ.get("MILES_HOST_IP", ""),
        },
        # The switch that makes the run asynchronous; there is no --async flag.
        train_script="train_async.py",
    )


app = typer.Typer(add_completion=False)


@app.callback()
def _cli() -> None:
    """Qwen3.8-27B PPO launcher (see README.md)."""


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    app()
