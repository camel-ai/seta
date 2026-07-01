"""DeepSeek-V4 SGLang model backend with TITO + DSML tool-call parsing.

Talks directly to SGLang's native ``/generate`` endpoint (token-in / token-out)
and parses DeepSeek-V4's native DSML tool-call format. Designed to work with
the same SGLang server miles spins up internally — no SGLang flag changes
required (per the rule: never modify ``run_deepseek_v4.py`` config).

Compared to ``sglang_miles_model.SGLangModel`` (which targets Qwen3 + Hermes XML):

- Default tool parser: ``DSMLToolCallParser`` instead of ``HermesToolCallParser``
- Drops the Qwen3-specific ``enable_thinking`` knob
- Drops the ``logprob_start_len=None when compute_logprobs_for_new_tokens_only``
  workaround. We do NOT pass ``logprob_start_len`` at all (matches miles'
  default in ``sglang_rollout.py``), so SGLang returns only
  ``output_token_logprobs`` for the newly generated tokens. This keeps
  RadixCache hot across turns — passing ``logprob_start_len=0`` would force
  recomputation of input logprobs every call and drop prefix-cache hit rate
  to 0 (per SGLang's own pitfall guide).
- Smaller, V4-only surface area

Reuses ``SGLangClient`` and ``TokenManager`` from ``sglang_miles_model`` —
those are model-agnostic and there's no reason to fork them.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import ChatCompletion, ChatCompletionChunk, ModelType
from camel.utils import BaseTokenCounter
from openai import AsyncStream, Stream
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager,
    ChatCompletionStreamManager,
)
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from .client import SGLangClient
from .token import TokenManager
from .token_counter import TokenCounter
from .tool_parser import ToolCallParser

from .dsml_tool_parser import DSMLToolCallParser, _TOOL_CALLS_BLOCK

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


# Mirror the optional-tracing pattern in sglang_miles_model.sglang_model
if os.environ.get("LANGFUSE_ENABLED", "False").lower() == "true":
    try:
        from langfuse.decorators import observe
    except ImportError:
        from camel.utils import observe
elif os.environ.get("TRACEROOT_ENABLED", "False").lower() == "true":
    try:
        from traceroot import trace as observe  # type: ignore[import]
    except ImportError:
        from camel.utils import observe
else:
    from camel.utils import observe


logger = logging.getLogger(__name__)


# encoding_dsv4.py is the canonical DeepSeek-V4 chat encoder, vendored next to
# this module. The HF tokenizer config for /data/models/DeepSeek-V4-Flash-FP8
# does NOT ship a chat_template, so we cannot rely on
# `tokenizer.apply_chat_template`. Instead we import the vendored encoding_dsv4
# by file path (importing the sglang package in venv_cpu would drag in heavy
# deps we don't have here), and tokenize the rendered text with the HF
# tokenizer. The resulting prompt matches what miles' SGLang server itself
# produces — verified against /tmp/harbor/.../conv_*.json.
# download from https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/encoding/encoding_dsv4.py
_SGLANG_DSV4_PATH = os.path.join(os.path.dirname(__file__), "encoding_dsv4.py")


def _load_dsv4_encoder():
    spec = importlib.util.spec_from_file_location(
        "_sglang_encoding_dsv4", _SGLANG_DSV4_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load DeepSeek-V4 encoder from {_SGLANG_DSV4_PATH}; "
            "this file should be vendored alongside this module."
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DeepSeekV4SGLangModel(BaseModelBackend):
    """SGLang ``/generate`` backend specialised for DeepSeek-V4-Flash / -Pro.

    Token-in / token-out across multi-turn loops. Records per-token IDs and
    logprobs in ``self.token_manager`` so the rollout function can extract them
    directly into a miles ``Sample`` (no detokenize → retokenize round trip).

    Args:
        model_type: HF path or model id (e.g. ``/data/models/DeepSeek-V4-Flash-FP8``).
        tokenizer: HF tokenizer for chat-template formatting and tool-result encoding.
        base_url: SGLang server URL. Default ``http://localhost:30000``.
        model_config_dict: Sampling parameters dict (temperature, top_p,
            max_new_tokens, ...). Note: the agent-side ``max_tokens`` in
            ``seta_env_config.yaml`` is mapped to SGLang's ``max_new_tokens``
            via the agent layer; we pass ``model_config_dict`` through verbatim.
        tool_call_parser: Override tool parser. Default ``DSMLToolCallParser()``.
        return_logprobs: Whether to request logprobs for new tokens. Default True.
        api_key, url: Kept for ``BaseModelBackend`` signature compatibility; unused.
    """

    def __init__(
        self,
        model_type: ModelType | str,
        tokenizer: "PreTrainedTokenizerBase",
        client: SGLangClient | None = None,
        base_url: str = "http://localhost:30000",
        tool_call_parser: ToolCallParser | None = None,
        model_config_dict: dict[str, Any] | None = None,
        api_key: str | None = None,
        url: str | None = None,
        token_counter: BaseTokenCounter | None = None,
        timeout: float | None = None,
        max_retries: int = 60,
        return_logprobs: bool = True,
        thinking_mode: str = "thinking",
        reasoning_effort: str | None = None,
        return_routed_experts: bool = False,
        return_indexer_topk: bool = False,
        session_id: str | None = None,
        **_unused: Any,
    ) -> None:
        """
        thinking_mode: "thinking" (default) or "chat".
            Matches miles' deepseek-v4 launcher default
            (run_deepseek_v4.py:404 sets thinking=true on apply_chat_template).
            "thinking" mode lets the model emit <think>…</think> blocks before
            tool calls / final answers; "chat" suppresses them.

            For convenience the YAML config can put `thinking_mode` inside
            `model_config_dict`; we pop it out below so it doesn't leak into
            sampling_params on every /generate call.
        """
        # Allow YAML to specify thinking_mode inside model_config_dict
        if model_config_dict and "thinking_mode" in model_config_dict:
            thinking_mode = model_config_dict.pop("thinking_mode")
        # Same for reasoning_effort (None | "high" | "max").
        # "max" prepends the REASONING_EFFORT_MAX prompt to the first message
        # (encoding_dsv4.py:295); "high" is accepted by the encoder but has no
        # effect today; None disables.
        if model_config_dict and "reasoning_effort" in model_config_dict:
            reasoning_effort = model_config_dict.pop("reasoning_effort")
        # Allow request-side override of routing-capture flags via
        # model_config_dict (env_service threads args.use_rollout_routing_replay
        # / args.use_rollout_indexer_replay through this dict).
        if model_config_dict and "return_routed_experts" in model_config_dict:
            return_routed_experts = bool(model_config_dict.pop("return_routed_experts"))
        if model_config_dict and "return_indexer_topk" in model_config_dict:
            return_indexer_topk = bool(model_config_dict.pop("return_indexer_topk"))
        super().__init__(
            model_type,
            model_config_dict or {},
            api_key,
            url,
            token_counter,
            timeout,
            max_retries,
        )

        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser or DSMLToolCallParser()
        self._return_logprobs = return_logprobs
        self._return_routed_experts = return_routed_experts
        self._return_indexer_topk = return_indexer_topk
        # Routing capture buffers. SGLang's /generate returns base64-encoded
        # int32 buffers covering tokens[: seqlen-1] of the FULL request seqlen
        # on every call (cached prefix tokens included — their routing was
        # recorded on first prefill). We overwrite on each turn; the last call
        # covers the entire concatenated trajectory.
        self._last_routed_experts_b64: str | None = None
        self._last_indexer_topk_b64: str | None = None
        # token_count = len(input_ids) + len(generated_tokens) at the time of
        # the last /generate call, i.e. the seqlen used to slice [: seqlen-1].
        self._last_routing_token_count: int = 0
        # Routing buffer SHAPES, so downstream decode (generate_with_camel,
        # build_rollout_dump) is self-describing instead of hardcoding per-model
        # constants. routed_experts is (seqlen-1, num_hidden_layers,
        # num_experts_per_tok) — SGLang does NOT return its shape, so we source
        # it from the model config. indexer_topk's num_layers DOES come back
        # from the server (meta_info.indexer_topk_num_layers); its topk dim is
        # recovered from the buffer length at dump time.
        self._routed_experts_num_layers, self._routed_experts_topk = (
            self._resolve_routing_dims(model_type)
        )
        self._last_indexer_num_layers: int | None = None
        self._dsv4 = _load_dsv4_encoder()
        if thinking_mode not in ("thinking", "chat"):
            raise ValueError(
                f"thinking_mode must be 'thinking' or 'chat', got {thinking_mode!r}"
            )
        self._thinking_mode = thinking_mode

        if reasoning_effort not in (None, "high", "max"):
            raise ValueError(
                f"reasoning_effort must be None, 'high', or 'max', got {reasoning_effort!r}"
            )
        self._reasoning_effort = reasoning_effort

        # /generate endpoint — strip /v1 if someone passes an OpenAI-style URL
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/v1"):
            cleaned = cleaned[: -len("/v1")]
        self._base_url = cleaned

        self._timeout = timeout
        self._max_retries = max_retries

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            # session_id is forwarded to SGLangClient so every /generate POST
            # carries X-Session-Id; the miles-router ConsistentHashMiddleware
            # pins all turns of the same trajectory to one SGLang engine.
            self._client = SGLangClient(
                self._base_url,
                timeout=timeout,
                max_retries=max_retries,
                session_id=session_id,
            )
            self._owns_client = True

        # TITO state
        self.token_manager = TokenManager()
        self._processed_message_count: int = 0
        self._current_tools: list[dict] | None = None
        self.tool_parse_errors: dict[str, int] = {}

        # Cumulative SGLang prefix-cache accounting across the trajectory.
        # SGLang returns per-call cached_tokens/prompt_tokens in meta_info;
        # miles' Sample.prefix_cache_info expects the trajectory-level sum
        # (it does cached_tokens / total_prompt_tokens for the hit rate).
        # Without this, the env_service-backed agent flow reports 0 in
        # wandb because miles' custom-generate path never sees meta_info.
        self._cumulative_prompt_tokens: int = 0
        self._cumulative_cached_tokens: int = 0
        # Cumulative per-token entropy from SGLang sampler (only populated if
        # return_logprobs=True and the SGLang build has the entropy patch).
        # entropy_sum is sum-of-per-token-entropy; entropy_count is num tokens.
        # Per-trajectory mean = entropy_sum / entropy_count.
        self._cumulative_entropy_sum: float = 0.0
        self._cumulative_entropy_count: int = 0

        logger.debug(
            "DeepSeekV4SGLangModel initialised: base_url=%s tokenizer=%s",
            self._base_url,
            getattr(tokenizer, "name_or_path", str(tokenizer)),
        )

    # ------------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset accumulated tokens for a new episode."""
        self.token_manager.reset()
        self._processed_message_count = 0
        self._current_tools = None
        self.tool_parse_errors = {}
        self._cumulative_prompt_tokens = 0
        self._cumulative_cached_tokens = 0
        self._cumulative_entropy_sum = 0.0
        self._cumulative_entropy_count = 0

    def get_cache_stats(self) -> dict:
        """Return cumulative SGLang prefix-cache accounting for this episode.

        Shape matches what miles' Sample.PrefixCacheInfo.add() expects:
        ``cached_tokens`` and ``prompt_tokens`` are summed across all /generate
        calls made during the trajectory's agent loop.
        """
        mean_entropy = (
            self._cumulative_entropy_sum / self._cumulative_entropy_count
            if self._cumulative_entropy_count > 0
            else None
        )
        return {
            "prompt_tokens": self._cumulative_prompt_tokens,
            "cached_tokens": self._cumulative_cached_tokens,
            "entropy_sum": self._cumulative_entropy_sum,
            "entropy_count": self._cumulative_entropy_count,
            "mean_entropy": mean_entropy,
        }

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    # ------------------------------------------------------------------------
    # required by BaseModelBackend
    # ------------------------------------------------------------------------

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Token counter using the HF tokenizer."""
        if not self._token_counter:
            self._token_counter = TokenCounter(self.tokenizer)
        return self._token_counter

    @property
    def stream(self) -> bool:
        """Always non-streaming — better parallelism for RL training."""
        return False

    # ------------------------------------------------------------------------
    # TITO state dump (for stage-2 audit + miles Sample construction)
    # ------------------------------------------------------------------------

    @staticmethod
    def _resolve_routing_dims(model_type) -> tuple[int | None, int | None]:
        """Resolve (num_layers, topk) for the routed-experts buffer from the
        model config, so the R3 decode is self-describing rather than relying on
        hardcoded per-model constants. routed_experts is shaped
        (seqlen-1, num_hidden_layers, num_experts_per_tok); SGLang returns only
        the flat buffer, never its shape. Reads config.json directly (no
        remote-code import) for a local checkpoint dir; falls back to AutoConfig
        for a hub id. Returns (None, None) if it can't be determined — callers
        then fall back to args / defaults.
        """
        try:
            cfg_path = os.path.join(str(model_type), "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
            else:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(str(model_type), trust_remote_code=True).to_dict()
            text_cfg = cfg.get("text_config") or cfg  # multimodal nests it
            nl = text_cfg.get("num_hidden_layers")
            tk = text_cfg.get("num_experts_per_tok")
            return (int(nl) if nl else None, int(tk) if tk else None)
        except Exception as e:
            logger.warning("could not resolve routing dims from %r: %s", model_type, e)
            return (None, None)

    def _compute_indexer_topk_k(self) -> int | None:
        """Recover the indexer topk dim from the captured buffer length.
        indexer_topk is (seqlen-1, num_layers, k); the server gives num_layers
        (``_last_indexer_num_layers``), so k = len(buffer) / ((seqlen-1)*layers).
        """
        b64 = self._last_indexer_topk_b64
        nl = self._last_indexer_num_layers
        tc = self._last_routing_token_count
        if not b64 or not nl or tc <= 1:
            return None
        n_int32 = len(base64.b64decode(b64.encode("ascii"))) // 4  # int32 = 4 bytes
        denom = (tc - 1) * int(nl)
        return n_int32 // denom if denom and n_int32 % denom == 0 else None

    def dump_tito_state(self, path: str) -> None:
        """Dump the accumulated TokenManager state in miles ``Sample``-shaped
        JSON for offline audit and downstream adapter consumption.

        Per ``miles.utils.types.Sample.validate``:
          - ``tokens``: full prompt + response token IDs
          - ``response_length``: count of post-prompt tokens
          - ``loss_mask``: length == response_length (response-only); 1 for
            assistant tokens, 0 for tool-result tokens
          - ``rollout_log_probs``: length == response_length (response-only)

        The TokenManager internally tracks loss_mask per-token over the full
        sequence (prompt=0, response=1, tool-result=0). We slice off the first
        prompt segment to get the miles-shaped response-only arrays.

        See `scripts/miles/docs/sample_contract.md` for the full contract.
        """
        import json
        from pathlib import Path as _Path

        full_token_ids = self.token_manager.token_ids
        full_loss_mask = self.token_manager.loss_mask  # 1 for response, 0 for prompt/tool
        full_logprobs = self.token_manager.logprobs    # may contain None
        segments = self.token_manager.segment_info     # [(is_response, length), ...]

        # ── R3 trailing-token truncation ────────────────────────────────────
        # Multi-turn agents can leave tokens in token_manager.token_ids that
        # were never fed through a /generate call — typically the FINAL turn's
        # tool result tokens when the loop terminates (max_iteration, abort,
        # sandbox crash, or any non-task_finished reason) after a tool exec
        # rather than after another /generate. Those trailing tokens carry
        # neither routing data nor training signal (loss_mask=0). They violate
        # miles' R3 invariant `rollout_routed_experts.shape[0] == len(tokens) - 1`
        # and crash actor._fill_replay_data with an AssertionError.
        #
        # Truncate token_ids / loss_mask / logprobs / segments back to
        # `_last_routing_token_count` (the seqlen of the last /generate call's
        # input+output) so the invariant holds. Loss is mathematically
        # unchanged since the dropped tokens all had loss_mask=0.
        if (
            (self._return_routed_experts or self._return_indexer_topk)
            and self._last_routing_token_count
            and self._last_routing_token_count < len(full_token_ids)
        ):
            target_len = self._last_routing_token_count
            dropped = len(full_token_ids) - target_len
            logger.warning(
                "R3 trailing-token truncation: dropping %d tokens "
                "(token_manager=%d, last_routing_token_count=%d). "
                "These are post-last-/generate tokens (tool results or "
                "partial output on abort) with no routing data.",
                dropped, len(full_token_ids), target_len,
            )
            full_token_ids = full_token_ids[:target_len]
            full_loss_mask = full_loss_mask[:target_len]
            full_logprobs = full_logprobs[:target_len]
            # Truncate segments to match: walk through, keeping each segment
            # whole until the running total would exceed target_len, then
            # truncate the boundary segment and drop the rest.
            new_segments: list[tuple] = []
            running = 0
            for is_resp, length in segments:
                if running + length <= target_len:
                    new_segments.append((is_resp, length))
                    running += length
                else:
                    remaining = target_len - running
                    if remaining > 0:
                        new_segments.append((is_resp, remaining))
                    break
            segments = new_segments

        # The first segment is always the initial prompt (system + tools + user).
        # Strip it to get miles' response-only view. Subsequent prompt-style
        # segments (tool results) are part of the response, with loss_mask=0.
        prompt_len = segments[0][1] if segments else 0
        response_token_ids = full_token_ids[prompt_len:]
        response_loss_mask = full_loss_mask[prompt_len:]
        response_logprobs = full_logprobs[prompt_len:]

        # Replace None logprobs with 0.0 placeholders for dense JSON; track
        # presence separately. Tool-result tokens (loss_mask=0) get
        # placeholders since SGLang never assigned them probability.
        clean_logprobs: list[float] = []
        logprob_present_mask: list[int] = []
        for lp in response_logprobs:
            if lp is None:
                clean_logprobs.append(0.0)
                logprob_present_mask.append(0)
            else:
                clean_logprobs.append(float(lp))
                logprob_present_mask.append(1)

        # Decode response text — for miles' `sample.response` (used by RM).
        # skip_special_tokens=True drops EOS markers (matches what SGLang
        # serves in /v1/chat/completions content).
        assistant_only_ids = [
            tid for tid, m in zip(response_token_ids, response_loss_mask) if m == 1
        ]
        response_text = (
            self.tokenizer.decode(assistant_only_ids, skip_special_tokens=True)
            if assistant_only_ids
            else ""
        )
        first_prompt_text = (
            self.tokenizer.decode(full_token_ids[:prompt_len])
            if prompt_len
            else ""
        )

        # Self-consistency assertions before serialising — catch shape bugs early
        response_length = len(response_token_ids)
        assert len(response_loss_mask) == response_length, (
            f"loss_mask length {len(response_loss_mask)} != response_length {response_length}"
        )
        assert len(clean_logprobs) == response_length, (
            f"rollout_log_probs length {len(clean_logprobs)} != response_length {response_length}"
        )
        assert len(full_token_ids) == prompt_len + response_length, (
            f"tokens {len(full_token_ids)} != prompt_len {prompt_len} + response_length {response_length}"
        )

        state: dict[str, Any] = {
            "schema_version": 2,  # bumped: response-only loss_mask + logprobs
            "model_type": str(self.model_type),
            "tokenizer_name_or_path": getattr(
                self.tokenizer, "name_or_path", str(self.tokenizer)
            ),
            # ─── miles Sample fields ─────────────────────────────────
            "tokens": list(full_token_ids),
            "response_length": response_length,
            "loss_mask": list(response_loss_mask),          # length == response_length
            "rollout_log_probs": clean_logprobs,            # length == response_length
            "response": response_text,
            # ─── diagnostic / non-essential ──────────────────────────
            "prompt_length": prompt_len,
            "first_prompt_text": first_prompt_text,
            "logprob_present_mask": logprob_present_mask,   # length == response_length
            "segments": [
                {"is_response": bool(is_resp), "length": length}
                for is_resp, length in segments
            ],
            "tool_parse_errors": dict(self.tool_parse_errors),
            # R3 routing capture (None if not enabled or SGLang server missing
            # --enable-return-routed-experts). The b64 buffer is the LAST
            # /generate call's full-trajectory buffer; token_count is the
            # seqlen used for the [: seqlen-1] slice. Downstream
            # build_rollout_dump.py decodes and validates alignment.
            "rollout_routed_experts_b64": self._last_routed_experts_b64,
            "rollout_indexer_topk_b64": self._last_indexer_topk_b64,
            "rollout_routing_token_count": self._last_routing_token_count,
            # Self-describing routing buffer shapes so downstream consumers
            # (generate_with_camel, build_rollout_dump) decode without hardcoded
            # per-model constants. routed_experts dims come from the model
            # config; indexer dims from the server + buffer length.
            "rollout_routed_experts_num_layers": self._routed_experts_num_layers,
            "rollout_routed_experts_topk": self._routed_experts_topk,
            "rollout_indexer_num_layers": self._last_indexer_num_layers,
            "rollout_indexer_topk_k": self._compute_indexer_topk_k(),
        }

        out = _Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------------------
    # message → prompt formatting
    # ------------------------------------------------------------------------

    @staticmethod
    def _format_message_for_template(message: OpenAIMessage) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message["role"]}
        result["content"] = message.get("content") or ""

        if message.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in message["tool_calls"]
            ]

        if message["role"] == "tool" and "tool_call_id" in message:
            result["tool_call_id"] = message["tool_call_id"]

        return result

    @staticmethod
    def _sort_tool_results(messages: list[OpenAIMessage]) -> list[OpenAIMessage]:
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        other_messages = [m for m in messages if m.get("role") != "tool"]
        tool_messages.sort(key=lambda m: m.get("tool_call_id", ""))
        return other_messages + tool_messages

    def format_prompt(
        self,
        messages: list[OpenAIMessage],
        tools: list[dict] | None = None,
        is_first_call: bool = True,
    ) -> str:
        """Render OpenAI-format messages into a DeepSeek-V4 prompt string.

        Uses ``encoding_dsv4.encode_messages`` from the bundled sglang
        installation — same encoder that miles' SGLang server uses internally.
        Falls back to the HF chat template if for some reason the encoder is
        unavailable.

        ``is_first_call=True`` adds the BOS token (start of conversation).
        Subsequent turns set this to False so we don't re-emit BOS when
        appending tool results.
        """
        chat_messages = [self._format_message_for_template(m) for m in messages]

        # encoding_dsv4 expects tools attached to a message via the "tools" key.
        # It looks for any message with `m.get("tools")`. Attach to the first
        # message (typically system) so they're picked up.
        if tools and chat_messages:
            chat_messages[0] = {**chat_messages[0], "tools": tools}

        return self._dsv4.encode_messages(
            chat_messages,
            thinking_mode=self._thinking_mode,
            add_default_bos_token=is_first_call,
            reasoning_effort=self._reasoning_effort,
        )

    def tokenize_prompt_messages(
        self,
        messages: list[OpenAIMessage],
    ) -> list[int] | None:
        """Tokenize prompt for the next /generate call.

        - First call: full prompt with tools.
        - Subsequent calls: only the new messages (e.g. tool results), prepended
          with the parser's ``message_separator`` so the chat-template framing
          stays consistent across turns.
        """
        if len(self.token_manager) == 0:
            formatted = self.format_prompt(
                messages, tools=self._current_tools, is_first_call=True
            )
            return self.tokenizer.encode(formatted, add_special_tokens=False)

        if len(messages) > self._processed_message_count:
            new_messages = self._sort_tool_results(
                messages[self._processed_message_count :]
            )
            formatted = self.format_prompt(new_messages, is_first_call=False)
            if self.tool_call_parser:
                formatted = self.tool_call_parser.message_separator + formatted
            return self.tokenizer.encode(formatted, add_special_tokens=False)

        return None

    # ------------------------------------------------------------------------
    # response building
    # ------------------------------------------------------------------------

    @staticmethod
    def _extract_logprobs(response: dict[str, Any], key: str) -> list[float] | None:
        meta_info = response.get("meta_info", {})
        logprobs = meta_info.get(key) or response.get(key)
        if isinstance(logprobs, list) and logprobs:
            return [entry[0] for entry in logprobs]
        return None

    def _build_chat_completion(
        self,
        text: str,
        tool_calls: list[ChatCompletionMessageToolCall] | None,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> ChatCompletion:
        message = ChatCompletionMessage(
            role="assistant",
            content=text or None,
            tool_calls=tool_calls or None,
        )
        choice = Choice(index=0, message=message, finish_reason=finish_reason)  # type: ignore[arg-type]
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ChatCompletion(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            choices=[choice],
            created=int(time.time()),
            model=str(self.model_type),
            object="chat.completion",
            usage=usage,
        )

    # ------------------------------------------------------------------------
    # BaseModelBackend interface
    # ------------------------------------------------------------------------

    @observe()
    def _run(
        self,
        messages: list[OpenAIMessage],
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> (
        ChatCompletion
        | Stream[ChatCompletionChunk]
        | ChatCompletionStreamManager[BaseModel]
    ):
        raise NotImplementedError(
            "DeepSeekV4SGLangModel only supports async (_arun)"
        )

    @observe()
    async def _arun(
        self,
        messages: list[OpenAIMessage],
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> (
        ChatCompletion
        | AsyncStream[ChatCompletionChunk]
        | AsyncChatCompletionStreamManager[BaseModel]
    ):
        # First call only — capture tool schema for chat template
        if tools and self._current_tools is None:
            self._current_tools = tools
            logger.debug("Captured %d tool schemas for chat template", len(tools))

        sampling_params: dict[str, Any] = dict(self.model_config_dict)

        new_input_tokens = self.tokenize_prompt_messages(messages)
        input_ids = self.token_manager.token_ids + (new_input_tokens or [])

        try:
            response = await self._client.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=self._return_logprobs,
                return_routed_experts=self._return_routed_experts,
                return_indexer_topk=self._return_indexer_topk,
                # NB: do NOT pass logprob_start_len. Setting it to 0 forces
                # SGLang to recompute logprobs for the entire input, which
                # drops RadixCache prefix-cache hit rate to 0 (per SGLang's
                # own pitfall guide). Default behaviour returns only
                # output_token_logprobs — exactly what we need for TITO,
                # since prompt/tool-result tokens have loss_mask=0 anyway
                # and don't contribute to the GRPO loss.
            )
            text = response.get("text", "")
            output_ids = response.get("output_ids", [])
            output_logprobs = self._extract_logprobs(response, "output_token_logprobs")
            input_logprobs = self._extract_logprobs(response, "input_token_logprobs")
            meta_info = response.get("meta_info", {})

            # Capture R3 routing for this call. SGLang returns a base64 buffer
            # of int32 shape (seqlen-1, num_layers, topk) covering every token
            # in the full request seqlen. In multi-turn, each subsequent call's
            # input grows to include prior turns, so the LAST call's buffer
            # covers the entire trajectory. Overwrite-on-each-turn:
            if self._return_routed_experts:
                b64 = meta_info.get("routed_experts")
                if b64:
                    self._last_routed_experts_b64 = b64
                    self._last_routing_token_count = len(input_ids) + len(output_ids)
            if self._return_indexer_topk:
                b64 = meta_info.get("indexer_topk")
                if b64:
                    self._last_indexer_topk_b64 = b64
                    # Server reports the indexer's layer count (unlike
                    # routed_experts); keep it so the decode is self-describing.
                    self._last_indexer_num_layers = meta_info.get("indexer_topk_num_layers")
                    # token_count tracked once (same seqlen as routed_experts)
                    if not self._last_routing_token_count:
                        self._last_routing_token_count = len(input_ids) + len(output_ids)

            # Accumulate SGLang prefix-cache accounting for the trajectory.
            # Fall back to input_ids length when meta_info omits prompt_tokens
            # (mirrors the existing pattern at lines below for prompt_tokens).
            self._cumulative_prompt_tokens += int(
                meta_info.get("prompt_tokens") or len(input_ids)
            )
            self._cumulative_cached_tokens += int(
                meta_info.get("cached_tokens") or 0
            )

            # Accumulate per-token entropy of the sampling distribution.
            # Only present when the SGLang build has the entropy patch AND the
            # request enabled return_logprob=True.
            entropy_list = meta_info.get("output_token_entropy")
            if entropy_list:
                self._cumulative_entropy_sum += float(sum(entropy_list))
                self._cumulative_entropy_count += len(entropy_list)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_text = e.response.text.lower()
            if status == 400 and any(
                p in error_text
                for p in ("exceed", "too long", "max model len", "context length")
            ):
                raise RuntimeError(f"Context length exceeded: {e.response.text}") from e
            raise

        # Update TITO trajectory
        if new_input_tokens:
            new_input_logprobs = (
                input_logprobs[-len(new_input_tokens):] if input_logprobs else None
            )
            self.token_manager.add_prompt(
                token_ids=new_input_tokens, logprobs=new_input_logprobs
            )
        if output_ids:
            self.token_manager.add_response(
                token_ids=output_ids, logprobs=output_logprobs
            )
        self._processed_message_count = len(messages) + 1

        # Parse DSML tool calls out of the response text.
        # The parser extracts tool_calls but does NOT strip the DSML markup
        # from `text` — so without further action, the assistant's `content`
        # would carry the same tool-call block that's also serialized into
        # `tool_calls`, producing a duplicated representation in conv logs.
        # We strip it here so `content` ends up as prose-only.
        parsed = self.tool_call_parser.parse(text)
        content_only = _TOOL_CALLS_BLOCK.sub("", text).strip()
        openai_tool_calls: list[ChatCompletionMessageToolCall] | None = None
        if parsed:
            openai_tool_calls = []
            for tc in parsed:
                if tc.is_error:
                    self.tool_parse_errors[tc.name] = (
                        self.tool_parse_errors.get(tc.name, 0) + 1
                    )
                    logger.warning(
                        "DSML parse error for '%s': %s",
                        tc.name,
                        (tc.raw or "")[:200],
                    )
                openai_tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tc.id,
                        type="function",
                        function=Function(
                            name=tc.name, arguments=tc.payload
                        ),
                    )
                )

        finish_reason = "tool_calls" if parsed else "stop"
        if isinstance(meta_info.get("finish_reason"), dict):
            if meta_info["finish_reason"].get("type") == "length":
                finish_reason = "length"

        prompt_tokens = int(meta_info.get("prompt_tokens") or len(input_ids))
        completion_tokens = int(meta_info.get("completion_tokens") or len(output_ids))

        return self._build_chat_completion(
            text=content_only,
            tool_calls=openai_tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
