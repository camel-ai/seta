# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

"""SGLang native `/generate` API model backend for CAMEL with token-in/token-out training.

This backend uses SGLang's native HTTP APIs:
- `/generate` for text generation (returns output_ids directly)

It uses a HuggingFace tokenizer for:
- Applying chat templates (via tokenizer.apply_chat_template())
- Tokenizing prompts and tool results

This eliminates retokenization drift in RL training by maintaining token IDs
throughout the rollout instead of converting text back to tokens.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

import httpx
from camel.logger import get_logger
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import BaseTokenCounter, OpenAITokenCounter
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
from .tool_parser import HermesToolCallParser, ToolCallParser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

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


logger = get_logger(__name__)


class TokenCounter(BaseTokenCounter):
    """Token counter using HuggingFace tokenizer."""

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", tokens_per_message: int = 4):
        self.tokenizer = tokenizer
        self.tokens_per_message = tokens_per_message

    def count_tokens_from_messages(self, messages: list[OpenAIMessage]) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                if not isinstance(value, list):
                    num_tokens += len(self.tokenizer.encode(str(value)))
                else:
                    for item in value:
                        if item["type"] == "text":
                            num_tokens += len(self.tokenizer.encode(str(item["text"])))
                        else:
                            raise ValueError(f"Unsupported item type: {item['type']}")
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


class SGLangModel(BaseModelBackend):
    """SGLang native `/generate` API backend for CAMEL with TITO support.

    Uses a HuggingFace tokenizer for chat template formatting and SGLang's
    `/generate` endpoint for generation. Tracks token trajectories via `TokenManager`.

    Attributes:
        tokenizer: HuggingFace tokenizer for encoding/decoding.
        token_manager: Tracks tokens, logprobs, and masks for TITO training.
        tool_call_parser: Parser for extracting tool calls from model output.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        >>> client = SGLangClient("http://localhost:30000")
        >>> model = SGLangModel(
        ...     model_type="qwen3-4b",
        ...     tokenizer=tokenizer,
        ...     client=client,
        ... )
        >>> # After generation:
        >>> model.token_manager.token_ids    # Full token trajectory
        >>> model.token_manager.loss_mask    # Boolean mask for loss computation
        >>> model.token_manager.logprobs     # Log probabilities
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
        compute_logprobs_for_new_tokens_only: bool = False,  # Not used directly, but can be implemented via logprob_start_len
        enable_thinking: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SGLang model backend.

        Args:
            model_type: Model type identifier.
            tokenizer: HuggingFace tokenizer for chat template and tokenization.
            client: Optional `SGLangClient` for connection pooling and retry logic.
                    If `None`, creates a new client with base_url.
            base_url: SGLang server URL (default: http://localhost:30000).
            tool_call_parser: Parser for tool calls (default: HermesToolCallParser).
            model_config_dict: Configuration dict for sampling parameters.
            api_key: Not used (kept for compatibility).
            url: Not used (kept for compatibility).
            token_counter: Token counter (default: uses tokenizer).
            timeout: Request timeout in seconds, or None for infinite.
            max_retries: Maximum retry attempts (default: 60, like Slime).
            return_logprobs: Whether to return logprobs (default: True).
            enable_thinking: Enable thinking mode for Qwen3 hybrid models.
            **kwargs: Additional arguments.
        """
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
        self.tool_call_parser = tool_call_parser or HermesToolCallParser()
        # R3 routing-capture flags are NOT SGLang sampling params. The DeepSeek-V4
        # backend pops them and sends them top-level (and captures routed_experts);
        # the generic backend cannot capture routing, so we drop them so they never
        # leak into the /generate sampling_params dict (`dict(self.model_config_dict)`
        # below in _arun) — passing `return_routed_experts` would raise
        # `SamplingParams.__init__() got an unexpected keyword argument` -> 500 on
        # every generate. GLM trains with R3 disabled (see run_glm47_flash_*).
        if isinstance(getattr(self, "model_config_dict", None), dict):
            self.model_config_dict.pop("return_routed_experts", None)
            self.model_config_dict.pop("return_indexer_topk", None)
        self._return_logprobs = return_logprobs
        self._enable_thinking = enable_thinking
        self.compute_logprobs_for_new_tokens_only = compute_logprobs_for_new_tokens_only

        # HTTP client setup
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = SGLangClient(
                self._base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
            self._owns_client = True

        # TITO state
        self.token_manager = TokenManager()
        self._processed_message_count: int = 0
        self._current_tools: list[dict] | None = None

        # Parse error tracking (per tool name)
        self.tool_parse_errors: dict[str, int] = {}

        logger.debug(f"SGLangModel initialized: base_url={self._base_url}")

    def reset(self) -> None:
        """Reset token accumulation for a new episode.

        Call this at episode start. Clears all accumulated tokens and resets
        internal state for tool tracking.
        """
        self.token_manager.reset()
        self._processed_message_count = 0
        self._current_tools = None
        self.tool_parse_errors = {}

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client:
            await self._client.close()

    def dump_tito_state(self, path: str) -> None:
        """Dump accumulated TokenManager state as miles ``Sample``-shaped JSON.

        terminal_env calls this at trajectory close (it checks ``hasattr(model,
        'dump_tito_state')``); env_service then reads the file
        (HARBOR_ROOT/trials/<trial>/<uid>/tito_state.json) and copies the miles
        Sample fields onto a Sample for the custom-generate-fn adapter. Without
        this method the generic backend produces NO training sample (sample=None
        -> trajectory dropped).

        Mirrors DeepSeekV4SGLangModel.dump_tito_state but WITHOUT routed-experts
        (R3) capture: the generic backend never requests return_routed_experts,
        so the routing fields are emitted as None (valid when training runs with
        --use-rollout-routing-replay OFF).

        Contract (miles.utils.types.Sample.validate):
          tokens            : full prompt+response token IDs
          response_length   : count of post-first-prompt tokens
          loss_mask         : len == response_length (1=assistant, 0=tool-result)
          rollout_log_probs : len == response_length
        """
        import json
        from pathlib import Path as _Path

        full_token_ids = self.token_manager.token_ids
        full_loss_mask = self.token_manager.loss_mask   # 1=response, 0=prompt/tool
        full_logprobs = self.token_manager.logprobs     # may contain None
        segments = self.token_manager.segment_info      # [(is_response, length), ...]

        # First segment is the initial prompt (system + tools + user); strip it
        # for miles' response-only view. Later prompt-style segments (tool
        # results) stay in the response with loss_mask=0.
        prompt_len = segments[0][1] if segments else 0
        response_token_ids = full_token_ids[prompt_len:]
        response_loss_mask = full_loss_mask[prompt_len:]
        response_logprobs = full_logprobs[prompt_len:]

        # None logprobs -> 0.0 placeholders (dense JSON); track presence.
        clean_logprobs: list[float] = []
        logprob_present_mask: list[int] = []
        for lp in response_logprobs:
            if lp is None:
                clean_logprobs.append(0.0)
                logprob_present_mask.append(0)
            else:
                clean_logprobs.append(float(lp))
                logprob_present_mask.append(1)

        assistant_only_ids = [
            tid for tid, m in zip(response_token_ids, response_loss_mask) if m == 1
        ]
        response_text = (
            self.tokenizer.decode(assistant_only_ids, skip_special_tokens=True)
            if assistant_only_ids else ""
        )
        first_prompt_text = (
            self.tokenizer.decode(full_token_ids[:prompt_len]) if prompt_len else ""
        )

        response_length = len(response_token_ids)
        assert len(response_loss_mask) == response_length, (
            f"loss_mask {len(response_loss_mask)} != response_length {response_length}"
        )
        assert len(clean_logprobs) == response_length, (
            f"rollout_log_probs {len(clean_logprobs)} != response_length {response_length}"
        )
        assert len(full_token_ids) == prompt_len + response_length, (
            f"tokens {len(full_token_ids)} != prompt {prompt_len} + response {response_length}"
        )

        state: dict[str, Any] = {
            "schema_version": 2,
            "model_type": str(self.model_type),
            "tokenizer_name_or_path": getattr(
                self.tokenizer, "name_or_path", str(self.tokenizer)
            ),
            # ─── miles Sample fields ─────────────────────────────────
            "tokens": list(full_token_ids),
            "response_length": response_length,
            "loss_mask": list(response_loss_mask),
            "rollout_log_probs": clean_logprobs,
            "response": response_text,
            # ─── diagnostic / non-essential ──────────────────────────
            "prompt_length": prompt_len,
            "first_prompt_text": first_prompt_text,
            "logprob_present_mask": logprob_present_mask,
            "segments": [
                {"is_response": bool(is_resp), "length": length}
                for is_resp, length in segments
            ],
            "tool_parse_errors": dict(self.tool_parse_errors),
            # R3 disabled for the generic backend -> no routing capture.
            "rollout_routed_experts_b64": None,
            "rollout_indexer_topk_b64": None,
            "rollout_routing_token_count": None,
            "rollout_routed_experts_num_layers": None,
            "rollout_routed_experts_topk": None,
            "rollout_indexer_num_layers": None,
            "rollout_indexer_topk_k": None,
        }

        out = _Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(state))

    # -------------------------------------------------------------------------
    # Chat template and message formatting
    # -------------------------------------------------------------------------

    def _format_message_for_template(self, message: OpenAIMessage) -> dict[str, Any]:
        """Format a single OpenAI message for chat template.

        Converts tool_calls to the format expected by chat templates.
        """
        result: dict[str, Any] = {"role": message["role"]}

        # Handle content
        content = message.get("content")
        if content is not None:
            result["content"] = content
        else:
            result["content"] = ""

        # Handle tool_calls (assistant messages)
        if "tool_calls" in message and message["tool_calls"]:
            # Convert to format expected by chat template
            tool_calls = []
            for tc in message["tool_calls"]:
                tool_calls.append({
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                })
            result["tool_calls"] = tool_calls

        # Handle tool results (tool role messages)
        if message["role"] == "tool" and "tool_call_id" in message:
            result["tool_call_id"] = message["tool_call_id"]

        return result

    def _sort_tool_results(self, messages: list[OpenAIMessage]) -> list[OpenAIMessage]:
        """Sort tool result messages by tool_call_id for deterministic ordering.

        Tool results may arrive in any order due to parallel execution.
        Sorting by ID ensures deterministic token sequences for TITO.
        """
        # Separate tool messages from others
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        other_messages = [m for m in messages if m.get("role") != "tool"]

        # Sort tool messages by tool_call_id
        tool_messages.sort(key=lambda m: m.get("tool_call_id", ""))

        # Reconstruct: other messages first, then sorted tool messages
        # (typically there are no other messages mixed with tool results)
        return other_messages + tool_messages

    def format_prompt(
        self,
        messages: list[OpenAIMessage],
        tools: list[dict] | None = None,
    ) -> str:
        """Format messages into a prompt ready for model generation.

        Applies the HuggingFace chat template with `add_generation_prompt=True`,
        which appends the assistant turn prefix for the model to continue.
        """
        chat_messages = [self._format_message_for_template(m) for m in messages]

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        # Only pass enable_thinking if explicitly set (for Qwen3 hybrid models)
        if self._enable_thinking is not None:
            kwargs["enable_thinking"] = self._enable_thinking

        if tools:
            kwargs["tools"] = tools

        return self.tokenizer.apply_chat_template(chat_messages, **kwargs)

    def tokenize_prompt_messages(
        self,
        messages: list[OpenAIMessage],
    ) -> list[int] | None:
        """Tokenize prompt messages for the next generation call.

        First call: tokenizes full prompt with tools.
        Subsequent calls: tokenizes only new messages (tool results),
        prepending the message separator to align with chat template formatting.

        Returns:
            Token IDs for new prompt tokens, or None if no new messages.
        """
        # First call: full prompt with tools
        if len(self.token_manager) == 0:
            formatted = self.format_prompt(messages, tools=self._current_tools)
            return self.tokenizer.encode(formatted, add_special_tokens=False)

        # Subsequent calls: only new messages
        if len(messages) > self._processed_message_count:
            new_messages = self._sort_tool_results(messages[self._processed_message_count:])
            formatted = self.format_prompt(new_messages)

            # Prepend message separator to align with chat template
            if self.tool_call_parser:
                formatted = self.tool_call_parser.message_separator + formatted

            return self.tokenizer.encode(formatted, add_special_tokens=False)

        return None

    # -------------------------------------------------------------------------
    # Tool formatting
    # -------------------------------------------------------------------------

    def _format_tools_for_template(self, tools: list[dict[str, Any]]) -> list[dict]:
        """Format CAMEL tools for chat template.

        CAMEL tools are already in OpenAI format, just pass through.
        """
        return tools

    # -------------------------------------------------------------------------
    # Response construction
    # -------------------------------------------------------------------------

    def _extract_logprobs(self, response: dict[str, Any], key: str) -> list[float] | None:
        """Extract logprobs from SGLang response."""
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
        """Build a ChatCompletion response object."""
        message = ChatCompletionMessage(
            role="assistant",
            content=text if text else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason,  # type: ignore
        )

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

    # -------------------------------------------------------------------------
    # Model interface implementation
    # -------------------------------------------------------------------------

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
        """Synchronous inference - not implemented for TITO model."""
        raise NotImplementedError("Use _arun for async inference with TITO support")

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
        """Async inference with TITO (token-in/token-out) support.

        Tokenizes messages, calls SGLang /generate, and updates
        the TITO trajectory with input/output tokens and logprobs.

        Args:
            messages: Message list in OpenAI format.
            response_format: Not supported.
            tools: Tool schemas in OpenAI format.

        Returns:
            ChatCompletion with the model's response.
        """
        # Format tools (only on first call)
        if tools and not self._current_tools:
            self._current_tools = self._format_tools_for_template(tools)
            logger.debug(f"Tools formatted: {len(self._current_tools)} tools")

        # Prepare request
        sampling_params: dict[str, Any] = dict(self.model_config_dict)
        new_input_tokens = self.tokenize_prompt_messages(messages)

        # Token IDs tracked in token_manager to ensure token-in feature
        input_ids = self.token_manager.token_ids + (new_input_tokens or [])

        # Don't pass logprob_start_len to enable KV caching via RadixCache
        # Setting it to 0 forces full recomputation and breaks caching
        # RadixCache automatically matches prefixes without needing rid

        try:
            response = await self._client.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=self._return_logprobs,
                logprob_start_len=None if self.compute_logprobs_for_new_tokens_only else 0,
            )

            # Extract response data
            text = response.get("text", "")
            output_ids = response.get("output_ids", [])
            output_logprobs = self._extract_logprobs(response, "output_token_logprobs")
            input_logprobs = self._extract_logprobs(response, "input_token_logprobs")
            meta_info = response.get("meta_info", {})

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_text = e.response.text.lower()

            # Context/prompt length exceeded
            if status == 400:
                length_patterns = ["exceed", "too long", "max model len", "maximum length", "context length"]
                if any(p in error_text for p in length_patterns):
                    raise RuntimeError(f"Context length exceeded: {e.response.text}") from e
                logger.warning(f"Unexpected 400 error: {e.response.text}")

            raise

        # Update TITO trajectory
        if new_input_tokens:
            new_input_logprobs = input_logprobs[-len(new_input_tokens):] if input_logprobs else None
            self.token_manager.add_prompt(token_ids=new_input_tokens, logprobs=new_input_logprobs)

        if output_ids:
            self.token_manager.add_response(token_ids=output_ids, logprobs=output_logprobs)

        self._processed_message_count = len(messages) + 1

        # Parse tool calls from output text
        parsed_tool_calls = self.tool_call_parser.parse(text)
        openai_tool_calls: list[ChatCompletionMessageToolCall] | None = None

        if parsed_tool_calls:
            openai_tool_calls = []
            for tc in parsed_tool_calls:
                if tc.is_error:
                    logger.warning(f"Tool parse error for '{tc.name}': {(tc.raw or '')[:100]}")
                    self.tool_parse_errors[tc.name] = self.tool_parse_errors.get(tc.name, 0) + 1

                openai_tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tc.id,
                        type="function",
                        function=Function(
                            name=tc.name,
                            arguments=tc.payload,
                        ),
                    )
                )

        # Determine finish reason
        finish_reason = "tool_calls" if parsed_tool_calls else "stop"
        if meta_info and isinstance(meta_info.get("finish_reason"), dict):
            if meta_info["finish_reason"].get("type") == "length":
                finish_reason = "length"

        # Extract usage
        prompt_tokens = int(meta_info.get("prompt_tokens") or len(input_ids))
        completion_tokens = int(meta_info.get("completion_tokens") or len(output_ids))

        return self._build_chat_completion(
            text=text,
            tool_calls=openai_tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Initialize the token counter for the model backend."""
        if not self._token_counter:
            self._token_counter = TokenCounter(self.tokenizer)
        return self._token_counter

    @property
    def stream(self) -> bool:
        """Returns whether the model is in stream mode.

        TITO model always uses non-streaming for better parallelism.
        """
        return False
