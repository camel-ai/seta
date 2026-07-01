# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang native `/generate` API model provider for token-in/token-out training.

This provider uses SGLang's native HTTP APIs:
- `/generate` for text generation (returns output_ids directly)

It uses a HuggingFace tokenizer for:
- Applying chat templates (via tokenizer.apply_chat_template())
- Tokenizing prompts and tool results

This eliminates retokenization drift in RL training by maintaining token IDs
throughout the rollout instead of converting text back to tokens.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterable,
    Iterator,
    Type,
    TypedDict,
    cast,
)

import httpx
from pydantic import BaseModel
from strands.models import Model
from strands.models.openai import OpenAIModel
from strands.types.content import Messages, SystemContentBlock
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import Unpack, override

from .client import SGLangClient
from .token import TokenManager
from .tool_parser import HermesToolCallParser, ToolCallParser, ToolCallParseResult

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class SGLangModel(Model):
    """SGLang native `/generate` API provider with token-in/token-out (TITO) support.

    Uses a HuggingFace tokenizer for chat template formatting and SGLang's
    `/generate` endpoint for generation. Tracks token trajectories via `TokenManager`.

    Attributes:
        tokenizer: HuggingFace tokenizer for encoding/decoding.
        token_manager: Tracks tokens, logprobs, and masks for TITO training.
        tool_call_parser: Parser for extracting tool calls from model output.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
        >>> model = SGLangModel(tokenizer=tokenizer, base_url="http://localhost:30000")
        >>> # After generation:
        >>> model.token_manager.token_ids    # Full token trajectory
        >>> model.token_manager.loss_mask    # Boolean mask for loss computation
        >>> model.token_manager.logprobs     # Log probabilities
    """

    class SGLangConfig(TypedDict, total=False):
        """Configuration options for SGLang native API."""

        base_url: str  # SGLang server URL (default: http://localhost:30000)
        model_id: str | None  # Optional model identifier
        params: dict[str, Any] | None  # Default sampling parameters
        timeout: float | None  # Request timeout in seconds, or None for infinite (default: None, like SLIME)
        return_logprobs: bool  # Return logprobs for all tokens (default: True)
        enable_thinking: bool | None  # Enable thinking mode for Qwen3 hybrid models (default: None = auto)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tool_call_parser: ToolCallParser | None = None,
        client: SGLangClient | None = None,
        **model_config: Unpack[SGLangConfig],
    ) -> None:
        """Initialize SGLang model provider.

        Args:
            tokenizer: HuggingFace tokenizer for chat template and tokenization.
            tool_call_parser: Parser for tool calls (default: HermesToolCallParser).
            client: Optional `SGLangClient` for connection pooling and retry logic.
                    If `None`, creates a new ephemeral client per-request (not recommended for high concurrency like RL training).
            **model_config: See SGLangConfig for available options.
        """
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser or HermesToolCallParser()

        # HTTP client setup
        base_url = str(model_config.get("base_url") or "http://localhost:30000").rstrip("/")
        timeout = model_config.get("timeout")  # None = infinite, like SLIME

        self._client = client
        self._base_url = base_url
        self._timeout = timeout

        # Thinking mode for Qwen3 hybrid models (default: None = don't pass, let template decide)
        # Set explicitly to True/False only for models that support enable_thinking parameter
        self._enable_thinking: bool | None = model_config.get("enable_thinking")

        # Store config
        self.config = dict(model_config)
        self.config["base_url"] = base_url

        # TITO state
        self.token_manager = TokenManager()
        self._processed_message_count: int = 0
        self._current_tools: list[dict] | None = None

        # Parse error tracking (per tool name)
        self.tool_parse_errors: dict[str, int] = {}

        logger.debug(f"initialized with config: {self.config}")

    def _get_client(self) -> SGLangClient:
        """Get SGLangClient - shared if provided, otherwise create new."""
        if self._client is not None:
            return self._client
        # Create per-request client (not ideal for high concurrency)
        return SGLangClient(self._base_url, timeout=self._timeout)

    def reset(self) -> None:
        """Reset token accumulation for a new episode.

        Call this at episode start. Clears all accumulated tokens and resets
        internal state for tool tracking.
        """
        self.token_manager.reset()
        self._processed_message_count = 0
        self._current_tools = None
        self.tool_parse_errors = {}

    # -------------------------------------------------------------------------
    # Model interface implementation
    # -------------------------------------------------------------------------

    @override
    def update_config(self, **model_config: Unpack[SGLangConfig]) -> None:  # type: ignore[override]
        """Update the model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        if "base_url" in model_config and model_config["base_url"]:
            self.config["base_url"] = str(model_config["base_url"]).rstrip("/")
        self.config.update(model_config)

    @override
    def get_config(self) -> SGLangConfig:
        """Get the model configuration.

        Returns:
            The model configuration dict.
        """
        return cast(SGLangModel.SGLangConfig, self.config)

    # -------------------------------------------------------------------------
    # Chat template and message formatting
    # -------------------------------------------------------------------------

    @classmethod
    def _format_message_content(cls, message: dict[str, Any]) -> None:
        """Format a single message's content for chat templates.

        Flattens content arrays and preserves raw content including tool call
        markup to maintain exact generation order for TITO reconstruction.
        Modifies the message in-place.
        """
        # Flatten content from [{"text": "..."}, ...] to "..."
        # Find first text block (content array may have other block types)
        if "content" in message and isinstance(message["content"], list):
            text_content = ""
            for block in message["content"]:
                if "text" in block:
                    text_content = block["text"]
                    break
            message["content"] = text_content

        # Remove strands-processed tool_calls field and let the chat template handle it.
        if "tool_calls" in message:
            del message["tool_calls"]

    @classmethod
    def format_request_messages(cls, messages: Messages, system_prompt: str | None = None) -> list[dict[str, Any]]:
        """Convert strands Messages to OpenAI format for chat templates.

        Uses strands' OpenAIModel formatter and flattens content
        for compatibility with HuggingFace apply_chat_template.
        """
        result = OpenAIModel.format_request_messages(messages=messages, system_prompt=system_prompt)

        for message in result:
            cls._format_message_content(message)

        return result

    def _format_tools(self, tool_specs: list[ToolSpec]) -> list[dict]:
        """Format strands ToolSpecs to OpenAI format for chat templates."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.get("name", ""),
                    "description": spec.get("description", ""),
                    "parameters": spec.get("inputSchema", {}),
                },
            }
            for spec in tool_specs
        ]

    def format_prompt(
        self,
        messages: Messages,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """Format messages into a prompt ready for model generation.

        Applies the HuggingFace chat template with `add_generation_prompt=True`,
        which appends the assistant turn prefix for the model to continue.

        For Qwen3 hybrid thinking models, `enable_thinking` controls whether
        the model uses its internal reasoning mode. Only passed to template
        when explicitly set (not None) to avoid affecting non-Qwen3 models.

        The result is manually tokenized (not model-generated) and added to
        the TITO trajectory with `loss_mask=False`.
        """
        chat_messages = self.format_request_messages(messages, system_prompt)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Only pass enable_thinking if explicitly set (for Qwen3 hybrid models)
        # This avoids affecting non-Qwen3 models that don't support this parameter
        if self._enable_thinking is not None:
            kwargs["enable_thinking"] = self._enable_thinking
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(chat_messages, **kwargs)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def tokenize_prompt_messages(
        self,
        messages: Messages,
        system_prompt: str | None,
    ) -> list[int] | None:
        """Tokenize prompt messages for the next generation call.

        First call: tokenizes full prompt with system prompt and tools.
        Subsequent calls: tokenizes only new messages (tool results, user messages),
        prepending the message separator to align with chat template formatting.
        """
        # First call: full prompt with tools
        if len(self.token_manager) == 0:
            formatted = self.format_prompt(messages, system_prompt, tools=self._current_tools)
            return self.tokenizer.encode(formatted, add_special_tokens=False)

        # Subsequent calls: only new messages
        if len(messages) > self._processed_message_count:
            new_messages = self._sort_tool_results(messages[self._processed_message_count :])
            formatted = self.format_prompt(new_messages)

            # Prepend message separator to align with chat template.
            # The model generates up to <|im_end|>, but the chat template adds
            # a separator (e.g., "\n") before the next <|im_start|>.
            if self.tool_call_parser:
                formatted = self.tool_call_parser.message_separator + formatted

            return self.tokenizer.encode(formatted, add_special_tokens=False)

        return None

    def _sort_tool_results(self, messages: Messages) -> Messages:
        """Sort tool results by ID to match original call order (IDs are sequential: call_0000, call_0001, ...)."""
        result = []
        for msg in messages:
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                result.append(msg)
                continue
            content = msg["content"]
            tool_results = [b for b in content if isinstance(b, dict) and "toolResult" in b]
            if not tool_results:
                result.append(msg)
                continue
            other = [b for b in content if not (isinstance(b, dict) and "toolResult" in b)]
            tool_results.sort(key=lambda b: b.get("toolResult", {}).get("toolUseId", ""))
            result.append({**msg, "content": other + tool_results})
        return result

    def _yield_tool_use_events(
        self,
        tool_calls: list[ToolCallParseResult],
    ) -> Iterator[StreamEvent]:
        """Yield toolUse stream events for parsed tool calls.

        Each tool call emits three events following the Strands streaming protocol:
        - `contentBlockStart`: begins block with toolUseId and name
        - `contentBlockDelta`: contains the tool input (delta = incremental data)
        - `contentBlockStop`: ends the block
        """
        for tool_call in tool_calls:
            if tool_call.is_error:
                logger.warning(f"Tool parse error for '{tool_call.name}': {(tool_call.raw or '')[:100]}")
                # Track parse error count per tool name
                self.tool_parse_errors[tool_call.name] = self.tool_parse_errors.get(tool_call.name, 0) + 1

            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": tool_call.name,
                        }
                    }
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {
                        "toolUse": {
                            "input": tool_call.payload,
                        }
                    }
                }
            }
            yield {"contentBlockStop": {}}

    def _extract_logprobs(self, event: dict[str, Any], key: str) -> list[float] | None:
        """Extract logprobs from SGLang event (format: [[logprob, token_id, ...], ...])."""
        meta_info = event.get("meta_info", {})
        logprobs = meta_info.get(key) or event.get(key)
        if isinstance(logprobs, list) and logprobs:
            return [entry[0] for entry in logprobs]
        return None

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream generation from SGLang /generate endpoint.

        Tokenizes messages, calls /generate, streams text deltas, and updates
        the TITO trajectory with input/output tokens and logprobs.
        """
        # Format tools (only on first call)
        if tool_specs and not self._current_tools:
            self._current_tools = self._format_tools(tool_specs)
            logger.debug(f"tools formatted: {len(self._current_tools)} tools")

        # Prepare request
        config = self.get_config()
        sampling_params: dict[str, Any] = dict(config.get("params") or {})
        return_logprobs = config.get("return_logprobs", True)
        new_input_tokens = self.tokenize_prompt_messages(messages, system_prompt)
        # Tracking token IDs in token_manager to ensure the token-in feature
        input_ids = self.token_manager.token_ids + (new_input_tokens or [])

        # Start message
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}

        # Call SGLangClient (non-streaming POST for better parallelism)
        client = self._get_client()
        ephemeral_client = self._client is None  # Ephemeral clients are closed after use

        try:
            response = await client.generate(
                input_ids=input_ids,
                model=config.get("model_id"),
                sampling_params=sampling_params,
                return_logprob=return_logprobs,
            )

            # Extract response data
            text = response.get("text", "")
            output_ids = response.get("output_ids", [])
            output_logprobs = self._extract_logprobs(response, "output_token_logprobs")
            input_logprobs = self._extract_logprobs(response, "input_token_logprobs")
            meta_info = response.get("meta_info", {})

            # Yield text as single delta (non-streaming gives complete text at once)
            if text:
                yield {"contentBlockDelta": {"delta": {"text": text}}}

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_text = e.response.text.lower()
            # Context/prompt length exceeded (various SGLang error patterns)
            if status == 400:
                length_patterns = ["exceed", "too long", "max model len", "maximum length", "context length"]
                if any(p in error_text for p in length_patterns):
                    raise ContextWindowOverflowException(f"Context length exceeded: {e.response.text}") from e
                # Log unexpected 400 errors for debugging
                logger.warning(f"Unexpected 400 error: {e.response.text}")
            # Rate limiting / service unavailable
            if status in (429, 503):
                raise ModelThrottledException(f"Service throttled (status={status}): {e.response.text}") from e
            raise  # Re-raise other HTTP errors
        finally:
            if ephemeral_client:
                await client.close()

        # Update TITO trajectory
        if new_input_tokens:
            new_input_logprobs = input_logprobs[-len(new_input_tokens) :] if input_logprobs else None
            self.token_manager.add_prompt(token_ids=new_input_tokens, logprobs=new_input_logprobs)
        if output_ids:
            self.token_manager.add_response(token_ids=output_ids, logprobs=output_logprobs)
        self._processed_message_count = len(messages) + 1

        # End text block, start tool use blocks if there are any tool calls
        yield {"contentBlockStop": {}}

        # Parse tool calls and yield events
        parsed_tool_calls = self.tool_call_parser.parse(text)
        for event in self._yield_tool_use_events(parsed_tool_calls):
            yield event

        # Determine stop reason
        stop_reason: str = "tool_use" if parsed_tool_calls else "end_turn"
        if meta_info and isinstance(meta_info.get("finish_reason"), dict):
            if meta_info["finish_reason"].get("type") == "length":
                stop_reason = "max_tokens"

        yield {"messageStop": {"stopReason": cast(Any, stop_reason)}}

        # Yield usage metadata
        if meta_info:
            prompt_tokens = int(meta_info.get("prompt_tokens") or 0)
            completion_tokens = int(meta_info.get("completion_tokens") or 0)
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": prompt_tokens,
                        "outputTokens": completion_tokens,
                        "totalTokens": prompt_tokens + completion_tokens,
                    },
                    "metrics": {"latencyMs": int(float(meta_info.get("e2e_latency") or 0) * 1000)},
                }
            }

    @override
    async def structured_output(
        self,
        output_model: Type[BaseModel],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, BaseModel | Any], None]:
        """Not implemented for training-only model.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("structured_output is not supported by SGLangModel (training-only)")
        # Make this a generator to satisfy the type signature
        yield {}  # pragma: no cover
