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

"""SGLang HTTP client with connection pooling and retry logic.

Aligned with Slime's http_utils.py for RL training stability:
- Aggressive retry (60 attempts by default)
- Retries on all transient errors (like Slime)
- Infinite timeout by default for long generations
- Non-streaming POST for better parallelism (no SSE overhead)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# OpenAI's default connection limit (from openai/_constants.py)
DEFAULT_MAX_CONNECTIONS = 1000

# Non-retryable HTTP status codes
#
# Reference: OpenAI Python SDK (_base_client.py) retries: 408, 409, 429, 5xx
# Reference: Slime (http_utils.py) retries ALL errors for local SGLang servers
#
# Our hybrid approach for local SGLang during RL training:
# - 401/403/404: Don't retry (auth/routing errors won't self-resolve)
# - 400 with context length error: Don't retry (prompt too long won't fix itself)
# - 400 other: Retry (transient for local servers - weight reloading, memory pressure)
# - 408/409/429/5xx: Retry (same as OpenAI SDK)
# - Connection errors: Retry (same as OpenAI SDK)
NON_RETRYABLE_STATUS_CODES = {401, 403, 404}  # Auth failed, forbidden, endpoint not found


class SGLangClient:
    """Async HTTP client for SGLang server with connection pooling and retry.

    Designed for RL training stability with aggressive retry on transient errors.
    Aligned with Slime's http_utils.py approach.

    Uses non-streaming POST requests for better parallelism in high-concurrency
    training scenarios (no SSE overhead, connections released immediately).

    Example:
        >>> async with SGLangClient("http://localhost:30000") as client:
        ...     result = await client.generate(input_ids=[1, 2, 3])
        ...     print(result["text"])

        >>> # For RL training with infinite timeout (like Slime):
        >>> client = SGLangClient("http://localhost:30000", timeout=None)

        >>> # From Slime training args:
        >>> client = SGLangClient.from_slime_args(args)
    """

    def __init__(
        self,
        base_url: str,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        timeout: float | None = None,
        connect_timeout: float = 5.0,
        max_retries: int = 60,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize SGLang client.

        Args:
            base_url: SGLang server URL (e.g., "http://localhost:30000").
            max_connections: Maximum concurrent connections (default: 1000).
            timeout: Request timeout in seconds, or None for infinite (default: None, like Slime).
            connect_timeout: TCP connection timeout in seconds (default: 5s).
            max_retries: Maximum retry attempts on transient errors (default: 60, like Slime).
            retry_delay: Delay between retries in seconds (default: 1.0).
        """
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # timeout=None means infinite wait (like Slime's httpx.Timeout(None))
        http_timeout = httpx.Timeout(timeout, connect=connect_timeout) if timeout else httpx.Timeout(None)

        # FIX: Set max_keepalive_connections equal to max_connections to avoid connection churn
        # By default httpx only keeps 20 connections alive, which can cause issues with high concurrency
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=http_timeout,
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )

        logger.info(
            f"SGLangClient initialized: base_url={self.base_url}, "
            f"max_connections={max_connections}, max_keepalive={max_connections}, "
            f"timeout={timeout}, max_retries={max_retries}"
        )

    @classmethod
    def from_slime_args(cls, args: Any, **overrides: Any) -> SGLangClient:
        """Create SGLangClient from Slime's training args.

        Matches Slime's [`init_http_client`](https://github.com/THUDM/slime/blob/main/slime/utils/http_utils.py) formula:

            max_connections = concurrency * (num_gpus / gpus_per_engine)

        where `num_gpus ÷ gpus_per_engine` is the number of SGLang server instances,
        and `concurrency` is the max concurrent requests each instance can handle.
        This ensures enough connections to fully saturate all server instances.

        Args:
            args: Slime's args namespace with required attributes:
                - sglang_router_ip: SGLang router IP address
                - sglang_router_port: SGLang router port
                - sglang_server_concurrency: Concurrency per SGLang server
                - rollout_num_gpus: Total GPUs for rollout
                - rollout_num_gpus_per_engine: GPUs per engine
            **overrides: Override any configuration values.

        Returns:
            Configured SGLangClient instance.

        Example:
            >>> client = SGLangClient.from_slime_args(args)
            >>> model = SGLangModel(tokenizer=tokenizer, client=client)
        """
        return cls(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            # Matches Slime's init_http_client formula
            max_connections=args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine,
            **overrides,
        )

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        await self._client.aclose()

    async def __aenter__(self) -> SGLangClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.close()

    def _is_retryable_error(self, e: Exception) -> bool:
        """Check if an error is retryable.

        Aligned with Slime's philosophy: retry aggressively on most errors.
        For local SGLang servers, most 400 errors are transient (weight reloading, memory pressure).

        Non-retryable:
        - 401/403/404: Auth/routing errors that won't self-resolve
        - 400 with "context length exceeded": Prompt too long, retrying won't help
        """
        if isinstance(e, httpx.HTTPStatusError):
            status = e.response.status_code
            # Don't retry auth/routing errors
            if status in NON_RETRYABLE_STATUS_CODES:
                return False
            # Don't retry context length errors (retrying won't help)
            if status == 400:
                error_text = e.response.text.lower()
                length_patterns = ["exceed", "too long", "max model len", "maximum length", "context length"]
                if any(p in error_text for p in length_patterns):
                    return False
            # Retry everything else: 5xx, 408, 429, other 400s, etc.
            return True
        # Retry all connection/timeout errors
        return True

    async def generate(
        self,
        input_ids: list[int],
        *,
        model: str | None = None,
        sampling_params: dict[str, Any] | None = None,
        return_logprob: bool = True,
        logprob_start_len: int | None = None,
        return_routed_experts: bool = False,
        return_indexer_topk: bool = False,
    ) -> dict[str, Any]:
        """Generate from SGLang `/generate` endpoint.

        Uses non-streaming POST for better parallelism in RL training.
        Connections are released immediately after response, enabling
        high concurrency without SSE overhead.

        Args:
            input_ids: Input token IDs.
            model: Optional model identifier.
            sampling_params: Sampling parameters (temperature, top_p, max_new_tokens, etc.).
            return_logprob: Whether to return log probabilities (default: True).
            logprob_start_len: Start position for logprob computation (default: 0).

        Returns:
            Response dict with text, output_ids, meta_info (logprobs, finish_reason, etc.).

        Raises:
            httpx.HTTPStatusError: For non-retryable errors (401, 403, 404, context length exceeded)
                or after all retries exhausted. Other 400 errors are retried.
            httpx.ConnectError: For connection failures after retries exhausted.
        """
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "stream": False,  # Always non-streaming for RL training
        }

        if model:
            payload["model"] = model

        if sampling_params:
            payload["sampling_params"] = sampling_params

        if return_logprob:
            payload["return_logprob"] = True
            if logprob_start_len is not None:
                payload["logprob_start_len"] = logprob_start_len

        if return_routed_experts:
            payload["return_routed_experts"] = True
        if return_indexer_topk:
            payload["return_indexer_topk"] = True

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post("/generate", json=payload)
                response.raise_for_status()
                return response.json()

            except Exception as e:
                last_error = e

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    raise  # Non-retryable error (e.g., 400 Bad Request)

                # Log and retry
                response_text = e.response.text if isinstance(e, httpx.HTTPStatusError) else None
                if attempt < self.max_retries:
                    logger.warning(
                        f"SGLang request failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{type(e).__name__}: {e}, response={response_text}. Retrying in {self.retry_delay}s..."
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"SGLang request failed after {self.max_retries + 1} attempts: "
                        f"{type(e).__name__}: {e}, response={response_text}"
                    )
                    raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error

        # This should never happen, but satisfies type checker
        raise RuntimeError("Unexpected state in generate()")

    async def health(self) -> bool:
        """Check if SGLang server is healthy.

        Returns:
            True if server responds to `/health` endpoint, False otherwise.
        """
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_model_info(self) -> dict[str, Any]:
        """Get model information from the SGLang server.

        Returns:
            Dict containing model info from `/get_model_info` endpoint. Important fields include:
            - model_path: HuggingFace model ID or local path
            - tokenizer_path: Tokenizer path (may differ from model_path)
        """
        response = await self._client.get("/get_model_info")
        response.raise_for_status()
        return response.json()
