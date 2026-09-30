"""Miles agent function for the Harbor agent server.

Used as ``--custom-agent-function-path common.harbor_agent.run`` together
with ``--custom-generate-function-path
miles.rollout.generate_hub.agentic_tool_call.generate``.

For every rollout sample, ``run`` posts one trial to the Harbor agent server
(``python -m harbor.agent_server``, ``POST /run``). Harbor starts the task's
sandbox (Daytona, Modal, GKE, Docker, ...), runs the agent against the Miles
session URL, verifies the result and returns the reward. The returned dict is
merged into ``sample.metadata``; ``common.harbor_rollout.reward_func`` reads the
reward from there.

``abort`` is registered as the Miles abort hook: when oversampling has filled a
batch it posts ``/flush`` so Harbor cancels the surplus in-flight trials.

Environment variables (all optional unless noted):
  AGENT_SERVER_URL                  Harbor agent server, default http://localhost:11000
  AGENT_MODEL_NAME                  model name sent to the agent, default "model"
  HARBOR_AGENT_NAME                 terminus-2 (default) | camel
  HARBOR_AGENT_MAX_ITERATIONS       agent turn cap, default 50
  HARBOR_MAX_SEQ_LEN                agent context bound in tokens
  HARBOR_AGENT_CALL_TIMEOUT_SEC     client deadline per trial (queue + sandbox + agent + verify)
  HARBOR_TERMINUS_PARSER            terminus-2 parser, e.g. xml | json
  HARBOR_INTERLEAVED_THINKING       terminus-2: keep reasoning between turns (true/false)
  HARBOR_TERMINUS_ENABLE_SUMMARIZE  terminus-2 context summarization (true/false)
  HARBOR_CAMEL_MAX_COMPACTIONS      camel context compactions (int)
  MILES_ROUTER_EXTERNAL_HOST        rewrite the session URL host the agent calls
"""

import asyncio
import logging
import os
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunparse

from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_CALL_TIMEOUT_SEC = 10800.0


def _optional_bool_env(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, or yes/no")


def _optional_non_negative_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        raise ValueError(f"{name} must be a non-negative integer") from None
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run a single task instance via the Harbor agent server."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    agent_server_url = os.getenv("AGENT_SERVER_URL", "http://localhost:11000")
    model_name = os.getenv("AGENT_MODEL_NAME", "model")

    session_url = f"{base_url}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        port = parsed.port
        netloc = f"{external_host}:{port}" if port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))

    # Keep the Miles training/session sample bound separate from Harbor's agent
    # context bound. The stock generator adds max_seq_len to metadata; do not let
    # that silently reduce a long Harbor trajectory to the training sample length.
    agent_metadata = {k: v for k, v in metadata.items() if k != "max_seq_len"}

    agent_name = os.getenv("HARBOR_AGENT_NAME", "terminus-2")
    sampling_params = {
        **request_kwargs,
        "max_iterations": int(os.getenv("HARBOR_AGENT_MAX_ITERATIONS", "50")),
    }
    if agent_name == "camel":
        # Preserve malformed/length-cut model output as terminal policy
        # behavior. Do not append corrective feedback or regenerate it.
        sampling_params["response_feedback"] = False
        sampling_params["max_response_feedback"] = 0
        # CAMEL's own context compaction. Harbor reads it from sampling_params
        # and pops it into agent kwargs before the rest reaches the model
        # config, so it stays agent policy rather than a sampling knob. Unset
        # sends nothing and leaves Harbor's default of 0, so runs that predate
        # this variable keep their recorded behavior.
        max_compactions = _optional_non_negative_int_env(
            "HARBOR_CAMEL_MAX_COMPACTIONS"
        )
        if max_compactions is not None:
            sampling_params["max_compactions"] = max_compactions
    elif agent_name == "terminus-2":
        parser_name = os.getenv("HARBOR_TERMINUS_PARSER")
        if parser_name:
            sampling_params["parser_name"] = parser_name
        interleaved_thinking = _optional_bool_env("HARBOR_INTERLEAVED_THINKING")
        if interleaved_thinking is not None:
            sampling_params["interleaved_thinking"] = interleaved_thinking
        # Terminus-2's own context summarization. Harbor reads it from
        # sampling_params and pops it before the rest reaches the model config,
        # so it stays agent policy rather than a sampling knob. Unset sends
        # nothing and leaves Harbor's default, so runs that predate this
        # variable keep their recorded behavior.
        enable_summarize = _optional_bool_env("HARBOR_TERMINUS_ENABLE_SUMMARIZE")
        if enable_summarize is not None:
            sampling_params["enable_summarize"] = enable_summarize

    request: dict[str, Any] = {
        **agent_metadata,
        "base_url": session_url,
        "model": f"openai/{model_name}",
        "sampling_params": sampling_params,
        "agent_name": agent_name,
        "max_seq_len": int(os.getenv("HARBOR_MAX_SEQ_LEN", "1048576")),
    }

    session_server_id = metadata.get("session_server_id")
    if session_server_id is not None:
        if external_host:
            port = urlsplit(f"http://{session_server_id}").port
            session_server_id = f"{external_host}:{port}"
        request["session_server_id"] = session_server_id

    session_server_instance_id = metadata.get("session_server_instance_id")
    if session_server_instance_id is not None:
        request["session_server_instance_id"] = session_server_instance_id

    agent_call_timeout_sec = _positive_float_env(
        "HARBOR_AGENT_CALL_TIMEOUT_SEC",
        _DEFAULT_AGENT_CALL_TIMEOUT_SEC,
    )
    try:
        response = await asyncio.wait_for(
            post(f"{agent_server_url}/run", request),
            timeout=agent_call_timeout_sec,
        )
    except asyncio.TimeoutError:
        # This is a total client-side deadline: Harbor queue wait, sandbox
        # startup, agent execution, and verification all consume it. Closing the
        # HTTP wait is not an explicit Harbor cancellation.
        logger.error(
            "Agent server call timed out after %ss",
            agent_call_timeout_sec,
        )
        return None
    except asyncio.CancelledError:
        logger.warning("Agent server call cancelled (sibling task failure?)")
        return None
    except Exception as e:
        logger.error(f"Agent server call failed: {e}")
        return None

    return {
        "reward": response.get("reward", 0.0),
        "exit_status": response.get("exit_status", ""),
        "eval_report": response.get("eval_report", {}),
        "agent_metrics": response.get("agent_metrics", {}),
    }


async def abort(args) -> None:
    """Cancel Harbor trials left in flight after Miles oversampling aborts."""
    agent_server_url = os.getenv("AGENT_SERVER_URL")
    if not agent_server_url:
        reason = "AGENT_SERVER_URL is unset"
        logger.error("[HARBOR-ABORT] FAILED: %s", reason)
        raise RuntimeError(reason)

    instance_ids = {
        instance_id
        for instance_id in (
            getattr(args, "session_server_instance_ids", None) or {}
        ).values()
        if instance_id
    }
    singular_id = getattr(args, "session_server_instance_id", None)
    if singular_id:
        instance_ids.add(singular_id)
    if not instance_ids:
        reason = (
            "Miles supplied neither session_server_instance_ids nor "
            "session_server_instance_id"
        )
        logger.error("[HARBOR-ABORT] FAILED: %s", reason)
        raise RuntimeError(reason)

    ordered_ids = sorted(instance_ids)
    results = await asyncio.gather(
        *(
            post(
                f"{agent_server_url.rstrip('/')}/flush",
                {"session_server_instance_id": instance_id},
                max_retries=3,
            )
            for instance_id in ordered_ids
        ),
        return_exceptions=True,
    )

    failures = []
    cancelled = 0
    for instance_id, result in zip(ordered_ids, results, strict=True):
        if isinstance(result, Exception):
            reason = f"instance={instance_id}: {type(result).__name__}: {result}"
            failures.append(reason)
            logger.error("[HARBOR-ABORT] FAILED: %s", reason)
            continue
        if not isinstance(result, dict) or not isinstance(result.get("cancelled"), int):
            reason = f"instance={instance_id}: invalid response {result!r}"
            failures.append(reason)
            logger.error("[HARBOR-ABORT] FAILED: %s", reason)
            continue
        instance_cancelled = result["cancelled"]
        cancelled += instance_cancelled
        logger.info(
            "[HARBOR-ABORT] FLUSHED instance=%s cancelled=%d",
            instance_id,
            instance_cancelled,
        )

    if failures:
        raise RuntimeError("; ".join(failures))
    logger.info(
        "[HARBOR-ABORT] COMPLETE instances=%d cancelled=%d",
        len(ordered_ids),
        cancelled,
    )
