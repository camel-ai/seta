"""Miles agent function that runs one trajectory on the seta env_service.

    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path common.env_service_agent.run

Miles' agentic generate function opens a session on the Miles session server and
calls ``run`` with that session's URL. ``run`` posts the task to the env_service
(``POST /step``); the env_service starts the task's sandbox, runs seta's CAMEL agent
against the session URL (OpenAI-compatible), verifies the result and returns the
reward. The session server records the exact tokens of every model call, so training
sees what the model generated; this module only moves the task and the reward.

The returned dict is merged into the sample metadata, where
``common.env_service_reward.reward_func`` reads ``reward``.

Environment (forwarded to the Ray rollout workers by the launcher):
    CAMEL_ENV_SERVICE_URL       env_service base URL (fallback AGENT_SERVER_URL,
                                default http://localhost:8002)
    CAMEL_DATASET_NAME          env_service resolves a task as
                                DATASET_ROOT/<CAMEL_DATASET_NAME>/<instance_id>
                                (fallback DATASET_NAME, default seta-env-v2)
    CAMEL_TRIAL_NAME            groups the trial folders of one run
                                (fallback TRIAL_NAME, default empty)
    MILES_ROUTER_EXTERNAL_HOST  rewrites the host of the session URL, for an env_service
                                that reaches the session server under another name
"""

import asyncio
import json
import logging
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

# Client-side bound on one /step call. A trajectory still running after this is
# dropped (no reward) even if the env_service would let it finish.
STEP_CALL_TIMEOUT_SEC = 3600


def prompt_to_instruction(prompt: Any) -> str:
    """Task instruction from a Miles sample prompt.

    Without --apply-chat-template the prompt is the dataset's ``prompt`` field
    verbatim: either the instruction text or a chat list such as
    ``[{"role": "user", "content": "<instruction>"}]`` (as written by
    common/build_prompt_dataset.py). The env_service builds its own conversation,
    so only the instruction text is sent.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        return "\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt)


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one task instance on the env_service; None on a failed call."""
    metadata = metadata or {}

    agent_server_url = os.getenv("CAMEL_ENV_SERVICE_URL") or os.getenv("AGENT_SERVER_URL", "http://localhost:8002")
    dataset_name = os.getenv("CAMEL_DATASET_NAME") or os.getenv("DATASET_NAME", "seta-env-v2")
    trial_name = os.getenv("CAMEL_TRIAL_NAME") or os.getenv("TRIAL_NAME", "")

    # Let the env_service reach the session server from its own network.
    session_url = base_url
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        port = parsed.port
        netloc = f"{external_host}:{port}" if port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))

    # The session server serves POST /sessions/{id}/v1/chat/completions and the
    # OpenAI client appends /chat/completions to its base URL, so the base URL must
    # end with /v1.
    if not session_url.endswith("/v1"):
        session_url = session_url.rstrip("/") + "/v1"

    # Parquet datasets may store metadata as a JSON string.
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    metadata = metadata or {}

    task_name = metadata.get("instance_id", "")
    # base_url is http://host:port/sessions/{session_id}
    session_id = ""
    if "/sessions/" in base_url:
        session_id = base_url.split("/sessions/")[1].split("/")[0]

    uid = f"{task_name}_{session_id}" if session_id else task_name
    traj_i = metadata.get("index", 0)

    request = {
        "task": {
            "task_name": task_name,
            "instruction": prompt_to_instruction(prompt),
        },
        "uid": uid,
        "traj_i": traj_i,
        "model_url": session_url,
        "model_api_key": "dummy",
        "dataset_name": dataset_name,
        "task_name": task_name,
        "trial_name": trial_name,
    }

    logger.info(f"[env_service_agent] task={task_name} uid={uid} server={agent_server_url} session={session_url}")

    try:
        response = await asyncio.wait_for(
            post(f"{agent_server_url}/step", request),
            timeout=STEP_CALL_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.error(f"env_service call timed out after {STEP_CALL_TIMEOUT_SEC}s for {task_name}")
        return None
    except asyncio.CancelledError:
        logger.warning(f"env_service call cancelled for {task_name}")
        return None
    except Exception as e:
        logger.error(f"env_service call failed for {task_name}: {e}")
        return None

    run_info = response.get("run_info") or {}
    reward = response.get("reward")
    error = response.get("error")

    if error:
        logger.warning(f"env_service returned error for {task_name}: {error}")

    agent_metrics = {}
    timings = run_info.get("timings", {})
    agent_summary = run_info.get("agent_summary", {})
    if timings:
        flat_timings = {k: v for k, v in timings.items() if isinstance(v, (int, float))}
        if flat_timings:
            agent_metrics["total_time"] = sum(flat_timings.values())
            agent_metrics.update({f"stage_{k}": v for k, v in flat_timings.items()})
    if agent_summary:
        agent_metrics.update(agent_summary)

    error_info = run_info.get("error_info", {})
    if error_info:
        exit_status = error_info.get("stage", "AgentError")
    elif reward is not None:
        exit_status = "Submitted"
    else:
        exit_status = "Unknown"

    return {
        "reward": reward if reward is not None else 0.0,
        "exit_status": exit_status,
        "eval_report": run_info.get("evaluation", {}),
        "agent_metrics": agent_metrics,
    }
