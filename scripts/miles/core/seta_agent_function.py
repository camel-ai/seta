"""
Custom agent function for miles agentic_tool_call.generate.

Dispatches to a seta_env env_service (POST /step) and returns
env metadata as a plain dict. The generate layer merges this into
sample.metadata so downstream reward models can extract reward, eval
reports, etc.

Mirrors swe_agent_function.py from miles/examples/experimental/swe-agent-v2
but targets env_service instead of Harbor.

Environment variables:
    AGENT_SERVER_URL    env_service URL (default: http://localhost:8002)
    DATASET_NAME        dataset name for task dir resolution (default: seta-env-v2)
    TRIAL_NAME          trial name for organizing logs (default: "")
    OUTPUT_ROOT         local folder to dump per-trajectory results (default: outputs/miles_rollout)
    MILES_ROUTER_EXTERNAL_HOST  hostname rewrite for agent containers
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from miles.utils.http_utils import post

logger = logging.getLogger(__name__)


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run a single task instance via seta_env env_service."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    # Unified with generate_with_camel's env var names (CAMEL_*) so the launcher
    # forwards a single set; fall back to the standalone names for direct/test use.
    agent_server_url = os.getenv("CAMEL_ENV_SERVICE_URL") or os.getenv("AGENT_SERVER_URL", "http://localhost:8002")
    dataset_name = os.getenv("CAMEL_DATASET_NAME") or os.getenv("DATASET_NAME", "seta-env-v2")
    trial_name = os.getenv("CAMEL_TRIAL_NAME") or os.getenv("TRIAL_NAME", "")
    output_root = os.getenv("OUTPUT_ROOT", "outputs/miles_rollout")

    # Rewrite hostname if MILES_ROUTER_EXTERNAL_HOST is set
    # (so env_service can reach Miles Router from its network context)
    session_url = base_url
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        port = parsed.port
        netloc = f"{external_host}:{port}" if port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))

    # The miles router expects POST /sessions/{id}/v1/chat/completions.
    # The OpenAI client appends /chat/completions to base_url, so base_url
    # must end with /v1 for the full path to match the router route.
    if not session_url.endswith("/v1"):
        session_url = session_url.rstrip("/") + "/v1"

    # metadata may arrive as a JSON string from parquet — parse it if needed
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    metadata = metadata or {}

    # Extract task_name and session_id
    task_name = metadata.get("instance_id", "")
    # base_url format: http://host:port/sessions/{session_id}
    session_id = ""
    if "/sessions/" in base_url:
        session_id = base_url.split("/sessions/")[1].split("/")[0]

    uid = f"{task_name}_{session_id}" if session_id else task_name
    traj_i = metadata.get("index", 0)

    # Build StepRequest for env_service /step endpoint
    request = {
        "task": {
            "task_name": task_name,
            "instruction": prompt if isinstance(prompt, str) else str(prompt),
        },
        "uid": uid,
        "traj_i": traj_i,
        "model_url": session_url,
        "model_api_key": "dummy",
        "dataset_name": dataset_name,
        "task_name": task_name,
        "trial_name": trial_name,
    }

    logger.info(
        f"[seta_agent] task={task_name} uid={uid} "
        f"server={agent_server_url} session={session_url}"
    )

    try:
        response = await asyncio.wait_for(
            post(f"{agent_server_url}/step", request),
            timeout=3600,  # 1 hour max per trial
        )
    except asyncio.TimeoutError:
        logger.error(f"env_service call timed out after 3600s for {task_name}")
        return None
    except asyncio.CancelledError:
        logger.warning(f"env_service call cancelled for {task_name}")
        return None
    except Exception as e:
        logger.error(f"env_service call failed for {task_name}: {e}")
        return None

    # Extract results
    run_info = response.get("run_info") or {}
    reward = response.get("reward")
    error = response.get("error")

    if error:
        logger.warning(f"env_service returned error for {task_name}: {error}")

    # Dump results to local output folder for debugging
    # try:
    #     out_dir = Path(output_root)
    #     if trial_name:
    #         out_dir = out_dir / trial_name
    #     out_dir = out_dir / uid
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     with open(out_dir / "result.json", "w") as f:
    #         json.dump(response, f, indent=2, default=str)
    # except Exception as e:
    #     logger.warning(f"Failed to dump result for {uid}: {e}")

    # Build agent_metrics from run_info
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

    # Determine exit status
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
