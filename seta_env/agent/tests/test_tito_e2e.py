#!/usr/bin/env python3
"""End-to-end TITO test with miles-compatible session tracking.

Runs the TITO agent in a Docker container against sglang Qwen3-8B.
Mocks the OpenAI client to intercept each chat completion call and:
  1. Miles prepare_pretokenized (Qwen3TITOTokenizer.merge_tokens)
  2. sglang /generate with input_ids → REAL output token IDs
  3. Qwen25 tool call parser on raw text (same parser sglang uses)
  4. Miles update_pretokenized_state (prefix invariant with REAL IDs)
  5. Construct ChatCompletion for the agent to consume

This fully mimics the miles training pipeline: every token ID is real,
no retokenization of model output.

Usage:
    python seta_env/agent/tests/test_tito_e2e.py

Prerequisites:
    - sglang serving Qwen/Qwen3-8B on http://localhost:30000/v1
      (with --tool-call-parser qwen25)
    - Docker daemon running
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/home/ubuntu/miles")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Config ───────────────────────────────────────────────────────────────
TASK_DIR = REPO_ROOT / "dataset" / "seta-env-v2" / "stack_overflow__888"
OUTPUT_DIR = REPO_ROOT / "outputs" / "test_tito_e2e"
CAMEL_LOG_DIR = OUTPUT_DIR / "CAMEL_LOG_DIR"
TERMINAL_LOG_DIR = OUTPUT_DIR / "terminal_logs"

CONTAINER_NAME = "tito_e2e_888"
IMAGE_NAME = "tito_e2e_888_img"

MODEL_TYPE = "Qwen/Qwen3-8B"
SGLANG_URL = "http://localhost:30000"
MODEL_URL = f"{SGLANG_URL}/v1"
MODEL_CONFIG = {"max_tokens": 4096, "stream": False}
MAX_ITERATIONS = 10


# =====================================================================
# Docker
# =====================================================================

def docker_setup():
    env_dir = TASK_DIR / "environment"
    print(f"[docker] Building image ...")
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, str(env_dir)],
                   check=True, capture_output=True)
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER_NAME,
                    "--cpus", "1", "--memory", "2g",
                    IMAGE_NAME, "sleep", "infinity"], check=True)
    print(f"[docker] Container '{CONTAINER_NAME}' running.")


# =====================================================================
# Miles-compatible session tracker
# =====================================================================

class MilesSessionTracker:
    """Mimics the miles session server's TITO tracking.

    Uses LinearTrajectory + Qwen3TITOTokenizer matching the flow
    in miles/rollout/session/sessions.py chat_completions().
    """

    def __init__(self):
        from miles.rollout.session.linear_trajectory import LinearTrajectory
        from miles.utils.chat_template_utils import apply_chat_template
        from miles.utils.chat_template_utils.tito_tokenizer import get_tito_tokenizer
        from miles.utils.processing_utils import load_tokenizer

        print("[session] Loading tokenizer ...")
        self.tokenizer = load_tokenizer("Qwen/Qwen3-8B", trust_remote_code=True)
        self.tito_tokenizer = get_tito_tokenizer(
            self.tokenizer, tokenizer_type="qwen3", allowed_append_roles=["tool"],
        )
        self.comparator = self.tito_tokenizer.create_comparator()
        self.apply_chat_template = apply_chat_template

        self.session = LinearTrajectory()
        self.results = {"passed": 0, "failed": 0, "checks": []}
        self.turn_count = 0
        self.last_tools = None

        print(f"[session] Qwen3TITOTokenizer ready "
              f"(im_end={self.tito_tokenizer._im_end_id})")

    def _check(self, name: str, ok: bool, detail: str = ""):
        status = "PASS" if ok else "FAIL"
        self.results["passed" if ok else "failed"] += 1
        self.results["checks"].append({
            "turn": self.turn_count, "check": name,
            "status": status, "detail": detail,
        })
        tag = f"Turn {self.turn_count}"
        if ok:
            print(f"    [{status}] {tag}: {name}" +
                  (f" ({detail})" if detail else ""))
        else:
            print(f"    [{status}] {tag}: {name} — {detail}")

    def prepare_request(self, messages: list[dict], tools) -> list[int] | None:
        """Phase 1: validate + pretokenize. Returns input_ids or None."""
        self.turn_count += 1
        self.last_tools = tools
        print(f"\n  [session] Turn {self.turn_count}: "
              f"{len(messages)} msgs, "
              f"session has {len(self.session.messages)} stored / "
              f"{self.session.num_assistant} checkpoints")

        if not self.session.token_ids:
            return None

        try:
            pretokenized = self.session.prepare_pretokenized(
                messages, tools=tools, tito_tokenizer=self.tito_tokenizer,
            )
        except Exception as e:
            self._check("prepare_pretokenized", False, str(e)[:200])
            return None

        if pretokenized is not None:
            self._check("prepare_pretokenized", True,
                         f"{len(pretokenized['input_ids'])} input_ids")
            return pretokenized["input_ids"]
        return None

    def process_response(
        self,
        request_messages: list[dict],
        assistant_message: dict,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
    ):
        """Phase 3: validate prefix invariant + update state with REAL IDs."""
        try:
            self.session.update_pretokenized_state(
                request_messages, assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=self.tito_tokenizer.max_trim_tokens,
            )
            self._check("update_pretokenized_state", True,
                         f"checkpoint {self.session.num_assistant}, "
                         f"{len(self.session.token_ids)} token_ids")
        except Exception as e:
            self._check("update_pretokenized_state", False, str(e)[:300])
            # Fallback: advance session so next turn can proceed
            self.session.messages = list(request_messages) + [assistant_message]
            self.session.trajectory_token_ids.append(
                prompt_token_ids + completion_token_ids
            )
            self.session.num_assistant += 1

    def final_mismatch_check(self):
        """compute_session_mismatch on final state (matches miles SessionRegistry)."""
        if not self.session.token_ids:
            return
        try:
            expected_ids = self.apply_chat_template(
                self.session.messages, tokenizer=self.tokenizer,
                tools=self.last_tools,
                add_generation_prompt=False, tokenize=True,
            )
            mismatches = self.comparator.compare_sequences(
                expected_ids, self.session.token_ids,
            )
            non_asst = [m for m in mismatches
                        if m.type.value != "assistant_text"]
            asst_only = [m for m in mismatches
                         if m.type.value == "assistant_text"]
            if non_asst:
                for m in non_asst[:3]:
                    print(f"    mismatch: type={m.type.value} "
                          f"seg={m.segment_index} "
                          f"detail={m.detail[:100]}")
                self._check("final_mismatch", False,
                            f"{len(non_asst)} non-assistant mismatches")
            else:
                for m in asst_only:
                    print(f"    assistant mismatch seg={m.segment_index}:")
                    print(f"      expected: {repr(m.expected_text[:150])}")
                    print(f"      actual:   {repr(m.actual_text[:150])}")
                self._check("final_mismatch", True,
                            f"{len(asst_only)} assistant-only (OK)")
        except Exception as e:
            self._check("final_mismatch", False, str(e)[:200])


# =====================================================================
# Mocked client: /generate + Qwen25 tool parser
# =====================================================================

def create_mocked_model(tracker: MilesSessionTracker):
    """Create TITOChatModel with mocked client that calls /generate
    and parses tools with the same Qwen25Detector sglang uses."""

    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall, Function,
    )
    from openai.types import CompletionUsage
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from seta_env.models.tito_chat_model import TITOChatModel

    http_client = httpx.AsyncClient(timeout=180.0)

    async def mock_chat_create(**kwargs):
        """Intercept chat completion:
        1. Miles prepare_pretokenized
        2. sglang /generate with input_ids → REAL token IDs
        3. Qwen25 tool parser on raw text
        4. Miles update_pretokenized_state with REAL IDs
        5. Return ChatCompletion to agent
        """
        messages = kwargs.get("messages", [])
        tools_raw = kwargs.get("tools")
        max_tokens = kwargs.get("max_tokens", 4096)

        # --- Miles Phase 1: prepare_pretokenized ---
        input_ids = tracker.prepare_request(messages, tools_raw)

        if input_ids is None:
            # First turn: tokenize from scratch
            input_ids = tracker.apply_chat_template(
                messages, tokenizer=tracker.tokenizer,
                tools=tools_raw, add_generation_prompt=True, tokenize=True,
            )

        prompt_token_ids = list(input_ids)

        # --- Phase 2: sglang /generate with REAL token IDs ---
        resp = await http_client.post(
            f"{SGLANG_URL}/generate",
            json={
                "input_ids": prompt_token_ids,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                },
                "return_logprob": True,
            },
        )
        resp.raise_for_status()
        gen_data = resp.json()

        meta_info = gen_data.get("meta_info", {})
        output_token_logprobs = meta_info.get("output_token_logprobs", [])
        cached_tokens = meta_info.get("cached_tokens", 0)
        raw_text = gen_data.get("text", "")

        # REAL completion token IDs from model output (matches miles line 201)
        completion_token_ids = [t[1] for t in output_token_logprobs]

        # --- Phase 2b: Parse tool calls using Qwen25Detector ---
        # Same parser sglang uses with --tool-call-parser qwen25
        tool_calls_openai = None
        content = raw_text
        finish_reason = "stop"

        if tools_raw:
            # Build sglang Tool objects for the parser
            from pydantic import TypeAdapter
            from sglang.srt.entrypoints.openai.protocol import Tool
            sglang_tools = TypeAdapter(list[Tool]).validate_python(tools_raw)

            parser = FunctionCallParser(sglang_tools, "qwen25")
            if parser.has_tool_call(raw_text):
                content, call_items = parser.parse_non_stream(raw_text)
                if call_items:
                    finish_reason = "tool_calls"
                    tool_calls_openai = []
                    for call_item in call_items:
                        tool_calls_openai.append(
                            ChatCompletionMessageToolCall(
                                id=f"call_{uuid.uuid4().hex[:24]}",
                                type="function",
                                function=Function(
                                    name=call_item.name,
                                    arguments=call_item.parameters,
                                ),
                            )
                        )

        # --- Build assistant message for miles session ---
        assistant_msg = {"role": "assistant", "content": content or ""}
        if tool_calls_openai:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls_openai
            ]

        # --- Miles Phase 3: update_pretokenized_state with REAL IDs ---
        tracker.process_response(
            messages, assistant_msg,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
        )

        print(f"    [sglang] prompt={len(prompt_token_ids)} "
              f"completion={len(completion_token_ids)} "
              f"cached={cached_tokens} "
              f"tool_calls={len(tool_calls_openai) if tool_calls_openai else 0}")

        # --- Construct ChatCompletion for the agent ---
        message = ChatCompletionMessage(
            role="assistant",
            content=content or "",
            tool_calls=tool_calls_openai,
        )
        choice = Choice(
            index=0, message=message,
            finish_reason=finish_reason,
        )
        usage = CompletionUsage(
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(completion_token_ids),
            total_tokens=len(prompt_token_ids) + len(completion_token_ids),
        )
        completion = ChatCompletion(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            choices=[choice],
            created=int(time.time()),
            model=MODEL_TYPE,
            object="chat.completion",
            usage=usage,
        )
        return completion

    # Build model, replace async client
    model = TITOChatModel(
        model_type=MODEL_TYPE, model_config_dict=MODEL_CONFIG,
        api_key="EMPTY", url=MODEL_URL, tito_validate=True,
    )
    model._log_enabled = True
    model._log_dir = str(CAMEL_LOG_DIR)

    model._async_client = MagicMock()
    model._async_client.chat.completions.create = AsyncMock(
        side_effect=mock_chat_create
    )
    model._http_client = http_client
    return model


# =====================================================================
# Agent factory
# =====================================================================

def create_agent(model):
    from camel.messages import BaseMessage
    from seta_env.agent.prompt_loader import load_system_message
    from seta_env.agent.tito_train_agent import AgentTrainTITO
    from seta_env.toolkits.terminal_toolkit_docker import TerminalToolkit

    toolkit = TerminalToolkit(
        timeout=30.0, docker_container_name=CONTAINER_NAME,
        working_directory="/opt/pipeline",
        session_logs_dir=str(TERMINAL_LOG_DIR),
    )
    tool_names = ["shell_exec", "shell_view", "shell_wait",
                  "shell_write_to_process", "shell_kill_process",
                  "shell_write_content_to_file"]
    tools = [t for t in toolkit.get_tools()
             if t.get_function_name() in tool_names]

    system_message = load_system_message("sys_prompt_base")
    agent = AgentTrainTITO(
        system_message=BaseMessage.make_assistant_message(
            role_name="Developer Agent", content=system_message,
        ),
        model=model, tools=tools,
        token_limit=28672 - 4096,
        max_iteration=MAX_ITERATIONS,
        task_name="stack_overflow__888",
        summarize_threshold=None,
    )
    agent.reset()
    return agent, toolkit


# =====================================================================
# Main
# =====================================================================

async def run_test():
    from camel.messages import BaseMessage

    instruction = (TASK_DIR / "instruction.md").read_text()

    # Create miles session tracker
    tracker = MilesSessionTracker()

    # Create model with mocked client → /generate + qwen25 parser
    model = create_mocked_model(tracker)

    # Create agent
    agent, toolkit = create_agent(model)
    print(f"\n[agent] {len(agent.tool_dict)} tools, "
          f"token_limit={agent._token_limit}, "
          f"max_iter={MAX_ITERATIONS}")

    # Run
    print("\n" + "=" * 60)
    print("RUNNING AGENT (each turn: /generate + qwen25 parser + miles validation)")
    print("=" * 60)

    input_msg = BaseMessage.make_user_message(
        role_name="User", content=instruction,
    )
    response = await agent.astep(input_msg)

    # Cleanup
    await model._http_client.aclose()

    # Agent summary
    meta = agent.meta_info_record
    print(f"\n[agent] Done: {meta['iteration_count']} iterations, "
          f"{meta['total_tool_calls']} tool calls, "
          f"reason={meta['termination_reason']}")

    # TITO session summary
    session_msgs = model._session_messages
    n_asst = sum(1 for m in session_msgs if m.get("role") == "assistant")
    think_preserved = sum(
        1 for m in session_msgs
        if m.get("role") == "assistant"
        and "<think>" in (m.get("content") or "")
    )
    print(f"[tito]  Session: {len(session_msgs)} msgs ({n_asst} assistant)")
    print(f"[tito]  <think> preserved in {think_preserved}/{n_asst} "
          f"assistant messages")

    # Final miles mismatch check
    print("\n" + "=" * 60)
    print("FINAL MILES SESSION MISMATCH CHECK")
    print("=" * 60)
    tracker.final_mismatch_check()

    # Results
    r = tracker.results
    print("\n" + "=" * 60)
    p, f = r["passed"], r["failed"]
    total = p + f
    print(f"RESULTS: {p}/{total} passed, {f}/{total} failed")
    if f == 0:
        print("ALL CHECKS PASSED — true TITO with real token IDs")
    else:
        print("FAILED CHECKS:")
        for c in r["checks"]:
            if c["status"] == "FAIL":
                print(f"  Turn {c['turn']}: {c['check']} — {c['detail']}")
    print("=" * 60)

    # Save
    results_path = OUTPUT_DIR / "validation_results.json"
    with open(results_path, "w") as fh:
        json.dump(r, fh, indent=2, default=str)
    print(f"\nResults: {results_path}")

    return f


def main():
    for d in [OUTPUT_DIR, CAMEL_LOG_DIR, TERMINAL_LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    docker_setup()
    try:
        failures = asyncio.run(run_test())
    finally:
        print(f"\n[docker] Container left running: "
              f"docker exec -it {CONTAINER_NAME} bash")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
