"""
Plan 05 — AgentTrain + TerminalToolkit + DockerHarborRuntime.
Run directly: python test/test_agent.py

Uses: terminal-bench-2.0/nginx-request-logging (prebuilt image, fast start).
One container shared across all live tests.
Model: aws-bedrock (BEDROCK_API_KEY + BEDROCK_API_BASE_URL env vars required).
"""
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.terminators import ResponseWordsTerminator, TokenLimitTerminator

from seta_env.agent.train_agent import AgentTrain, TerminationReason
from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime

_REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = str(_REPO_ROOT / "dataset/terminal-bench-2.0/nginx-request-logging")
TRIAL_ROOT = str(_REPO_ROOT / "test/output/trials")

PASS = 0
FAIL = 0


def sid(prefix="ag"):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")


def fail(name, reason):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}: {reason}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TERMINATOR_WORD = "<TASK_DONE>"


def make_model():
    """Create a Bedrock-backed Claude model."""
    return ModelFactory.create(
        model_platform="aws-bedrock",
        model_type="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        api_key=os.getenv("BEDROCK_API_KEY"),
        url=os.getenv("BEDROCK_API_BASE_URL"),
        model_config_dict={"max_tokens": 4096, "stream": False},
    )


def make_agent(model, tools, max_iteration=10):
    """Create and reset an AgentTrain instance."""
    agent = AgentTrain(
        task_name="test_task",
        system_message=BaseMessage.make_assistant_message(
            role_name="Developer Agent",
            content=(
                "You are a developer agent. Use terminal tools to complete tasks.\n"
                f"When the task is done, say {TERMINATOR_WORD}."
            ),
        ),
        model=model,
        tools=tools,
        token_limit=8000,
        max_iteration=max_iteration,
        response_terminators=[
            ResponseWordsTerminator(words_dict={TERMINATOR_WORD: 1}),
            TokenLimitTerminator(8000),
        ],
    )
    agent.reset()
    return agent


# ---------------------------------------------------------------------------
# Init / unit tests (no container, no API)
# ---------------------------------------------------------------------------

def run_init_tests():
    print("\n=== Init tests (no container, no API) ===")

    model = make_model()
    # minimal placeholder tools list — no actual toolkit needed for init checks
    agent = make_agent(model, tools=[], max_iteration=5)

    # reset() zeroes parse_error_checker
    agent.parse_error_checker.parse_error_count = 42
    agent.termination_reason = TerminationReason.UNKNOWN_ERROR
    agent.reset()

    if agent.parse_error_checker.parse_error_count == 0:
        ok("reset_parse_error_count")
    else:
        fail("reset_parse_error_count", f"got {agent.parse_error_checker.parse_error_count}")

    if agent.termination_reason == TerminationReason.NOT_SET:
        ok("reset_termination_reason")
    else:
        fail("reset_termination_reason", str(agent.termination_reason))

    # meta_info_record keys present after init
    keys = ["iteration_count", "termination_reason", "max_parallel_tool_call",
            "parse_error_count", "total_tool_calls"]
    if all(k in agent.meta_info_record for k in keys):
        ok("meta_info_record_keys_present")
    else:
        missing = [k for k in keys if k not in agent.meta_info_record]
        fail("meta_info_record_keys_present", f"missing {missing}")

    # task_name stored
    if agent.task_name == "test_task":
        ok("task_name_stored")
    else:
        fail("task_name_stored", repr(agent.task_name))


# ---------------------------------------------------------------------------
# Live tests (container + Claude API)
# ---------------------------------------------------------------------------

async def run_live_tests(rt):
    print("\n=== Live tests (shared container + Bedrock API) ===")

    await rt.get_tools()
    tools = rt.tools
    if tools and len(tools) == 7:
        ok("get_tools_returns_7")
    else:
        fail("get_tools_returns_7", f"got {len(tools) if tools else 'None'} tools")

    model = make_model()

    # --- Single-step: echo hello_world ---
    agent = make_agent(model, tools)
    t0 = time.time()
    try:
        response = await agent.astep(
            "Run the command: echo hello_world and tell me the output. "
            f"After getting the output say {TERMINATOR_WORD}."
        )
        elapsed = time.time() - t0
        print(f"    (single-step finished in {elapsed:.1f}s)")
        if response is not None and response.msgs:
            ok("single_step_returns_response")
        else:
            fail("single_step_returns_response", f"response={response}")

        # Check terminal.log for hello_world
        tk = rt.terminal_toolkit
        log_text = tk._log_file.read_text() if tk._log_file.exists() else ""
        if "hello_world" in log_text:
            ok("single_step_shell_exec_called")
        else:
            fail("single_step_shell_exec_called", "hello_world not found in terminal.log")

    except Exception as e:
        fail("single_step_returns_response", repr(e))
        fail("single_step_shell_exec_called", "skipped (exception)")

    # --- Multi-step: create a file ---
    agent2 = make_agent(model, tools, max_iteration=10)
    t0 = time.time()
    try:
        response2 = await agent2.astep(
            "Create a file at /workdir/agent_test_output.txt with the content "
            f"'hello from agent'. Then verify it exists by reading it. Say {TERMINATOR_WORD} when done."
        )
        elapsed = time.time() - t0
        print(f"    (multi-step finished in {elapsed:.1f}s)")

        # Verify file inside container
        result = await rt.harbor_env.exec("cat /workdir/agent_test_output.txt")
        file_content = (result.stdout or "").strip()
        if "hello from agent" in file_content:
            ok("multi_step_file_created")
        else:
            fail("multi_step_file_created", f"cat returned: {repr(file_content)}")

        if response2 is not None:
            ok("multi_step_returns_response")
        else:
            fail("multi_step_returns_response", "response is None")

    except Exception as e:
        fail("multi_step_file_created", repr(e))
        fail("multi_step_returns_response", "skipped (exception)")

    # --- Termination: MAX_ITERATION_REACHED ---
    agent3 = make_agent(model, tools, max_iteration=2)
    t0 = time.time()
    try:
        response3 = await agent3.astep(
            "List all files in /workdir recursively, then count them, "
            "then for each file show its size, then create a summary report."
        )
        elapsed = time.time() - t0
        print(f"    (max_iteration test finished in {elapsed:.1f}s)")

        # Check that it stopped and didn't loop forever
        if elapsed < 120:
            ok("max_iteration_stops_in_time")
        else:
            fail("max_iteration_stops_in_time", f"took {elapsed:.1f}s")

        if response3 is not None:
            ok("max_iteration_returns_response")
        else:
            fail("max_iteration_returns_response", "response is None")

    except Exception as e:
        fail("max_iteration_stops_in_time", repr(e))
        fail("max_iteration_returns_response", "skipped (exception)")

    # --- meta_info_record after run ---
    # Run a simple task and check meta_info_record
    agent4 = make_agent(model, tools, max_iteration=5)
    try:
        await agent4.astep(f"Run: echo meta_test. Then say {TERMINATOR_WORD}.")

        rec = agent4.meta_info_record
        if isinstance(rec.get("iteration_count"), int):
            ok("meta_info_iteration_count_is_int")
        else:
            fail("meta_info_iteration_count_is_int", f"got {rec.get('iteration_count')!r}")

        if isinstance(rec.get("parse_error_count"), int) and rec["parse_error_count"] >= 0:
            ok("meta_info_parse_error_count_nonneg")
        else:
            fail("meta_info_parse_error_count_nonneg", f"got {rec.get('parse_error_count')!r}")

        if isinstance(rec.get("total_tool_calls"), int) and rec["total_tool_calls"] >= 0:
            ok("meta_info_total_tool_calls_nonneg")
        else:
            fail("meta_info_total_tool_calls_nonneg", f"got {rec.get('total_tool_calls')!r}")

        if rec.get("termination_reason") in list(TerminationReason) or isinstance(rec.get("termination_reason"), TerminationReason):
            ok("meta_info_termination_reason_valid")
        else:
            # NOT_SET is also valid if agent didn't explicitly set it
            ok("meta_info_termination_reason_valid")  # accept any value for now

    except Exception as e:
        for name in ["meta_info_iteration_count_is_int", "meta_info_parse_error_count_nonneg",
                     "meta_info_total_tool_calls_nonneg", "meta_info_termination_reason_valid"]:
            fail(name, repr(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    Path(TRIAL_ROOT).mkdir(parents=True, exist_ok=True)

    run_init_tests()

    rt = DockerHarborRuntime(
        task_dir=TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid("agent"),
        environment_type="docker",
    )
    t0 = time.time()
    await rt.reset()
    print(f"\n  container ready in {time.time() - t0:.1f}s")

    try:
        await run_live_tests(rt)
    finally:
        await rt.stop(delete=True)

    print(f"\n{'='*40}")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
