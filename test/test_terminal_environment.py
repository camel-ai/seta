"""
Plan 08 — TerminalEnvironment full end-to-end with concurrent tasks.
Run directly: python test/test_terminal_environment.py

Runs two tasks concurrently (asyncio.gather) and checks run_info, reward,
timings, and agent_summary. Also tests stage-1 failure path.
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camel.models import ModelFactory

from seta_env.environments.terminal_env import TerminalEnvironment

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAL_ROOT = str(_REPO_ROOT / "test/output/trials")
TASK_BASE  = str(_REPO_ROOT / "test/output/env_tasks")
DOCKER_IMAGE = "alexgshaw/nginx-request-logging:20251031"

PASS = 0
FAIL = 0


def sid(prefix="env"):
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

def make_task_dir(base: Path, name: str, instruction: str, test_script: str) -> Path:
    """Create a minimal Harbor-format task directory."""
    task_dir = base / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(f"""\
version = "1.0"

[metadata]
difficulty = "easy"

[verifier]
timeout_sec = 60.0

[environment]
docker_image = "{DOCKER_IMAGE}"
""")
    (task_dir / "instruction.md").write_text(instruction + "\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(f"FROM {DOCKER_IMAGE}\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(test_script)
    test_sh.chmod(0o755)
    return task_dir


def make_model():
    return ModelFactory.create(
        model_platform="aws-bedrock",
        model_type="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        api_key=os.getenv("BEDROCK_API_KEY"),
        url=os.getenv("BEDROCK_API_BASE_URL"),
        model_config_dict={"max_tokens": 4096, "stream": False},
    )


def make_env() -> TerminalEnvironment:
    """Each concurrent task needs its own TerminalEnvironment and model instance
    to avoid _log_dir mutation races during _reset_agent."""
    return TerminalEnvironment(
        agent_config={
            "system_message": (
                "You are a developer agent. Use shell tools to complete the task. "
                "When done, stop calling tools."
            ),
            "max_total_tokens": 8000,
            "max_iteration": 5,
            "working_directory": "/workdir",
            "tool_names": ["shell_exec", "shell_view", "shell_write_content_to_file"],
        },
        model_config={"model": make_model()},
        runtime_config={
            "trial_root": TRIAL_ROOT,
            "environment_type": "docker",
        },
        env_config={"reward_fn": "pass_ratio"},
    )


def check_run_info(run_info, reward, prefix):
    """Common assertions on a run_info + reward pair."""
    expected_keys = {
        "task_name", "uid", "traj_i", "error_info",
        "timings", "reward", "evaluation", "agent_summary",
    }
    missing = expected_keys - set(run_info.keys())
    if not missing:
        ok(f"{prefix}_run_info_keys")
    else:
        fail(f"{prefix}_run_info_keys", f"missing: {missing}")

    if "5_close" in run_info["timings"]:
        ok(f"{prefix}_close_ran")
    else:
        fail(f"{prefix}_close_ran", f"timings keys: {list(run_info['timings'])}")

    if not run_info["error_info"]:
        ok(f"{prefix}_no_error")
    else:
        fail(f"{prefix}_no_error", repr(run_info["error_info"]))

    if isinstance(reward, float):
        ok(f"{prefix}_reward_is_float")
    else:
        fail(f"{prefix}_reward_is_float", repr(reward))

    summary = run_info.get("agent_summary", {})
    for key in ["iteration_count", "termination_reason"]:
        if key in summary:
            ok(f"{prefix}_summary_{key}")
        else:
            fail(f"{prefix}_summary_{key}", repr(summary))


# ---------------------------------------------------------------------------
# Concurrent happy-path test
# ---------------------------------------------------------------------------

async def run_concurrent_tasks():
    print("\n=== Concurrent tasks test (2 containers) ===")

    task_base = Path(TASK_BASE)
    task_base.mkdir(parents=True, exist_ok=True)

    task1_dir = make_task_dir(
        task_base, "create_result_file",
        instruction='Create a file at /workdir/result.txt containing the text "done".',
        test_script="""\
#!/bin/bash
mkdir -p /logs/verifier
if [ -f /workdir/result.txt ] && grep -q "done" /workdir/result.txt; then
    echo '{"file_created": 1}' > /logs/verifier/reward.json
else
    echo '{"file_created": 0}' > /logs/verifier/reward.json
fi
""",
    )

    task2_dir = make_task_dir(
        task_base, "create_hello_file",
        instruction='Create a file at /workdir/hello.txt containing the text "hello".',
        test_script="""\
#!/bin/bash
mkdir -p /logs/verifier
if [ -f /workdir/hello.txt ] && grep -q "hello" /workdir/hello.txt; then
    echo '{"file_created": 1}' > /logs/verifier/reward.json
else
    echo '{"file_created": 0}' > /logs/verifier/reward.json
fi
""",
    )

    tasks = [
        (
            {"task_name": "create_result_file", "task_path": str(task1_dir),
             "instruction": 'Create a file at /workdir/result.txt containing the text "done".'},
            sid("t1"),
            make_env(),
        ),
        (
            {"task_name": "create_hello_file", "task_path": str(task2_dir),
             "instruction": 'Create a file at /workdir/hello.txt containing the text "hello".'},
            sid("t2"),
            make_env(),
        ),
    ]

    print(f"  UIDs: {[uid for _, uid, _ in tasks]}")

    results = await asyncio.gather(
        *[env.step(task, uid=uid) for task, uid, env in tasks],
        return_exceptions=True,
    )

    for i, (res, (task, uid, _)) in enumerate(zip(results, tasks), 1):
        prefix = f"task{i}_{task['task_name']}"
        if isinstance(res, Exception):
            fail(f"{prefix}_no_exception", repr(res))
            continue
        run_info, reward = res
        check_run_info(run_info, reward, prefix)


# ---------------------------------------------------------------------------
# Stage-1 failure test
# ---------------------------------------------------------------------------

async def run_stage1_failure_test():
    print("\n=== Stage 1 failure test (nonexistent task dir) ===")

    env = make_env()
    task = {
        "task_name": "bad_task",
        "task_path": "/nonexistent/task/dir/does_not_exist",
        "instruction": "This should fail at stage 1.",
    }

    run_info, reward = await env.step(task, uid=sid("fail"))

    if run_info["error_info"].get("stage") == "1_reset_env":
        ok("stage1_error_stage_recorded")
    else:
        fail("stage1_error_stage_recorded", repr(run_info["error_info"]))

    if run_info["error_info"].get("error_message"):
        ok("stage1_error_message_nonempty")
    else:
        fail("stage1_error_message_nonempty", repr(run_info["error_info"]))

    if reward is None:
        ok("stage1_reward_is_none")
    else:
        fail("stage1_reward_is_none", repr(reward))

    if isinstance(run_info, dict):
        ok("stage1_run_info_returned")
    else:
        fail("stage1_run_info_returned", repr(type(run_info)))

    # 5_close may or may not be present depending on where init failed — just
    # verify that run_info is consistent (no crash)
    timings = run_info.get("timings", {})
    if "1_reset_env" in timings:
        ok("stage1_timing_recorded")
    else:
        fail("stage1_timing_recorded", repr(list(timings)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    Path(TRIAL_ROOT).mkdir(parents=True, exist_ok=True)

    await run_concurrent_tasks()
    await run_stage1_failure_test()

    print(f"\n{'='*40}")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
