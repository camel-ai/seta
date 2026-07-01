"""
Plan 07 — Reward function integration.
Run directly: python test/test_reward_integration.py

Tests reward_factory with real Verifier-shaped dicts and calculate_reward()
via a minimal namespace object — no containers needed for most tests.
One live integration test runs Verifier → reward_factory end-to-end.
"""
import asyncio
import sys
import types
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from seta_env.verifiers.reward_fn import reward_factory
from seta_env.environments.terminal_env import TerminalEnvironment

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAL_ROOT = str(_REPO_ROOT / "test/output/trials")
DOCKER_IMAGE = "alexgshaw/nginx-request-logging:20251031"

PASS = 0
FAIL = 0


def sid(prefix="rw"):
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
# reward_factory — pure dict inputs (Verifier-shaped outputs)
# ---------------------------------------------------------------------------

async def run_reward_factory_tests():
    print("\n=== reward_factory tests (no container) ===")

    # Variant A shape: {"reward": 1.0}  →  pass_ratio = 1.0
    try:
        r = await reward_factory("pass_ratio", evaluation_results={"reward": 1.0})
        if r == 1.0:
            ok("pass_ratio_reward_1_0")
        else:
            fail("pass_ratio_reward_1_0", repr(r))
    except Exception as e:
        fail("pass_ratio_reward_1_0", repr(e))

    # Variant B shape: {"reward": 0.0}  →  pass_ratio = 0.0
    try:
        r = await reward_factory("pass_ratio", evaluation_results={"reward": 0.0})
        if r == 0.0:
            ok("pass_ratio_reward_0_0")
        else:
            fail("pass_ratio_reward_0_0", repr(r))
    except Exception as e:
        fail("pass_ratio_reward_0_0", repr(e))

    # Variant C shape: {"test1": 1, "test2": 0}  →  pass_ratio = 0.5
    try:
        r = await reward_factory("pass_ratio", evaluation_results={"test1": 1, "test2": 0})
        if r == 0.5:
            ok("pass_ratio_mixed_0_5")
        else:
            fail("pass_ratio_mixed_0_5", repr(r))
    except Exception as e:
        fail("pass_ratio_mixed_0_5", repr(e))

    # Empty dict  →  0.0
    try:
        r = await reward_factory("pass_ratio", evaluation_results={})
        if r == 0.0:
            ok("pass_ratio_empty_dict")
        else:
            fail("pass_ratio_empty_dict", repr(r))
    except Exception as e:
        fail("pass_ratio_empty_dict", repr(e))

    # trajectory=None is silently ignored (pass_ratio uses **kwargs)
    try:
        r = await reward_factory("pass_ratio", evaluation_results={"t1": 1}, trajectory=None)
        if r == 1.0:
            ok("pass_ratio_trajectory_none_ignored")
        else:
            fail("pass_ratio_trajectory_none_ignored", repr(r))
    except Exception as e:
        fail("pass_ratio_trajectory_none_ignored", repr(e))

    # trajectory string is silently ignored
    try:
        r = await reward_factory(
            "pass_ratio",
            evaluation_results={"t1": 1, "t2": 0},
            trajectory="[User]: do something\n[Assistant]: done",
        )
        if r == 0.5:
            ok("pass_ratio_trajectory_str_ignored")
        else:
            fail("pass_ratio_trajectory_str_ignored", repr(r))
    except Exception as e:
        fail("pass_ratio_trajectory_str_ignored", repr(e))

    # Unknown reward_fn raises ValueError
    try:
        await reward_factory("nonexistent_fn", evaluation_results={"t1": 1})
        fail("unknown_reward_fn_raises", "expected ValueError")
    except ValueError:
        ok("unknown_reward_fn_raises")
    except Exception as e:
        fail("unknown_reward_fn_raises", repr(e))


# ---------------------------------------------------------------------------
# calculate_reward() — bound to a minimal namespace, no container needed
# ---------------------------------------------------------------------------

async def run_calculate_reward_tests():
    print("\n=== calculate_reward() tests (no container) ===")

    # Helper: bind TerminalEnvironment.calculate_reward to a fake self
    def make_env(**attrs):
        ns = types.SimpleNamespace(**attrs)
        ns.calculate_reward = TerminalEnvironment.calculate_reward.__get__(ns, type(ns))
        return ns

    # No CAMEL_LOG_DIR → FileNotFoundError caught internally, trajectory=None → reward returned
    env = make_env(
        output_path=Path("/nonexistent_camel_log_path"),
        evaluation_results={"t1": 1},
        reward_fn="pass_ratio",
    )
    try:
        r = await env.calculate_reward()
        if r == 1.0:
            ok("calculate_reward_no_trajectory_file")
        else:
            fail("calculate_reward_no_trajectory_file", repr(r))
    except Exception as e:
        fail("calculate_reward_no_trajectory_file", repr(e))

    # Unknown reward_fn → Exception caught internally → returns None
    env2 = make_env(
        output_path=Path("/nonexistent_camel_log_path"),
        evaluation_results={"t1": 1},
        reward_fn="nonexistent_fn",
    )
    try:
        r = await env2.calculate_reward()
        if r is None:
            ok("calculate_reward_unknown_fn_returns_none")
        else:
            fail("calculate_reward_unknown_fn_returns_none", repr(r))
    except Exception as e:
        fail("calculate_reward_unknown_fn_returns_none", repr(e))

    # Mixed results
    env3 = make_env(
        output_path=Path("/nonexistent_camel_log_path"),
        evaluation_results={"t1": 1, "t2": 0, "t3": 1},
        reward_fn="pass_ratio",
    )
    try:
        r = await env3.calculate_reward()
        if abs(r - 2/3) < 1e-9:
            ok("calculate_reward_mixed_results")
        else:
            fail("calculate_reward_mixed_results", repr(r))
    except Exception as e:
        fail("calculate_reward_mixed_results", repr(e))


# ---------------------------------------------------------------------------
# Integration: real Verifier output → reward_factory (one container)
# ---------------------------------------------------------------------------

async def run_integration_test():
    print("\n=== Integration test (Verifier → reward_factory, one container) ===")

    from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
    from seta_env.verifiers.verifier import Verifier

    tmp = _REPO_ROOT / "test/output/verifier_tasks"
    tmp.mkdir(parents=True, exist_ok=True)

    # Build a variant-C task dir (JSON reward) to get a real multi-key dict
    task_dir = tmp / "integ_reward"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "task.toml").write_text(f"""\
version = "1.0"
[metadata]
difficulty = "easy"
[verifier]
timeout_sec = 60.0
[environment]
docker_image = "{DOCKER_IMAGE}"
""")
    (task_dir / "instruction.md").write_text("Integration test task.\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(f"FROM {DOCKER_IMAGE}\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh = tests_dir / "test.sh"
    test_sh.write_text('#!/bin/bash\nmkdir -p /logs/verifier\necho \'{"t1": 1, "t2": 0, "t3": 1}\' > /logs/verifier/reward.json\n')
    test_sh.chmod(0o755)

    rt = DockerHarborRuntime(
        task_dir=str(task_dir),
        trial_root=TRIAL_ROOT,
        session_id=sid("integ"),
        environment_type="docker",
    )
    await rt.reset()
    verifier = Verifier(
        task=rt._task,
        trial_paths=rt._trial_paths,
        environment=rt.harbor_env,
    )
    try:
        evaluation_results = await verifier.verify()
        reward = await reward_factory("pass_ratio", evaluation_results=evaluation_results)

        if evaluation_results == {"t1": 1, "t2": 0, "t3": 1}:
            ok("integration_verifier_output_shape")
        else:
            fail("integration_verifier_output_shape", repr(evaluation_results))

        if abs(reward - 2/3) < 1e-9:
            ok("integration_reward_factory_result")
        else:
            fail("integration_reward_factory_result", repr(reward))

    except Exception as e:
        fail("integration_verifier_output_shape", repr(e))
        fail("integration_reward_factory_result", repr(e))
    finally:
        await rt.stop(delete=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    Path(TRIAL_ROOT).mkdir(parents=True, exist_ok=True)

    await run_reward_factory_tests()
    await run_calculate_reward_tests()
    await run_integration_test()

    print(f"\n{'='*40}")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
