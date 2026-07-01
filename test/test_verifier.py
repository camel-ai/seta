"""
Plan 06 — Verifier connected to live Docker runtime.
Run directly: python test/test_verifier.py

Creates minimal task dirs on disk for each variant so each runtime loads
the right test script.  One container per variant, all sequentially.
Unit tests for _parse_reward_text/_parse_reward_json need no container.
"""
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
from seta_env.verifiers.verifier import (
    AddTestsDirError,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    Verifier,
    VerifierOutputParseError,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAL_ROOT = str(_REPO_ROOT / "test/output/trials")
# Reuse prebuilt image from nginx task — fast start, no build needed
DOCKER_IMAGE = "alexgshaw/nginx-request-logging:20251031"

PASS = 0
FAIL = 0


def sid(prefix="vf"):
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

def make_task_dir(base: Path, name: str, script: str) -> Path:
    """Create a minimal Harbor-format task dir with a custom test.sh."""
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
    (task_dir / "instruction.md").write_text("Test task.\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(f"FROM {DOCKER_IMAGE}\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(script)
    test_sh.chmod(0o755)
    return task_dir


async def run_variant(task_dir: Path, session_id: str):
    """Start a runtime for the given task_dir, run verify(), stop and return result."""
    rt = DockerHarborRuntime(
        task_dir=str(task_dir),
        trial_root=TRIAL_ROOT,
        session_id=session_id,
        environment_type="docker",
    )
    await rt.reset()
    verifier = Verifier(
        task=rt._task,
        trial_paths=rt._trial_paths,
        environment=rt.harbor_env,
    )
    try:
        result = await verifier.verify()
        return result, rt._trial_paths
    finally:
        await rt.stop(delete=True)


# ---------------------------------------------------------------------------
# Unit tests — no container, just files on disk
# ---------------------------------------------------------------------------

def run_unit_tests(tmp: Path):
    print("\n=== Unit tests (_parse_reward_text / _parse_reward_json) ===")

    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    # Build a minimal task dir so we can construct a Verifier
    task_dir = make_task_dir(tmp, "unit_task", "#!/bin/bash\n")
    task = Task(task_dir)
    trial_dir = tmp / "unit_trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    verifier = Verifier(task=task, trial_paths=trial_paths, environment=None)

    # --- _parse_reward_text ---

    trial_paths.reward_text_path.write_text("0.75\n")
    try:
        result = verifier._parse_reward_text()
        if result == {"reward": 0.75}:
            ok("parse_reward_text_valid")
        else:
            fail("parse_reward_text_valid", repr(result))
    except Exception as e:
        fail("parse_reward_text_valid", repr(e))

    trial_paths.reward_text_path.write_text("not_a_number")
    try:
        verifier._parse_reward_text()
        fail("parse_reward_text_invalid", "expected VerifierOutputParseError")
    except VerifierOutputParseError:
        ok("parse_reward_text_invalid")
    except Exception as e:
        fail("parse_reward_text_invalid", repr(e))

    trial_paths.reward_text_path.write_text("")
    try:
        verifier._parse_reward_text()
        fail("parse_reward_text_empty", "expected RewardFileEmptyError")
    except RewardFileEmptyError:
        ok("parse_reward_text_empty")
    except Exception as e:
        fail("parse_reward_text_empty", repr(e))

    # --- _parse_reward_json ---

    trial_paths.reward_json_path.write_text('{"t1": 1, "t2": 0}')
    try:
        result = verifier._parse_reward_json()
        if result == {"t1": 1, "t2": 0}:
            ok("parse_reward_json_valid")
        else:
            fail("parse_reward_json_valid", repr(result))
    except Exception as e:
        fail("parse_reward_json_valid", repr(e))

    trial_paths.reward_json_path.write_text("not json")
    try:
        verifier._parse_reward_json()
        fail("parse_reward_json_invalid", "expected VerifierOutputParseError")
    except VerifierOutputParseError:
        ok("parse_reward_json_invalid")
    except Exception as e:
        fail("parse_reward_json_invalid", repr(e))

    trial_paths.reward_json_path.write_text("")
    try:
        verifier._parse_reward_json()
        fail("parse_reward_json_empty", "expected RewardFileEmptyError")
    except RewardFileEmptyError:
        ok("parse_reward_json_empty")
    except Exception as e:
        fail("parse_reward_json_empty", repr(e))


# ---------------------------------------------------------------------------
# Live tests — real container per variant
# ---------------------------------------------------------------------------

async def run_live_tests(tmp: Path):
    print("\n=== Live tests (real containers) ===")

    # Variant A — reward.txt = 1.0
    task_a = make_task_dir(tmp, "variant_a", """\
#!/bin/bash
mkdir -p /logs/verifier
echo "1.0" > /logs/verifier/reward.txt
""")
    try:
        result, trial_paths = await run_variant(task_a, sid("va"))
        if isinstance(result, dict):
            ok("variant_a_returns_dict")
        else:
            fail("variant_a_returns_dict", repr(result))
        if result.get("reward") == 1.0:
            ok("variant_a_reward_1_0")
        else:
            fail("variant_a_reward_1_0", repr(result))
        if trial_paths.test_stdout_path.exists():
            ok("variant_a_stdout_exists")
        else:
            fail("variant_a_stdout_exists", "test_stdout_path missing")
    except Exception as e:
        for name in ["variant_a_returns_dict", "variant_a_reward_1_0", "variant_a_stdout_exists"]:
            fail(name, repr(e))

    # Variant B — reward.txt = 0.0
    task_b = make_task_dir(tmp, "variant_b", """\
#!/bin/bash
mkdir -p /logs/verifier
echo "0.0" > /logs/verifier/reward.txt
""")
    try:
        result, _ = await run_variant(task_b, sid("vb"))
        if result.get("reward") == 0.0:
            ok("variant_b_reward_0_0")
        else:
            fail("variant_b_reward_0_0", repr(result))
    except Exception as e:
        fail("variant_b_reward_0_0", repr(e))

    # Variant C — reward.json
    task_c = make_task_dir(tmp, "variant_c", """\
#!/bin/bash
mkdir -p /logs/verifier
echo '{"test1": 1, "test2": 0}' > /logs/verifier/reward.json
""")
    try:
        result, _ = await run_variant(task_c, sid("vc"))
        if result == {"test1": 1, "test2": 0}:
            ok("variant_c_reward_json")
        else:
            fail("variant_c_reward_json", repr(result))
    except Exception as e:
        fail("variant_c_reward_json", repr(e))

    # Variant D — no reward file written
    task_d = make_task_dir(tmp, "variant_d", """\
#!/bin/bash
exit 0
""")
    try:
        await run_variant(task_d, sid("vd"))
        fail("variant_d_no_reward_raises", "expected RewardFileNotFoundError")
    except RewardFileNotFoundError:
        ok("variant_d_no_reward_raises")
    except Exception as e:
        fail("variant_d_no_reward_raises", repr(e))

    # Real task — nginx-request-logging with its actual test.sh.
    # Container starts empty (no nginx configured) so tests will fail and
    # reward.txt = 0.  We verify the full pipeline runs correctly end-to-end:
    # upload tests, execute real test script, download results, parse reward.
    print("\n  [real task test — installs packages, may take ~60s]")
    NGINX_TASK_DIR = str(_REPO_ROOT / "dataset/terminal-bench-2.0/nginx-request-logging")
    rt = DockerHarborRuntime(
        task_dir=NGINX_TASK_DIR,
        trial_root=TRIAL_ROOT,
        session_id=sid("real"),
        environment_type="docker",
    )
    await rt.reset()
    verifier = Verifier(
        task=rt._task,
        trial_paths=rt._trial_paths,
        environment=rt.harbor_env,
    )
    try:
        result = await verifier.verify()
        if isinstance(result, dict):
            ok("real_task_returns_dict")
        else:
            fail("real_task_returns_dict", repr(result))
        if "reward" in result and result["reward"] in (0, 0.0, 1, 1.0):
            ok("real_task_reward_is_numeric")
        else:
            fail("real_task_reward_is_numeric", repr(result))
        if rt._trial_paths.test_stdout_path.exists() and rt._trial_paths.test_stdout_path.stat().st_size > 0:
            ok("real_task_stdout_nonempty")
        else:
            fail("real_task_stdout_nonempty", "test stdout missing or empty")
    except Exception as e:
        for name in ["real_task_returns_dict", "real_task_reward_is_numeric", "real_task_stdout_nonempty"]:
            fail(name, repr(e))
    finally:
        await rt.stop(delete=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    tmp = _REPO_ROOT / "test/output/verifier_tasks"
    tmp.mkdir(parents=True, exist_ok=True)
    Path(TRIAL_ROOT).mkdir(parents=True, exist_ok=True)

    run_unit_tests(tmp)
    await run_live_tests(tmp)

    print(f"\n{'='*40}")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
