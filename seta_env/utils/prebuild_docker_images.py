"""Pre-build Docker images for one task or a tree of tasks, in parallel.

Given a single Harbor task directory or a root directory containing many tasks,
discover every task (identified by the presence of ``task.toml``) and call
``DockerHarborRuntime.build()`` on each one concurrently. The build itself is
delegated to the underlying Harbor environment (local docker, remote_docker,
daytona, modal, ...), so this module just orchestrates parallelism.

Use as a CLI:
    python -m seta_env.utils.prebuild_docker_images <path> [options]

Use as a module:
    from seta_env.utils.prebuild_docker_images import prebuild, discover_tasks

    await prebuild(
        path=Path("dataset/seta-env-harbor"),
        trial_root=Path("/tmp/prebuild_trials"),
        environment_type="docker",
        concurrency=8,
    )

CLI examples:
    # Single task, local docker
    python -m seta_env.utils.prebuild_docker_images dataset/seta-env-harbor/0

    # Whole dataset root, 8-way parallel, against a remote node manager
    python -m seta_env.utils.prebuild_docker_images dataset/seta-env-harbor \\
        --environment-type remote_docker \\
        --node-manager-url $NODE_MANAGER_URL \\
        --node-api-key $NODE_MANAGER_API_KEY \\
        --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Tuple

from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime


def discover_tasks(path: Path) -> List[Path]:
    """Return every task directory at or beneath ``path``.

    A directory qualifies as a task if it contains ``task.toml``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if (path / "task.toml").is_file():
        return [path]

    tasks = sorted({p.parent for p in path.rglob("task.toml")})
    return tasks


def _make_runtime(
    task_dir: Path,
    trial_root: Path,
    environment_type: str,
    node_manager_url: str | None,
    node_api_key: str | None,
) -> DockerHarborRuntime:
    session_id = f"prebuild-{task_dir.name}-{uuid.uuid4().hex[:8]}"
    kwargs: dict = {}
    if environment_type == "remote_docker":
        if not node_manager_url or not node_api_key:
            raise ValueError(
                "remote_docker requires --node-manager-url and --node-api-key"
            )
        kwargs["node_manager_url"] = node_manager_url
        kwargs["node_api_key"] = node_api_key

    return DockerHarborRuntime(
        task_dir=str(task_dir),
        trial_root=str(trial_root),
        session_id=session_id,
        environment_type=environment_type,
        **kwargs,
    )


async def _build_one(
    task_dir: Path,
    trial_root: Path,
    environment_type: str,
    node_manager_url: str | None,
    node_api_key: str | None,
    semaphore: asyncio.Semaphore,
    build_timeout: float | None,
) -> Tuple[Path, bool, str, float]:
    async with semaphore:
        t0 = time.monotonic()
        try:
            rt = _make_runtime(
                task_dir,
                trial_root,
                environment_type,
                node_manager_url,
                node_api_key,
            )
        except Exception as e:
            return task_dir, False, f"init error: {e!r}", time.monotonic() - t0

        try:
            if build_timeout is not None:
                await asyncio.wait_for(rt.build(), timeout=build_timeout)
            else:
                await rt.build()
            return task_dir, True, "ok", time.monotonic() - t0
        except asyncio.TimeoutError:
            return task_dir, False, f"timeout after {build_timeout}s", time.monotonic() - t0
        except Exception as e:
            return task_dir, False, f"{type(e).__name__}: {e}", time.monotonic() - t0
        finally:
            # Best-effort cleanup of the per-build logger handler. We do not
            # call stop() — the container was never started.
            try:
                rt._close_logger()
            except Exception:
                pass


async def prebuild(
    path: Path,
    trial_root: Path,
    environment_type: str,
    concurrency: int,
    node_manager_url: str | None,
    node_api_key: str | None,
    build_timeout: float | None,
) -> int:
    tasks = discover_tasks(path)
    if not tasks:
        print(f"No tasks found under {path} (expected task.toml).", file=sys.stderr)
        return 1

    print(f"Discovered {len(tasks)} task(s); building with concurrency={concurrency}")
    for t in tasks:
        print(f"  - {t}")

    trial_root.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    started = time.monotonic()
    coros = [
        _build_one(
            t,
            trial_root,
            environment_type,
            node_manager_url,
            node_api_key,
            sem,
            build_timeout,
        )
        for t in tasks
    ]
    results = await asyncio.gather(*coros)
    elapsed = time.monotonic() - started

    ok = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print("\n==================== Pre-build summary ====================")
    print(f"  total:   {len(results)}")
    print(f"  success: {len(ok)}")
    print(f"  failed:  {len(failed)}")
    print(f"  wall:    {elapsed:.1f}s")
    if ok:
        print("\nSucceeded:")
        for task_dir, _, _, dur in ok:
            print(f"  [OK]   ({dur:6.1f}s) {task_dir}")
    if failed:
        print("\nFailed:")
        for task_dir, _, msg, dur in failed:
            print(f"  [FAIL] ({dur:6.1f}s) {task_dir}  -- {msg}")

    return 0 if not failed else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-build Docker images for Harbor tasks in parallel."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to a single task directory or a root containing many tasks.",
    )
    p.add_argument(
        "--environment-type",
        default="docker",
        choices=["docker", "remote_docker", "daytona", "modal"],
        help="Harbor environment backend to build with (default: docker).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of builds running in parallel (default: 4).",
    )
    p.add_argument(
        "--trial-root",
        type=Path,
        default=Path("seta_env/utils/output/prebuild_trials"),
        help="Where to write per-build trial directories / logs.",
    )
    p.add_argument(
        "--node-manager-url",
        default=os.environ.get("NODE_MANAGER_URL"),
        help="Node manager URL (remote_docker only). Defaults to $NODE_MANAGER_URL.",
    )
    p.add_argument(
        "--node-api-key",
        default=os.environ.get("NODE_MANAGER_API_KEY"),
        help="Node manager API key (remote_docker only). Defaults to $NODE_MANAGER_API_KEY.",
    )
    p.add_argument(
        "--build-timeout",
        type=float,
        default=None,
        help="Per-task build timeout in seconds (default: no timeout).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(
        prebuild(
            path=args.path,
            trial_root=args.trial_root,
            environment_type=args.environment_type,
            concurrency=args.concurrency,
            node_manager_url=args.node_manager_url,
            node_api_key=args.node_api_key,
            build_timeout=args.build_timeout,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
