"""Async orchestrator for the evolution pipeline.

Three independent stages, each a method on EvolOrchestrator:
  run_evolve()  — multi-round evolution
  run_rollout() — scan task folders, fill rollout gaps
  run_verify()  — trajectory judge on all-fail rollouts
"""

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evol_config import (
    Config, EvolveRoundConfig, RolloutModelConfig,
    ROLLOUT_AGENT_CONFIG, ROLLOUT_RUNTIME_CONFIG, ROLLOUT_ENV_CONFIG,
    rollout_dir_for,
)
from pipeline_base import EvolutionOption, SynthResult
from io_utils import (
    is_harbor_task_complete, list_input_tasks, generate_variant_ids,
    read_synth_info, write_synth_info, make_synth_info,
    load_filter_csv, parse_variant_root, find_rollout_gaps,
    read_instruction, write_summary_csv,
)
from evol_task_pipeline import EvolTaskPipeline
from agents.claude_agents import (
    ClaudeEvolAgent, ClaudeDatapointAgent, ClaudeRateLimitError,
    ClaudeTrajectoryJudgeAgent,
)
from hf_utils import download_input_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evol queue building
# ---------------------------------------------------------------------------

def collect_evol_queue(
    rnd: EvolveRoundConfig,
    config: Config,
) -> List[Dict[str, Any]]:
    """Build the work queue for one evolution round."""
    input_dir = os.path.abspath(rnd.input_dir)
    output_dir = os.path.abspath(rnd.output_dir)
    strategy = rnd.strategy

    local_tasks = set(list_input_tasks(input_dir))
    logger.info("Found %d local input tasks in %s", len(local_tasks), input_dir)

    # Apply top-level filter_csv (seed-level: matches on root task_id).
    filter_ids = load_filter_csv(config.filter_csv) if config.filter_csv else None

    # Detect "seed round": input_dir has no synth_info.json in any task folder.
    # Chained rounds read previous-round output which always has synth_info.
    is_seed_round = not any(
        (Path(input_dir) / t / "synth_info.json").exists()
        for t in local_tasks
    )

    if filter_ids:
        # Match local variants whose root seed is in the filter.
        local_filtered = sorted(t for t in local_tasks if parse_variant_root(t) in filter_ids)
        matched_roots = {parse_variant_root(t) for t in local_filtered}
        # HF download only makes sense for the seed round. For chained rounds,
        # missing local variants = they failed earlier; don't try to download.
        remote_candidates = sorted(filter_ids - matched_roots)
        if is_seed_round and remote_candidates and config.huggingface.input_repo:
            tasks = local_filtered + remote_candidates
        else:
            tasks = local_filtered
    else:
        tasks = sorted(local_tasks)

    logger.info("After filter: %d tasks", len(tasks))

    queue: List[Dict[str, Any]] = []
    for task_id in tasks:
        variant_ids = generate_variant_ids(
            task_id, strategy.evol_strategy, strategy.max_variants,
        )
        all_done = True
        for vid in variant_ids:
            info = read_synth_info(output_dir, vid)
            if info is None:
                all_done = False
                break
            status = info.get("status", "")
            if status == "done":
                continue
            if status == "timeout" and config.evolve.skip_timeout:
                continue
            all_done = False
            break

        if all_done:
            continue

        queue.append({
            "task_id": task_id,
            "input_task_path": os.path.join(input_dir, task_id),
        })

    import random
    random.Random(42).shuffle(queue)
    logger.info("Evol queue: %d tasks to process", len(queue))
    return queue


def _update_synth_info(
    output_dir: str, vid: str, task_id: str, rnd: EvolveRoundConfig, **fields
) -> None:
    """Update synth_info.json, preserving existing fields (e.g. traj_judge)."""
    existing = read_synth_info(output_dir, vid) or {}
    base = make_synth_info(
        vid, task_id, rnd.name or rnd.strategy.evol_target,
        rnd.strategy.evol_target, rnd.input_dir, **fields,
    )
    existing.update(base)
    write_synth_info(output_dir, vid, existing)


# ---------------------------------------------------------------------------
# Evol worker
# ---------------------------------------------------------------------------

class _RateLimitGate:
    """Coordinates pausing all evol workers when rate limit is hit.

    Flow:
      1. Any worker hits rate limit → trigger() pauses all (resume_at = now + 600)
      2. At resume_at, exactly ONE worker becomes the probe (others keep waiting)
      3. Probe runs its task:
         - success → probe_success() releases all other workers
         - rate limit again → trigger() re-pauses; cycle repeats
    """

    def __init__(self, pause_duration: int = 600):
        self._pause_duration = pause_duration
        self._resume_at: float = 0.0        # monotonic time
        self._probe_in_progress: bool = False
        self._lock = asyncio.Lock()

    async def acquire(self, worker_id: int) -> bool:
        """Wait until this worker may proceed.

        Returns True if the worker is the probe (must call probe_success
        or trigger on completion). Returns False if the rate limit is
        inactive (normal flow).
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                if now < self._resume_at:
                    # still paused — wait then re-check
                    wait_s = self._resume_at - now
                elif self._probe_in_progress:
                    # another worker is probing — wait briefly
                    wait_s = 5.0
                elif self._resume_at > 0.0:
                    # pause expired and no probe yet — become the probe
                    self._probe_in_progress = True
                    logger.info("[W%d] Rate-limit probe: testing if limit cleared", worker_id)
                    return True
                else:
                    # no pause at all — normal flow
                    return False
            await asyncio.sleep(wait_s)

    async def trigger(self, worker_id: int) -> None:
        """Mark rate-limit pause; dedup concurrent triggers.

        If called while probing, ends the probe (with failure).
        """
        async with self._lock:
            now = time.monotonic()
            new_resume = now + self._pause_duration
            if new_resume > self._resume_at:
                self._resume_at = new_resume
                logger.error(
                    "[W%d] Rate limit hit — all workers paused for %ds",
                    worker_id, self._pause_duration,
                )
            self._probe_in_progress = False

    async def probe_success(self, worker_id: int) -> None:
        """Probe worker succeeded — release all other workers."""
        async with self._lock:
            if self._probe_in_progress:
                logger.info("[W%d] Rate-limit cleared — resuming all workers", worker_id)
            self._resume_at = 0.0
            self._probe_in_progress = False


async def _evol_worker(
    worker_id: int,
    queue: asyncio.Queue,
    results: list,
    rnd: EvolveRoundConfig,
    config: Config,
    gate: "_RateLimitGate",
) -> None:
    """Consume tasks from queue and run EvolTaskPipeline on each."""
    output_dir = os.path.abspath(rnd.output_dir)
    strategy = rnd.strategy

    evol_opt = EvolutionOption(
        strategy=strategy.evol_strategy,
        parameters={"target": strategy.evol_target},
        max_variants=strategy.max_variants,
    )

    pipeline = EvolTaskPipeline(
        evol_agent=ClaudeEvolAgent(),
        datapoint_agent=ClaudeDatapointAgent(),
        output_dir=output_dir,
        # Datapoint agent writes harbor validation under <rollout_dir>/_validation/.
        # Use the rollout-dir convention (<tasks_dir>_rollout) so validation
        # artifacts don't pollute the evolved-tasks output_dir.
        rollout_dir=rollout_dir_for(output_dir),
    )

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        task_id = item["task_id"]
        input_task_path = item["input_task_path"]

        # On-demand download (before gate so it doesn't retry on rate-limit)
        if not os.path.isdir(input_task_path):
            if config.huggingface.input_repo:
                logger.info("[W%d] Downloading %s from HF...", worker_id, task_id)
                ok = download_input_task(task_id, rnd.input_dir, config)
                if not ok:
                    logger.warning("[W%d] Download failed for %s", worker_id, task_id)
                    results.append((task_id, None))
                    queue.task_done()
                    continue
            else:
                logger.warning("[W%d] %s not found locally", worker_id, task_id)
                results.append((task_id, None))
                queue.task_done()
                continue

        # Mark in_progress (only for variants not already done)
        variant_ids = generate_variant_ids(task_id, strategy.evol_strategy, strategy.max_variants)
        for vid in variant_ids:
            existing = read_synth_info(output_dir, vid)
            if existing and existing.get("status") == "done":
                continue  # don't clobber completed work
            info = make_synth_info(
                vid, task_id, rnd.name or strategy.evol_target,
                strategy.evol_target, rnd.input_dir, status="in_progress",
            )
            write_synth_info(output_dir, vid, info)

        # Retry loop for rate-limit: retry the SAME item in place instead of
        # re-queueing past the sentinels (which would orphan it).
        synth_results = None
        task_done_cleanly = False
        while not task_done_cleanly:
            is_probe = await gate.acquire(worker_id)
            t0 = time.monotonic()
            try:
                synth_results = await asyncio.wait_for(
                    pipeline.run(task_id, input_task_path, evol_opt),
                    timeout=config.evolve.task_timeout_s,
                )
                task_done_cleanly = True
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                logger.warning("[W%d] %s timed out after %.0fs", worker_id, task_id, elapsed)
                for vid in variant_ids:
                    _update_synth_info(output_dir, vid, task_id, rnd,
                                       status="timeout", total_evol_time_s=elapsed)
                if is_probe:
                    await gate.probe_success(worker_id)  # not a rate-limit issue
                results.append((task_id, None))
                queue.task_done()
                break
            except ClaudeRateLimitError:
                # Gate-trigger makes acquire() block until probe succeeds on
                # some worker. Loop continues, acquire() will block, then retry.
                await gate.trigger(worker_id)
                continue
            except Exception as e:
                elapsed = time.monotonic() - t0
                logger.error("[W%d] %s failed: %s", worker_id, task_id, e, exc_info=True)
                for vid in variant_ids:
                    _update_synth_info(output_dir, vid, task_id, rnd,
                                       status="error", total_evol_time_s=elapsed,
                                       error=str(e))
                if is_probe:
                    await gate.probe_success(worker_id)  # not a rate-limit issue
                results.append((task_id, None))
                queue.task_done()
                break

        if not task_done_cleanly:
            # broke out due to timeout/error; already handled
            continue

        if is_probe:
            await gate.probe_success(worker_id)

        elapsed = time.monotonic() - t0
        for sr in synth_results:
            _update_synth_info(output_dir, sr.task_id_evol, task_id, rnd,
                               status="done", verdict=sr.verdict,
                               total_evol_time_s=elapsed)

        results.append((task_id, synth_results))
        queue.task_done()
        logger.info(
            "[W%d] %s done in %.0fs — %s", worker_id, task_id, elapsed,
            ", ".join(f"{r.task_id_evol}={r.verdict}" for r in synth_results),
        )


# ---------------------------------------------------------------------------
# Rollout worker
# ---------------------------------------------------------------------------

def _build_terminal_env_config(model_cfg: RolloutModelConfig, trial_root: str):
    """Build a TerminalEnvConfig from model entry + baked-in constants."""
    from seta_env.utils.configs import (
        TerminalEnvConfig, AgentConfig, ModelConfig, RuntimeConfig, EnvConfig,
    )
    agent_dict = dict(ROLLOUT_AGENT_CONFIG)
    if model_cfg.tito_enabled:
        agent_dict["agent"] = "tito_train_agent"

    return TerminalEnvConfig(
        agent=AgentConfig(**agent_dict),
        model=ModelConfig(
            model_platform=model_cfg.model_platform,
            model_type=model_cfg.model_type,
            url=model_cfg.url or "",
            api_key=model_cfg.api_key or "",
            tito_enabled=model_cfg.tito_enabled,
            tito_validate=model_cfg.tito_validate,
            model_config_dict=dict(model_cfg.model_config_dict or {}),
        ),
        runtime=RuntimeConfig(
            env_type=ROLLOUT_RUNTIME_CONFIG["env_type"],
            trial_root=trial_root,
        ),
        env=EnvConfig(
            reward_fn=ROLLOUT_ENV_CONFIG["reward_fn"],
            task_timeouts=ROLLOUT_ENV_CONFIG["task_timeouts"],
        ),
    )


async def _rollout_worker(
    worker_id: int,
    queue: asyncio.Queue,
    results: list,
    model_cfg: RolloutModelConfig,
    rollout_dir: str,
    n_trajs: int,
) -> None:
    """Consume tasks from queue and run GRPORollout on each."""
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
    )
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from seta_env.orchestrators.grpo_rollout import GRPORollout

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        task_id = item["task_id"]
        logger.info("[rollout W%d] Starting %s", worker_id, task_id)

        trial_root = os.path.join(rollout_dir, model_cfg.model_config_name, task_id)
        os.makedirs(trial_root, exist_ok=True)

        try:
            te_cfg = _build_terminal_env_config(model_cfg, trial_root)
            rollout = GRPORollout(cfg=te_cfg)
            t0 = time.monotonic()
            rollout_results = await rollout.run(
                task={
                    "task_name": item["task_name"],
                    "task_path": item["task_path"],
                    "instruction": item["instruction"],
                },
                n_trajs=n_trajs,
            )
            elapsed = time.monotonic() - t0
            results.append((item, rollout_results))
            logger.info("[rollout W%d] Done %s (%.0fs)", worker_id, task_id, elapsed)
        except Exception as exc:
            logger.error("[rollout W%d] Error on %s: %s", worker_id, task_id, exc, exc_info=True)
            results.append((item, None))
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# Verification helpers (ported from side branch)
# ---------------------------------------------------------------------------

def _read_tail(path: Path, max_chars: int = 8000) -> str:
    """Read a file's tail (last max_chars chars)."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    if len(text) > max_chars:
        return "... (truncated earlier output)\n" + text[-max_chars:]
    return text


def aggregate_failure_summary(rollout_task_dir: Path, task_path: Optional[Path] = None) -> str:
    """Build a complete evidence bundle for the trajectory judge.

    Embeds failure stats + traces, task source files (test, instruction,
    weights), and a tail of the representative terminal log, so the judge
    has everything in the prompt and needs no tool calls to gather evidence.
    """
    trial_dirs = sorted([
        d for d in rollout_task_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ])
    if not trial_dirs:
        return f"## Rollout Failure Summary: {rollout_task_dir.name}\nNo trials found.\n"

    n_passed = n_failed = 0
    test_failures: Dict[str, List[Dict[str, str]]] = {}

    for trial_dir in trial_dirs:
        reward_path = trial_dir / "verifier" / "reward.txt"
        ctrf_path = trial_dir / "verifier" / "ctrf.json"
        if not reward_path.exists():
            continue
        try:
            reward = float(reward_path.read_text().strip())
        except (ValueError, TypeError):
            continue

        if reward >= 1.0:
            n_passed += 1
        else:
            n_failed += 1

        if not ctrf_path.exists():
            continue
        try:
            ctrf = json.loads(ctrf_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        for t in ctrf.get("results", {}).get("tests", []):
            name = t.get("name", "unknown")
            if t.get("status") == "failed":
                trace = t.get("trace", "")
                if len(trace) > 800:
                    trace = trace[:800] + "\n... (truncated)"
                test_failures.setdefault(name, []).append({
                    "trial": trial_dir.name, "trace": trace,
                })

    n_total = n_passed + n_failed
    lines = [
        f"## Rollout Failure Summary: {rollout_task_dir.name}",
        f"Trials: {n_total} total, {n_passed} passed, {n_failed} failed", "",
    ]

    # --- Failure stats ---
    rep_trial = None
    if test_failures:
        sorted_failures = sorted(test_failures.items(), key=lambda kv: len(kv[1]), reverse=True)
        threshold = max(1, n_failed // 2)
        consistent = [(n, e) for n, e in sorted_failures if len(e) >= threshold]
        sporadic = [(n, e) for n, e in sorted_failures if len(e) < threshold]

        if consistent:
            lines.append(f"### Consistently Failing Tests (>={threshold}/{n_failed} failed trials):")
            for i, (name, entries) in enumerate(consistent, 1):
                rep = entries[0]
                if rep_trial is None:
                    rep_trial = rep["trial"]
                lines.append(f"\n**{i}. {name}** ({len(entries)}/{n_failed} failed trials)")
                lines.append("   Trace:")
                for tl in rep["trace"].splitlines()[:15]:
                    lines.append(f"   > {tl}")
            lines.append("")

        if sporadic:
            lines.append("### Sporadically Failing Tests:")
            for name, entries in sporadic:
                lines.append(f"- {name} ({len(entries)}/{n_failed} failed trials)")
            lines.append("")
    else:
        lines.append("No test-level failure data available.")
        lines.append("")

    # --- Task source files ---
    if task_path is not None:
        task_path = Path(task_path)

        def _embed(label: str, rel_path: str, max_chars: int = 6000) -> None:
            p = task_path / rel_path
            if not p.exists():
                return
            content = _read_tail(p, max_chars)
            lines.append(f"### {label} (`{rel_path}`)")
            lines.append("```")
            lines.append(content.rstrip())
            lines.append("```")
            lines.append("")

        _embed("Instruction", "instruction.md", max_chars=4000)
        _embed("Test file", "tests/test_outputs.py", max_chars=8000)
        _embed("Test weights", "weights.json", max_chars=2000)

    # --- Representative terminal log ---
    if rep_trial is None and trial_dirs:
        rep_trial = trial_dirs[0].name
    if rep_trial:
        log_path = rollout_task_dir / rep_trial / "terminal_logs" / "terminal.log"
        if log_path.exists():
            lines.append(f"### Representative agent terminal log (trial `{rep_trial}`, tail)")
            lines.append("```")
            lines.append(_read_tail(log_path, max_chars=10000).rstrip())
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def _collect_rollout_rows(rollout_dir: str, model_config_name: str) -> Dict[str, Dict[str, Any]]:
    """Return {task_id: metrics} for every task under a rollout model dir."""
    base = Path(rollout_dir) / model_config_name
    if not base.is_dir():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        rewards = []
        for traj_dir in sorted(task_dir.iterdir()):
            if not traj_dir.is_dir() or traj_dir.name.startswith("_"):
                continue
            rp = traj_dir / "verifier" / "reward.txt"
            if not rp.exists():
                continue
            try:
                rewards.append(float(rp.read_text().strip()))
            except (ValueError, TypeError):
                continue
        if not rewards:
            continue
        out[task_dir.name] = {
            "n_total": len(rewards),
            "n_passed": sum(1 for r in rewards if r >= 1.0),
            "n_failed": sum(1 for r in rewards if r < 1.0),
        }
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class EvolOrchestrator:
    """Config-driven orchestrator for the evolution pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self._filter_ids = (
            load_filter_csv(config.filter_csv) if config.filter_csv else None
        )

    # --- Evolve -----------------------------------------------------------

    async def run_evolve(self) -> dict:
        """Run all evolution rounds sequentially."""
        results: dict = {}
        for i, rnd in enumerate(self.config.evolve.rounds):
            label = rnd.name or f"round_{i+1}"
            logger.info("=== Evolve round: %s ===", label)
            os.makedirs(rnd.output_dir, exist_ok=True)

            work_items = collect_evol_queue(rnd, self.config)
            if not work_items:
                logger.info("[%s] Nothing to process.", label)
                results[label] = []
                continue

            q: asyncio.Queue = asyncio.Queue()
            for item in work_items:
                await q.put(item)
            n_workers = self.config.evolve.n_workers
            for _ in range(n_workers):
                await q.put(None)

            round_results: list = []
            gate = _RateLimitGate(pause_duration=600)

            workers = [
                _evol_worker(j, q, round_results, rnd, self.config, gate)
                for j in range(n_workers)
            ]
            await asyncio.gather(*workers)
            write_summary_csv(rnd.output_dir)
            results[label] = round_results
            logger.info("[%s] Done: %d tasks processed", label, len(round_results))

        return results

    # --- Rollout ----------------------------------------------------------

    async def run_rollout(self) -> dict:
        """Run rollout for all tasks_dirs × models.

        For each tasks_dir, all models run CONCURRENTLY (one pool per model).
        tasks_dirs themselves are processed sequentially.
        """
        rollout_cfg = self.config.rollout
        if not rollout_cfg.tasks_dirs or not rollout_cfg.models:
            logger.info("[rollout] No tasks_dirs or models configured.")
            return {}

        async def drain_one(tasks_dir: str, rd: str, model_cfg) -> tuple[str, list]:
            key = f"{os.path.basename(tasks_dir)}/{model_cfg.model_config_name}"
            gaps = find_rollout_gaps(
                tasks_dir, rd, model_cfg.model_config_name, self._filter_ids,
            )
            if not gaps:
                logger.info("[rollout] %s: no gaps.", key)
                return key, []
            logger.info("[rollout] %s: %d task(s) queued.", key, len(gaps))
            q: asyncio.Queue = asyncio.Queue()
            for g in gaps:
                await q.put(g)
            for _ in range(rollout_cfg.n_workers):
                await q.put(None)
            batch: list = []
            workers = [
                _rollout_worker(j, q, batch, model_cfg, rd, rollout_cfg.n_trajs)
                for j in range(rollout_cfg.n_workers)
            ]
            await asyncio.gather(*workers)
            return key, batch

        results: dict = {}
        for tasks_dir in rollout_cfg.tasks_dirs:
            rd = rollout_dir_for(tasks_dir)
            # All models for this tasks_dir run concurrently
            pair_results = await asyncio.gather(*[
                drain_one(tasks_dir, rd, m) for m in rollout_cfg.models
            ])
            for key, batch in pair_results:
                results[key] = batch
        return results

    # --- Verify -----------------------------------------------------------

    async def run_verify(self) -> dict:
        """Run trajectory judge on all-fail rolled-out tasks."""
        verify_cfg = self.config.verify
        if not verify_cfg.tasks_dirs:
            logger.info("[verify] No tasks_dirs configured.")
            return {}

        results: dict = {}
        judge = ClaudeTrajectoryJudgeAgent()

        for tasks_dir in verify_cfg.tasks_dirs:
            rd = rollout_dir_for(tasks_dir)
            rows = _collect_rollout_rows(rd, verify_cfg.model_config_name)
            if not rows:
                logger.info("[verify] No rollout data in %s", rd)
                continue

            targets = []
            for task_id, metrics in rows.items():
                n_total = metrics["n_total"]
                if n_total == 0:
                    continue
                pass_rate = metrics["n_passed"] / n_total
                if pass_rate > verify_cfg.max_pass_rate:
                    continue
                if self._filter_ids and parse_variant_root(task_id) not in self._filter_ids:
                    continue
                task_path = os.path.join(tasks_dir, task_id)
                rollout_task_dir = os.path.join(rd, verify_cfg.model_config_name, task_id)
                if not os.path.isdir(task_path) or not os.path.isdir(rollout_task_dir):
                    continue
                # Skip already judged
                info = read_synth_info(tasks_dir, task_id)
                if info and info.get("traj_judge"):
                    continue
                targets.append({
                    "task_id": task_id, "task_path": task_path,
                    "rollout_task_dir": rollout_task_dir,
                    "pass_rate": pass_rate,
                })

            logger.info("[verify] %s: %d task(s) to judge", os.path.basename(tasks_dir), len(targets))

            task_results: list = []
            i = 0
            while i < len(targets):
                target = targets[i]
                task_id = target["task_id"]
                summary = aggregate_failure_summary(
                    Path(target["rollout_task_dir"]),
                    task_path=Path(target["task_path"]),
                )
                try:
                    result = await judge.judge(
                        task_id=task_id,
                        task_path=target["task_path"],
                        rollout_dir=target["rollout_task_dir"],
                        failure_summary=summary,
                    )
                except ClaudeRateLimitError:
                    logger.error("[verify] Rate limit on %s — pausing 600s then retrying", task_id)
                    await asyncio.sleep(600)
                    continue  # retry same task (don't advance i)
                except Exception as exc:
                    logger.error("[verify] Error judging %s: %s", task_id, exc, exc_info=True)
                    task_results.append((task_id, None))
                    i += 1
                    continue
                i += 1

                verdict = result["verdict"]
                # Update synth_info
                info = read_synth_info(tasks_dir, task_id) or {}
                info["traj_judge"] = verdict
                info["traj_judge_timestamp"] = datetime.now(timezone.utc).isoformat()
                if verdict == "design_flaw":
                    info["verdict"] = "FAIL"
                    info["failure_source"] = "traj_analysis"
                write_synth_info(tasks_dir, task_id, info)
                task_results.append((task_id, verdict))
                logger.info("[verify] %s → %s", task_id, verdict)

            write_summary_csv(tasks_dir)
            results[os.path.basename(tasks_dir)] = task_results

        return results

    # --- Background pollers (concurrent with evolve) ------------------------

    async def _rollout_poller(self, results: dict, poll_interval: int = 30) -> None:
        """Poll for new PASS variants and roll them out. Runs as background task."""
        rollout_cfg = self.config.rollout
        if not rollout_cfg.tasks_dirs or not rollout_cfg.models:
            return

        async def drain(tasks_dir: str, rd: str, model_cfg) -> None:
            # Source of truth is disk state via find_rollout_gaps — errored
            # tasks stay as gaps and get retried on the next tick. Ticks
            # cannot overlap (the while-loop awaits gather), so there is no
            # risk of double-enqueueing the same task.
            key = f"{os.path.basename(tasks_dir)}/{model_cfg.model_config_name}"
            gaps = find_rollout_gaps(
                tasks_dir, rd, model_cfg.model_config_name, self._filter_ids,
            )
            if not gaps:
                return
            logger.info("[rollout_poller] %d task(s) to roll out for %s", len(gaps), key)
            q: asyncio.Queue = asyncio.Queue()
            for g in gaps:
                await q.put(g)
            for _ in range(rollout_cfg.n_workers):
                await q.put(None)
            batch: list = []
            workers = [
                _rollout_worker(j, q, batch, model_cfg, rd, rollout_cfg.n_trajs)
                for j in range(rollout_cfg.n_workers)
            ]
            await asyncio.gather(*workers)
            results.setdefault(key, []).extend(batch)

        while True:
            await asyncio.sleep(poll_interval)
            # Drain every (tasks_dir, model) pair concurrently so one model
            # with a long backlog can't starve another.
            await asyncio.gather(*[
                drain(tasks_dir, rollout_dir_for(tasks_dir), m)
                for tasks_dir in rollout_cfg.tasks_dirs
                for m in rollout_cfg.models
            ])

    async def _verify_poller(self, results: dict, poll_interval: int = 60) -> None:
        """Poll for completed rollouts needing verification. Runs as background task.

        On rate limit, pauses for 600s then retries — does NOT exit the poller.
        Tasks that error (non-rate-limit) are retried on the next poll iteration.
        """
        verify_cfg = self.config.verify
        if not verify_cfg.tasks_dirs:
            return

        judge = ClaudeTrajectoryJudgeAgent()

        while True:
            await asyncio.sleep(poll_interval)
            n_scanned = n_judged = 0
            for tasks_dir in verify_cfg.tasks_dirs:
                rd = rollout_dir_for(tasks_dir)
                rows = _collect_rollout_rows(rd, verify_cfg.model_config_name)

                for task_id, metrics in rows.items():
                    n_total = metrics["n_total"]
                    if n_total == 0:
                        continue
                    if metrics["n_passed"] / n_total > verify_cfg.max_pass_rate:
                        continue
                    if self._filter_ids and parse_variant_root(task_id) not in self._filter_ids:
                        continue
                    task_path = os.path.join(tasks_dir, task_id)
                    rollout_task_dir = os.path.join(rd, verify_cfg.model_config_name, task_id)
                    if not os.path.isdir(task_path) or not os.path.isdir(rollout_task_dir):
                        continue
                    info = read_synth_info(tasks_dir, task_id)
                    if info and info.get("traj_judge"):
                        continue

                    n_scanned += 1
                    summary = aggregate_failure_summary(
                        Path(rollout_task_dir), task_path=Path(task_path),
                    )
                    try:
                        result = await judge.judge(
                            task_id=task_id, task_path=task_path,
                            rollout_dir=rollout_task_dir, failure_summary=summary,
                        )
                    except ClaudeRateLimitError:
                        logger.error("[verify_poller] Rate limit on %s — pausing 600s", task_id)
                        await asyncio.sleep(600)
                        continue  # retry this task on next poll iteration
                    except Exception as exc:
                        logger.error("[verify_poller] Error on %s: %s", task_id, exc, exc_info=True)
                        continue  # retry on next poll iteration

                    verdict = result["verdict"]
                    info = read_synth_info(tasks_dir, task_id) or {}
                    info["traj_judge"] = verdict
                    info["traj_judge_timestamp"] = datetime.now(timezone.utc).isoformat()
                    if verdict == "design_flaw":
                        info["verdict"] = "FAIL"
                        info["failure_source"] = "traj_analysis"
                    write_synth_info(tasks_dir, task_id, info)
                    results.setdefault(os.path.basename(tasks_dir), []).append((task_id, verdict))
                    n_judged += 1
                    logger.info("[verify_poller] %s → %s", task_id, verdict)

            if n_scanned > 0:
                logger.info("[verify_poller] scan: %d scanned, %d judged", n_scanned, n_judged)

    # --- All stages -------------------------------------------------------

    async def _cancel_task(self, task: Optional[asyncio.Task]) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def run(self, stage: str = "all") -> dict:
        """Run pipeline stage(s). stage: evolve|rollout|verify|all.

        When stage="all", all three stages run concurrently: rollout and
        verify poll for new work while evolve produces it. After evolve
        finishes, a final sweep of rollout and verify runs.
        """
        results: dict = {}

        if stage == "all":
            rollout_results: dict = {}
            verify_results: dict = {}

            # Start pollers as background tasks
            rollout_task = asyncio.create_task(self._rollout_poller(rollout_results))
            verify_task = asyncio.create_task(self._verify_poller(verify_results))

            # Run evolve (blocks until all rounds done)
            results["evolve"] = await self.run_evolve()

            # Cancel pollers
            await self._cancel_task(rollout_task)
            await self._cancel_task(verify_task)

            # Final sweep to catch anything the pollers missed
            final_rollout = await self.run_rollout()
            for k, v in final_rollout.items():
                rollout_results.setdefault(k, []).extend(v)
            results["rollout"] = rollout_results

            final_verify = await self.run_verify()
            for k, v in final_verify.items():
                verify_results.setdefault(k, []).extend(v)
            results["verify"] = verify_results
        else:
            if stage == "evolve":
                results["evolve"] = await self.run_evolve()
            elif stage == "rollout":
                results["rollout"] = await self.run_rollout()
            elif stage == "verify":
                results["verify"] = await self.run_verify()

        return results

    def print_summary(self, results: dict) -> None:
        print(f"\n{'=' * 60}")
        print("EVOLUTION PIPELINE SUMMARY")
        print(f"{'=' * 60}")

        if "evolve" in results:
            for label, round_results in results["evolve"].items():
                total = len(round_results)
                success = sum(1 for _, sr in round_results if sr is not None)
                n_pass = sum(
                    1 for _, sr_list in round_results
                    if sr_list for sr in sr_list if sr.verdict == "PASS"
                )
                print(f"  [{label}] {total} tasks, {success} succeeded, {n_pass} PASS variants")

        if "rollout" in results:
            for key, batch in results["rollout"].items():
                done = sum(1 for _, r in batch if r is not None)
                print(f"  [rollout {key}] {done}/{len(batch)} completed")

        if "verify" in results:
            for key, verdicts in results["verify"].items():
                design_flaws = sum(1 for _, v in verdicts if v == "design_flaw")
                too_hard = sum(1 for _, v in verdicts if v == "too_hard")
                print(f"  [verify {key}] {design_flaws} design_flaw, {too_hard} too_hard")

        print(f"{'=' * 60}\n")
