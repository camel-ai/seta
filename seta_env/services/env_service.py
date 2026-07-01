"""Env Service — runs TerminalEnvironment.step() as a remote FastAPI service.

Deploys with its own TerminalEnvConfig (agent, runtime, env settings).
Caller only sends the task payload + model URL/api_key (from ProxySession).
Config can be updated via POST /config without redeployment.

Usage:
    ENV_SERVICE_CONFIG=config.yaml MAX_SLOTS=16 ENV_SERVICE_API_KEY=dev-key \
        uvicorn seta_env.services.env_service:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from seta_env.environments.terminal_env import TerminalEnvironment
from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
# Raised by harbor's Daytona canary when a snapshot built but never became
# schedulable (case B); build_fn keeps the snapshot instead of deleting it.
from harbor.environments.daytona import SnapshotPlacementDeferred

# Pre-import transformers.AutoTokenizer at module load time.
# In transformers 5.8.0 the top-level lazy importer is NOT thread-safe on
# first access — concurrent `from transformers import AutoTokenizer` calls
# from multiple worker threads race and one of them raises ImportError.
# Resolving the lazy name once here (on the main thread, before workers
# spin up) makes subsequent concurrent imports no-op attribute lookups.
try:
    from transformers import AutoTokenizer as _eager_AutoTokenizer  # noqa: F401
except ImportError:
    pass  # the worker that needs it will surface a clearer error later
from seta_env.utils.configs import (
    AgentConfig,
    EnvConfig,
    ModelConfig,
    RuntimeConfig,
    TerminalEnvConfig,
    build_agent_config,
    build_env_config,
    build_model_config,
)

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────

MAX_SLOTS = int(os.environ.get("MAX_SLOTS", "16"))
# P0 leak-cure toggle: when set, each /step runs in a KILLABLE forkserver
# subprocess (seta_env.services._step_subprocess) so STEP_TIMEOUT can SIGKILL a
# hung step instead of leaking an un-cancellable worker thread. Default OFF keeps
# the original in-thread behavior.
_USE_STEP_SUBPROCESS = os.environ.get("STEP_USE_SUBPROCESS", "0") == "1"
# Hard wall-clock cap on a single /step (trajectory). A step blocked past this
# (e.g. on an unresponsive Daytona sandbox with no lower-level timeout) is
# force-failed so its semaphore slot is released instead of leaked forever.
# Normal trajectories finish in ~17min; default 30min leaves headroom.
STEP_TIMEOUT_SECONDS = float(os.environ.get("STEP_TIMEOUT_SECONDS", "1800"))
# Hard cap on the BUILD stage (sandbox create). A failing/hung build (e.g. a
# doomed "No available runners" retry loop) must not hold its slot — and via
# BuildGate's single-flight it would otherwise also block every same-task
# waiter. Bounds build-stage slot occupation regardless of harbor's retries.
# One legit build attempt is build_timeout_sec (600s default), so 600 here
# allows a real build while killing a stuck one.
BUILD_TIMEOUT_SECONDS = float(os.environ.get("BUILD_TIMEOUT_SECONDS", "600"))
API_KEY = os.environ.get("ENV_SERVICE_API_KEY", "")
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/data/harbor/dataset"))
HARBOR_ROOT = Path(os.environ.get("HARBOR_ROOT", "/tmp/harbor"))
SETUP_DATASET = os.environ.get("SETUP_DATASET", "")  # comma-separated dataset names to download on startup
GC_INTERVAL_SEC = 300
BUILD_TTL_SEC = 3600


# ── Per-step thread pool for isolation ──────────────────────────────────────
#
# Each /step's TerminalEnvironment.step() execution is dispatched to one of
# MAX_SLOTS worker threads, each of which spins up its own asyncio event loop
# via asyncio.run(). This guarantees:
#
#   - One step's CPU-bound work (e.g. Daytona SDK's demux_log parsing a
#     megabyte of interleaved stdout/stderr) cannot freeze the env_service's
#     main event loop.
#   - One step's blocked tool call cannot block other steps' tool calls —
#     each step has its own loop.
#   - All async resources created inside te.step() (DeepSeekV4SGLangModel's
#     httpx.AsyncClient, harbor's AsyncDaytona) are created INSIDE the
#     per-step loop, satisfying httpx/asyncio's loop-affinity requirements.
#
# Sized to MAX_SLOTS so concurrent step counts can't exceed configured limit
# even if the semaphore is bypassed somehow. The thread name prefix makes
# stuck-step diagnosis (py-spy) much easier.
# max_workers is intentionally LARGER than MAX_SLOTS: when a step is force-failed
# by STEP_TIMEOUT_SECONDS, asyncio stops awaiting it but the worker thread keeps
# running (a thread can't be cancelled) until its blocked Daytona call unwinds.
# The semaphore (MAX_SLOTS) still bounds live concurrency; this slack just keeps
# such zombie threads from re-exhausting the pool and re-creating the deadlock.
_STEP_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.environ.get("STEP_EXECUTOR_WORKERS", str(MAX_SLOTS * 4))),
    thread_name_prefix="env-step",
)


def _load_terminal_env_config() -> TerminalEnvConfig:
    """Load TerminalEnvConfig from YAML file or use defaults."""
    config_path = os.environ.get("ENV_SERVICE_CONFIG", "")
    if config_path and Path(config_path).exists():
        data = yaml.safe_load(Path(config_path).read_text())
        te_data = data.get("terminal_env", data)
        # Build model config: keep as raw dict to preserve extra keys
        # like tito_enabled/tito_validate that ModelConfig doesn't have.
        # URL/api_key are overridden per /step request.
        model_cfg = None
        if "model" in te_data:
            model_data = te_data["model"]
            # Only create ModelConfig from standard fields
            model_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
            standard = {k: v for k, v in model_data.items() if k in model_fields}
            model_cfg = ModelConfig(**standard)
        cfg = TerminalEnvConfig(
            agent=AgentConfig(**te_data["agent"]) if "agent" in te_data else AgentConfig(),
            model=model_cfg,
            runtime=RuntimeConfig(**te_data["runtime"]) if "runtime" in te_data else RuntimeConfig(),
            env=EnvConfig(**te_data["env"]) if "env" in te_data else EnvConfig(),
        )
        # Store raw model dict for extra keys (tito_enabled, tito_validate)
        cfg._raw_model_config = te_data.get("model", {})
        logger.info("Loaded config from %s", config_path)
        return cfg
    return TerminalEnvConfig(model=None)


# ── Build Gate ──────────────────────────────────────────────────────────────


@dataclass
class BuildState:
    status: Literal["building", "built", "failed"]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None


class BuildGate:
    """Per-task_name single-flight build coordination.

    First caller builds; subsequent callers for the same task_name wait.
    Different task_names build in parallel (independent Events).
    """

    def __init__(self):
        self._gate_lock = asyncio.Lock()
        self._registry: dict[str, BuildState] = {}
        self._timestamps: dict[str, float] = {}

    async def ensure_built(self, task_name: str, build_fn) -> None:
        async with self._gate_lock:
            if task_name not in self._registry:
                self._registry[task_name] = BuildState(status="building")
                is_builder = True
            else:
                is_builder = False
            state = self._registry[task_name]

        if is_builder:
            try:
                # Hard wall-clock cap: a stuck/looping build can't pin its
                # build-permit (and, via the shared event below, all same-task
                # waiters) indefinitely. On timeout the build is abandoned and
                # recorded failed so the trajectory recycles.
                #
                # _build_semaphore bounds concurrent declarative builds so build
                # creates stay inside Daytona's shared create budget and don't
                # starve trajectory creates. The else-branch waiters only await
                # the event — they take NO build-permit — so the builder never
                # blocks its own group and the permit count == distinct tasks
                # building. This whole block runs OUTSIDE _slot_semaphore, so a
                # build never occupies a sandbox slot.
                async with _build_semaphore:
                    _bump_building(1)
                    try:
                        await asyncio.wait_for(build_fn(), timeout=BUILD_TIMEOUT_SECONDS)
                    finally:
                        _bump_building(-1)
                state.status = "built"
            except Exception as e:
                state.status = "failed"
                state.error = (
                    f"build timeout after {BUILD_TIMEOUT_SECONDS:.0f}s"
                    if isinstance(e, (asyncio.TimeoutError, TimeoutError))
                    else str(e)
                )
                logger.error("Build failed for %s: %s", task_name, state.error)
            finally:
                # ALWAYS set the event — even on timeout — so same-task waiters
                # are never stranded by a hung builder.
                self._timestamps[task_name] = time.monotonic()
                state.event.set()
        else:
            if state.status == "building":
                # Backstop: never wait on a builder longer than its own cap.
                try:
                    await asyncio.wait_for(
                        state.event.wait(), timeout=BUILD_TIMEOUT_SECONDS + 30
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    raise RuntimeError(f"Build wait timeout for {task_name}")

        if state.status == "failed":
            raise RuntimeError(f"Build failed for {task_name}: {state.error}")

    def clear(self, older_than: float = BUILD_TTL_SEC) -> int:
        """Expire stale registry entries WITHOUT forgetting successful builds.

        A 'built' entry mirrors a live Daytona snapshot and MUST persist for the
        whole service lifetime so the build gate is respected — a snapshot already
        on Daytona is reused via the fast snapshot.get path, never rebuilt. Built
        entries are removed ONLY by snapshot eviction (_ensure_snapshot_quota),
        which pops them in lockstep with snapshot.delete. Here we only expire
        'failed'/'building' entries past the TTL (so a transiently-failed or
        crashed-builder task can rebuild). older_than=0 (full reset via /cleanup)
        still clears everything, built included.
        """
        now = time.monotonic()
        to_remove = []
        for k, t in list(self._timestamps.items()):
            st = self._registry.get(k)
            if older_than > 0 and st is not None and st.status == "built":
                continue  # respect the build gate; snapshot eviction owns these
            if now - t > older_than:
                to_remove.append(k)
        for k in to_remove:
            self._registry.pop(k, None)
            self._timestamps.pop(k, None)
        return len(to_remove)

    @property
    def stats(self) -> dict:
        return {
            "building": sum(1 for s in self._registry.values() if s.status == "building"),
            "built": sum(1 for s in self._registry.values() if s.status == "built"),
            "failed": sum(1 for s in self._registry.values() if s.status == "failed"),
        }




# ── Global state ────────────────────────────────────────────────────────────

_build_gate = BuildGate()
_slot_semaphore = asyncio.Semaphore(MAX_SLOTS)
_active_count = 0

# ── Build/create decoupling — make a "slot" mean a RUNNING sandbox ────────────
# A declarative build warms Daytona's content-hash image cache by creating ONE
# throwaway build-sandbox (daytona.create, ~minutes cold / ~seconds warm) and
# deleting it; the group's trajectory creates then hit that warm cache. That
# build MUST NOT occupy one of the MAX_SLOTS sandbox slots while it runs — else
# a handful of slow builds park ~half the slots holding no live sandbox (the
# low-concurrency bug). So the build gate runs OUTSIDE _slot_semaphore, and the
# slot is taken ONLY for the create+run+eval phase. Two extra gates coordinate
# it without stampeding Daytona's shared ~500-creates/min budget (429 ceiling):
#
#   _build_semaphore    caps concurrent declarative builds. Each builder holds a
#                       Daytona create-permit (HARBOR_DAYTONA_MAX_CREATES) for the
#                       whole warmup-create, so BUILD_CONCURRENCY is kept well
#                       BELOW that cap so build-creates never starve the (fast,
#                       cache-hit) trajectory-creates of the shared create budget.
#   _inflight_semaphore backpressure: caps TOTAL steps in the pipeline
#                       (building + built-and-waiting-for-slot + running) to
#                       MAX_SLOTS + INFLIGHT_LEAD. Building therefore runs at most
#                       INFLIGHT_LEAD ahead of the slots: enough that a freed slot
#                       finds a ready-built step to fill immediately, but never so
#                       far ahead that we burn create budget / image cache on work
#                       the slots can't consume soon. This is the "always keep a
#                       few declarative builds ready" buffer, with a hard ceiling.
BUILD_CONCURRENCY = int(os.environ.get("BUILD_CONCURRENCY", "8"))
INFLIGHT_LEAD = int(os.environ.get("INFLIGHT_LEAD", "48"))
_build_semaphore = asyncio.Semaphore(BUILD_CONCURRENCY)
_inflight_semaphore = asyncio.Semaphore(MAX_SLOTS + INFLIGHT_LEAD)
_building_count = 0   # builders currently inside build_fn (observability)
_inflight_count = 0   # steps holding an inflight permit (observability)


def _bump_building(delta: int) -> None:
    global _building_count
    _building_count += delta


def _bump_inflight(delta: int) -> None:
    global _inflight_count
    _inflight_count += delta

# ── Admission pacer: the permanent cure for the batch-boundary thundering herd ─
# After any moment when many slots free at once -- a service restart, or a batch
# boundary where ~MAX_SLOTS trajectories finish together -- all those steps would
# otherwise ENTER the reset/create/eval phases in lockstep and stampede Daytona's
# runner proxy + create gate, producing the empty-message "1_reset_env:
# TimeoutError" / "Failed to add tests directory" storms (norunner=0%, i.e. NOT a
# capacity problem -- a synchronization problem).
#
# We space successive step admissions by a small min-interval (+jitter). The
# critical property: each trajectory takes ~constant wall-time, so staggering
# ENTRY automatically staggers EXIT (eval), which staggers the NEXT entry -- the
# de-phasing is SELF-SUSTAINING across every batch boundary. The pool, once
# paced, stays permanently de-phased; a restart can no longer re-synchronize it.
#
# At steady state this is essentially free: with MAX_SLOTS slots and ~minutes-long
# trajectories, step inter-arrival is several seconds -- far longer than the
# interval -- so the pacer is almost never the binding constraint. It only ever
# engages to smooth a stampede. Set ADMISSION_INTERVAL_SEC=0 to disable.
_ADMISSION_INTERVAL = float(os.environ.get("ADMISSION_INTERVAL_SEC", "0.8"))
_admission_lock = asyncio.Lock()
_last_admission = 0.0


async def _pace_admission():
    """Serialize step entry so no more than ~1 step begins reset per interval.
    Holding the lock across the sleep is intentional: it makes queued admissions
    drain one-per-interval (a leaky bucket), spreading a 192-wide stampede over
    ~MAX_SLOTS * interval seconds instead of firing them all at once."""
    global _last_admission
    if _ADMISSION_INTERVAL <= 0:
        return
    async with _admission_lock:
        now = time.monotonic()
        wait = _last_admission + _ADMISSION_INTERVAL - now
        if wait > 0:
            # jitter avoids re-clustering and adds entropy to the de-phasing
            await asyncio.sleep(wait + random.uniform(0, _ADMISSION_INTERVAL * 0.5))
        _last_admission = time.monotonic()
_dataset_locks: dict[str, asyncio.Lock] = {}
_te_config: TerminalEnvConfig = _load_terminal_env_config()

# ── Snapshot LRU eviction (for daytona env_type) ────────────────────────────
#
# Daytona caps user-created snapshots per tier (Tier 3 = 100). For a 988-task
# training dataset, each unique task registers a named snapshot — we'd hit
# the cap in the first ~100 tasks and all subsequent builds would fail with
# "Snapshot quota exceeded". To stay under the cap while still leveraging the
# fast snapshot.get → CreateSandboxFromSnapshotParams path, we evict the
# least-recently-used snapshot when approaching the quota.
#
# Two pieces of state, both maintained by the /step handler:
#   - _last_used_at[task_name] : monotonic timestamp, refreshed on every /step
#                                 entry → drives LRU ordering.
#   - _active_per_task[task_name] : reference count of in-flight /step calls
#                                    for this task → eviction NEVER touches a
#                                    task with active_per_task > 0 (otherwise
#                                    we'd race with a trajectory that's about
#                                    to call CreateSandboxFromSnapshotParams).
#
# `_ensure_snapshot_quota` is called inside `build_fn` BEFORE `rt.build()` so
# we only evict at build-time (when we know we're about to register a new
# snapshot). `_eviction_lock` serialises concurrent eviction passes.
_last_used_at: dict[str, float] = {}
_active_per_task: dict[str, int] = {}
_eviction_lock = asyncio.Lock()
SNAPSHOT_QUOTA = int(os.environ.get("DAYTONA_SNAPSHOT_QUOTA", "100"))
SNAPSHOT_QUOTA_HEADROOM = int(os.environ.get("DAYTONA_SNAPSHOT_QUOTA_HEADROOM", "16"))
# When > 0, every _ensure_snapshot_quota() pass first deletes any user-owned
# snapshot whose Daytona-side lastUsedAt is older than this many hours, even
# if we did NOT build it (other env_service instances and manual sessions on
# the same Daytona account also count against the per-tier quota; only the
# Daytona lastUsedAt timestamp is observable across producers). Skip-if-in-use
# is still enforced via _active_per_task. 0 = disabled (default).
SNAPSHOT_EVICT_AGE_HOURS = float(os.environ.get("DAYTONA_SNAPSHOT_EVICT_AGE_HOURS", "0"))


async def _ensure_snapshot_quota() -> None:
    """Before a new snapshot.create(), evict LRU built snapshots not in use.

    Key design choice — the count of "consumed" snapshots is the
    AUTHORITATIVE TOTAL FROM DAYTONA, not a local registry view. Other
    env_service runs (different Daytona account workspaces, prior runs that
    didn't clean up, manual probes) may also have snapshots that count
    against the same per-tier quota. We must observe the actual pull
    status to decide eviction, otherwise we under-count and overrun.

    Eviction targets remain limited to OUR registry: we can only delete
    snapshots we registered (and recorded in BuildGate). External
    snapshots stay untouched.

    Best-effort: never raises. If we can't free enough slots, the
    downstream snapshot.create() will surface the quota error itself.
    """
    async with _eviction_lock:
        try:
            from daytona import AsyncDaytona
            daytona = AsyncDaytona()
            try:
                # 1. Authoritative count from Daytona — paginate through all
                #    user snapshots. Filter out well-known SYSTEM/preset
                #    snapshots (daytona-*, daytonaio/*, android-*); the
                #    Daytona quota counts user-created snapshots only.
                #    Anything else is "user-created" whether or not we made
                #    it, and consumes our quota slot.
                total_user_snapshots = 0
                all_user_snapshots: list = []
                page_n = 1
                while True:
                    page = await daytona.snapshot.list(limit=200, page=page_n)
                    items = list(getattr(page, "items", page) or [])
                    user_items = [
                        s for s in items
                        if not s.name.startswith(("daytona-", "daytonaio/", "android-"))
                    ]
                    total_user_snapshots += len(user_items)
                    all_user_snapshots.extend(user_items)
                    if len(items) < 200:
                        break
                    page_n += 1

                # 1a. Age-based proactive eviction (optional, env-var gated).
                #     Runs even when under quota — its purpose is to keep the
                #     shared Daytona account clean across env_service
                #     instances. Filters by Daytona lastUsedAt (authoritative
                #     across producers) rather than our local _last_used_at.
                #     Skip snapshots an in-flight /step is using.
                if SNAPSHOT_EVICT_AGE_HOURS > 0:
                    import datetime as _dt
                    now_dt = _dt.datetime.now(_dt.timezone.utc)
                    threshold_s = SNAPSHOT_EVICT_AGE_HOURS * 3600
                    age_evict: list[tuple[float, object]] = []
                    for s in all_user_snapshots:
                        ts = getattr(s, "last_used_at", None) or getattr(s, "lastUsedAt", None)
                        if ts is None:
                            continue
                        try:
                            last = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        except Exception:
                            continue
                        age = (now_dt - last).total_seconds()
                        if age < threshold_s:
                            continue
                        if _active_per_task.get(s.name, 0) > 0:
                            continue
                        age_evict.append((age, s))
                    age_evict.sort(reverse=True)
                    for age, snap in age_evict:
                        try:
                            await daytona.snapshot.delete(snap)
                            _build_gate._registry.pop(snap.name, None)
                            _build_gate._timestamps.pop(snap.name, None)
                            _last_used_at.pop(snap.name, None)
                            total_user_snapshots -= 1
                            logger.info(
                                "Age-evicted snapshot %s (idle %.1fh, threshold %.1fh)",
                                snap.name, age / 3600, SNAPSHOT_EVICT_AGE_HOURS,
                            )
                        except Exception as e:
                            logger.warning("Age-evict failed for %s: %s", snap.name, e)

                if total_user_snapshots <= SNAPSHOT_QUOTA - SNAPSHOT_QUOTA_HEADROOM:
                    return  # plenty of headroom

                # 2. Eviction candidates: ONLY snapshots we know we own
                #    (in our BuildGate registry), built, currently idle.
                #    External snapshots may also be consuming quota but
                #    are not ours to evict.
                evictable: list[tuple[float, object]] = []
                for s in all_user_snapshots:
                    if s.name not in _build_gate._registry:
                        continue  # not ours — leave alone
                    state = _build_gate._registry.get(s.name)
                    if state is None or state.status != "built":
                        continue
                    if _active_per_task.get(s.name, 0) > 0:
                        continue
                    evictable.append((_last_used_at.get(s.name, 0.0), s))
                evictable.sort(key=lambda kv: kv[0])

                n_to_evict = total_user_snapshots - (SNAPSHOT_QUOTA - SNAPSHOT_QUOTA_HEADROOM)
                evicted = 0
                now = time.monotonic()
                for last_used, snap in evictable:
                    if evicted >= n_to_evict:
                        break
                    try:
                        await daytona.snapshot.delete(snap)
                        _build_gate._registry.pop(snap.name, None)
                        _build_gate._timestamps.pop(snap.name, None)
                        _last_used_at.pop(snap.name, None)
                        evicted += 1
                        logger.info(
                            "LRU evicted snapshot %s (idle %.0fs); "
                            "daytona total %d, target %d",
                            snap.name, now - last_used,
                            total_user_snapshots - evicted,
                            SNAPSHOT_QUOTA - SNAPSHOT_QUOTA_HEADROOM,
                        )
                    except Exception as e:
                        logger.warning("LRU evict failed for %s: %s", snap.name, e)
                if evicted < n_to_evict:
                    logger.warning(
                        "snapshot eviction freed %d/%d slots — daytona total "
                        "%d, quota %d; external snapshots may be holding "
                        "remaining slots",
                        evicted, n_to_evict,
                        total_user_snapshots - evicted, SNAPSHOT_QUOTA,
                    )
            finally:
                try: await daytona.close()
                except Exception: pass
        except Exception as e:
            logger.warning("_ensure_snapshot_quota outer failure: %s", e)


async def _reap_stale_sandboxes() -> None:
    """Startup: delete sandboxes orphaned by a PRIOR lifetime so the new process
    begins with a CLEAN Daytona CPU budget.

    A restart leaves the previous process's sandboxes RUNNING — they keep
    occupying the 250-CPU account quota. The new process then ramps up creating
    sandboxes and collides with "Total CPU limit exceeded" (and the resulting
    create-retry storm trips Daytona's "Too Many Requests" limiter) until the old
    ones drain (~15min via auto-stop). THAT collision is the post-relaunch warm-up
    failure spike. We delete every harbor-owned sandbox up front: at startup none
    belong to the new lifetime yet, so all are stale. Best-effort + bounded.
    """
    try:
        from daytona import AsyncDaytona
        from harbor.environments.daytona import (
            _HARBOR_OWNER_LABEL,
            _delete_sandbox_safe,
        )
        daytona = AsyncDaytona()
        try:
            res = await daytona.list()
            items = (
                res.items if hasattr(res, "items") and not isinstance(res, list)
                else res
            ) or []
            stale = [
                s for s in items
                if _HARBOR_OWNER_LABEL in (getattr(s, "labels", None) or {})
            ]
            logger.info(
                "Startup: reaping %d stale harbor sandbox(es) to free the CPU "
                "quota before serving (prevents warm-up CPU-limit collision)",
                len(stale),
            )
            sem = asyncio.Semaphore(16)

            async def _del(s):
                async with sem:
                    await _delete_sandbox_safe(s, logger=logger)

            await asyncio.gather(*[_del(s) for s in stale], return_exceptions=True)
            logger.info("Startup: stale-sandbox reap complete")
        finally:
            try: await daytona.close()
            except Exception: pass
    except Exception as e:
        logger.warning("startup stale-sandbox reap skipped: %s", e)


# Gentle-drain rate: delete one prior-lifetime sandbox per this many seconds. Slow
# enough that the new ramp's creates outpace the deletes (net demand POSITIVE, so
# Daytona holds/grows the runner pool instead of scaling it down), fast enough to
# clear the CPU quota in a couple minutes.
_ORPHAN_DRAIN_INTERVAL = float(os.environ.get("ORPHAN_DRAIN_INTERVAL_SEC", "1.5"))


async def _drain_orphans_gradually(orphans: list, client) -> None:
    """Background: delete the captured prior-lifetime sandboxes one at a time with
    a small gap, so CPU+runners free SMOOTHLY as the new ramp claims them — no
    'Total CPU limit' collision (gradual free) and no runner scale-down (deletes
    stay slower than creates, demand stays positive). The orphan set is snapshotted
    BEFORE serving, so this never deletes a new-lifetime sandbox. Best-effort.
    """
    from harbor.environments.daytona import _delete_sandbox_safe
    try:
        n = len(orphans)
        if n:
            logger.info(
                "Gentle-drain: deleting %d orphan(s) at ~1 per %.1fs (smooth "
                "CPU/runner handoff to the new ramp)", n, _ORPHAN_DRAIN_INTERVAL,
            )
        for i, s in enumerate(orphans):
            try:
                await _delete_sandbox_safe(s, logger=logger)
            except Exception as e:
                logger.warning("gentle-drain delete failed: %s", e)
            if n and (i + 1) % 25 == 0:
                logger.info("Gentle-drain progress: %d/%d", i + 1, n)
            await asyncio.sleep(_ORPHAN_DRAIN_INTERVAL)
        if n:
            logger.info("Gentle-drain: complete (%d sandbox(es) freed)", n)
    except asyncio.CancelledError:
        pass
    finally:
        if client is not None:
            try: await client.close()
            except Exception: pass


async def _reconcile_build_gate_from_daytona() -> None:
    """Make the BuildGate respect builds from a PRIOR service lifetime.

    On startup the in-memory BuildGate registry is empty — but Daytona may still
    hold ACTIVE snapshots we built before the restart. Without reconciliation the
    first /step for each such task re-enters build_fn (a wasted build-gate cycle)
    and LRU eviction can't recognise those snapshots as ours. We list existing
    user snapshots and mark each ACTIVE one 'built', so the first /step skips
    build_fn and goes straight to the fast CreateSandboxFromSnapshot path — a
    restart never cold-rebuilds an already-built task (this is what keeps the
    post-relaunch warm-up failure-free). Best-effort: never raises.
    """
    try:
        from daytona import AsyncDaytona
        from daytona._async.snapshot import SnapshotState
        daytona = AsyncDaytona()
        try:
            now = time.monotonic()
            n = 0
            page_n = 1
            while True:
                page = await daytona.snapshot.list(limit=200, page=page_n)
                items = list(getattr(page, "items", page) or [])
                for s in items:
                    name = getattr(s, "name", "") or ""
                    if name.startswith(("daytona-", "daytonaio/", "android-")):
                        continue
                    if getattr(s, "state", None) != SnapshotState.ACTIVE:
                        continue
                    if name in _build_gate._registry:
                        continue
                    st = BuildState(status="built")
                    st.event.set()
                    _build_gate._registry[name] = st
                    _build_gate._timestamps[name] = now
                    _last_used_at.setdefault(name, now)
                    n += 1
                if len(items) < 200:
                    break
                page_n += 1
            logger.info(
                "BuildGate reconciled %d existing ACTIVE snapshots from Daytona", n
            )
        finally:
            try: await daytona.close()
            except Exception: pass
    except Exception as e:
        logger.warning("BuildGate reconciliation skipped: %s", e)


# ── Request / Response models ───────────────────────────────────────────────


class StepRequest(BaseModel):
    """Thin request — env_service owns the TerminalEnvConfig.
    Caller only provides the task + model URL (from ProxySession)."""

    # Task to execute
    task: dict  # {"task_name", "instruction", ...}
    uid: str
    traj_i: int = 0

    # Model endpoint (from ProxySession env vars)
    model_url: str = ""      # OPENAI_BASE_URL from ProxySession
    model_api_key: str = ""  # OPENAI_API_KEY from ProxySession (= session_id)

    # Dataset info for path resolution
    dataset_name: str = ""
    task_name: str = ""

    # Trial name for organizing logs (e.g. "trial1-seta-env-v2-eval")
    trial_name: str = ""

    # R3 / indexer routing capture toggles. When set, env_service threads them
    # into model_config_dict so DeepSeekV4SGLangModel adds the corresponding
    # flags to its SGLang /generate payload and captures
    # meta_info.{routed_experts, indexer_topk} into tito_state.json.
    # generate_with_camel.py reads these from miles args (use_rollout_routing_replay
    # / use_rollout_indexer_replay) and passes them through.
    return_routed_experts: bool = False
    return_indexer_topk: bool = False


class StepResponse(BaseModel):
    run_info: dict | None = None
    reward: float | None = None
    error: str | None = None
    # miles Sample-shaped data ready for the custom-generate-fn adapter to copy
    # onto a Sample instance. None when the underlying model doesn't expose
    # `dump_tito_state` (e.g. OpenAI-compat path) or when the trajectory failed
    # before the agent ran. See scripts/miles/docs/sample_contract.md.
    sample: dict | None = None


class SetupRequest(BaseModel):
    dataset_name: str
    hf_token: str = ""


# ── Auth ────────────────────────────────────────────────────────────────────


def _check_auth(x_api_key: str) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")


# ── App lifecycle ───────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enlarge the asyncio DEFAULT executor. asyncio.to_thread (harbor's Daytona
    # create-semaphore acquire — which BLOCKS a thread while waiting for a permit —
    # plus SDK log-parsing etc.) runs on the default pool, whose default cap is
    # min(32, cpu+4)=32 — far too small once builds + Daytona ops run concurrently.
    # (P0's p.join now uses its own dedicated pool; see _step_subprocess.) These
    # threads are I/O-blocked, not CPU, so a large pool is cheap.
    try:
        _loop = asyncio.get_running_loop()
        _loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=int(os.environ.get("DEFAULT_EXECUTOR_WORKERS", "192")),
                thread_name_prefix="default-tt",
            )
        )
        logger.info("default executor enlarged to %s threads",
                    os.environ.get("DEFAULT_EXECUTOR_WORKERS", "192"))
    except Exception as _e:
        logger.warning("set_default_executor failed: %s", _e)
    gc_task = asyncio.create_task(_gc_loop())
    logger.info(
        "env_service started: max_slots=%d, dataset_root=%s", MAX_SLOTS, DATASET_ROOT
    )
    # GENTLE orphan drain. A restart orphans the prior lifetime's sandboxes (they
    # keep occupying the 250-CPU quota and the runner pool). Deleting them all at
    # once collapses the runner pool ("No available runners"); leaving them forces
    # the new ramp to wait ~15min for auto-stop. Instead: snapshot them NOW (before
    # serving, so we never target a sandbox this lifetime creates), then delete
    # them GRADUALLY in the background so CPU+runners free at ~the ramp's rate.
    # Deletes stay slower than creates -> net demand positive -> the pool holds;
    # the patient capacity-retry covers any momentary CPU overlap. Failure-free.
    _drain_client = None
    _drain_orphans: list = []
    try:
        from daytona import AsyncDaytona
        from harbor.environments.daytona import _HARBOR_OWNER_LABEL
        _drain_client = AsyncDaytona()
        _res = await asyncio.wait_for(_drain_client.list(), timeout=30)
        _items = (
            _res.items if hasattr(_res, "items") and not isinstance(_res, list)
            else _res
        ) or []
        _drain_orphans = [
            s for s in _items
            if _HARBOR_OWNER_LABEL in (getattr(s, "labels", None) or {})
        ]
        logger.info(
            "Gentle-drain: captured %d prior-lifetime sandbox(es) before serving",
            len(_drain_orphans),
        )
    except Exception as e:
        logger.warning("orphan capture for gentle-drain skipped: %s", e)
        _drain_client = None
    drain_task = asyncio.create_task(
        _drain_orphans_gradually(_drain_orphans, _drain_client)
    )
    # Respect builds from a prior lifetime: learn existing Daytona snapshots so a
    # restart reuses them (fast snapshot path) instead of cold-rebuilding every
    # task. Bounded + best-effort so a slow/unreachable Daytona never blocks start.
    try:
        await asyncio.wait_for(_reconcile_build_gate_from_daytona(), timeout=60)
    except Exception as e:
        logger.warning("startup snapshot reconciliation failed: %s", e)
    if SETUP_DATASET:
        for ds in [d.strip() for d in SETUP_DATASET.split(",") if d.strip()]:
            logger.info("Auto-setup dataset: %s", ds)
            try:
                result = await _download_dataset(ds)
                logger.info("Dataset %s: %s -> %s", ds, result["status"], result["path"])
            except Exception as e:
                logger.error("Failed to set up dataset %s: %s", ds, e)
    yield
    gc_task.cancel()
    drain_task.cancel()
    # Drain in-flight per-step worker threads at shutdown so we don't lose
    # trial state mid-execution.
    _STEP_EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Env Service", lifespan=lifespan)


# ── Endpoints ───────────────────────────────────────────────────────────────

# Leak/health observability (P1). The zombie-thread leak (a step that hangs in an
# uninterruptible Daytona/toolkit call leaves an un-cancellable thread behind even
# after STEP_TIMEOUT frees its slot) is invisible to active_steps/available_slots.
# Expose the process thread count + RSS + monotonic step counters so the external
# watchdog can restart this service BEFORE the accept loop starves.
_steps_completed_total = 0
_steps_failed_total = 0


def _proc_threads() -> int:
    try:
        return len(os.listdir("/proc/self/task"))
    except Exception:
        return -1


def _proc_rss_mb() -> int:
    try:
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    return int(_line.split()[1]) // 1024
    except Exception:
        pass
    return -1


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "max_slots": MAX_SLOTS,
        "available_slots": _slot_semaphore._value,
        "active_steps": _active_count,
        "build_gate": _build_gate.stats,
        # build/create-decoupling gauges: inflight = building + waiting-for-slot
        # + running (capped at inflight_cap); building_active = builders inside
        # build_fn; active_steps = live sandboxes (PHASE 2). A healthy steady
        # state holds active_steps≈MAX_SLOTS with inflight a little above it.
        "inflight": _inflight_count,
        "inflight_cap": MAX_SLOTS + INFLIGHT_LEAD,
        "building_active": _building_count,
        "build_concurrency": BUILD_CONCURRENCY,
        "dataset_root": str(DATASET_ROOT),
        # P1 leak signals for the watchdog
        "uvicorn_threads": _proc_threads(),
        "rss_mb": _proc_rss_mb(),
        "steps_completed_total": _steps_completed_total,
        "steps_failed_total": _steps_failed_total,
    }


@app.get("/config")
async def get_config(x_api_key: str = Header("")):
    """Return current TerminalEnvConfig."""
    _check_auth(x_api_key)
    return {"config": asdict(_te_config)}


@app.post("/config")
async def update_config(new_config: dict, x_api_key: str = Header("")):
    """Update TerminalEnvConfig without redeployment."""
    _check_auth(x_api_key)
    global _te_config

    te_data = new_config.get("terminal_env", new_config)
    _te_config = TerminalEnvConfig(
        agent=AgentConfig(**te_data["agent"]) if "agent" in te_data else _te_config.agent,
        model=None,
        runtime=RuntimeConfig(**te_data["runtime"]) if "runtime" in te_data else _te_config.runtime,
        env=EnvConfig(**te_data["env"]) if "env" in te_data else _te_config.env,
    )
    logger.info("Config updated via POST /config")
    return {"status": "ok", "config": asdict(_te_config)}


@app.post("/step")
async def step(req: StepRequest, x_api_key: str = Header("")):
    _check_auth(x_api_key)
    global _active_count

    # 1. Resolve task_dir
    task_name = req.task_name or req.task.get("task_name", "")
    dataset_name = req.dataset_name or req.task.get("dataset_name", "")
    if dataset_name and task_name:
        task_dir = str(DATASET_ROOT / dataset_name / task_name)
    else:
        task_dir = ""

    if not task_dir or not Path(task_dir).exists():
        return StepResponse(
            error=f"task_dir not found: {task_dir}. "
            f"Run POST /setup with dataset_name={dataset_name!r} first."
        )

    # 2. Build configs from service's TerminalEnvConfig
    agent_config = build_agent_config(_te_config.agent)
    env_config = build_env_config(_te_config.runtime, _te_config.env)

    # Model config: use raw YAML dict (preserves tito_enabled, tito_validate)
    # then override URL/api_key from request (Miles Router session URL)
    raw_model = getattr(_te_config, "_raw_model_config", {})
    if raw_model:
        model_config = dict(raw_model)
    elif _te_config.model is not None:
        model_config = build_model_config(_te_config.model)
    else:
        model_config = {
            "model_platform": "sglang",
            "model_type": "",
            "model_config_dict": {"max_tokens": _te_config.agent.max_total_tokens, "stream": False},
        }
    # Override URL and api_key from request (Miles Router session URL)
    if req.model_url:
        model_config["url"] = req.model_url
    if req.model_api_key:
        model_config["api_key"] = req.model_api_key

    # Thread routing-capture toggles into model_config_dict — they're popped
    # by DeepSeekV4SGLangModel.__init__ before being merged into sampling_params.
    if req.return_routed_experts or req.return_indexer_topk:
        mcd = dict(model_config.get("model_config_dict") or {})
        if req.return_routed_experts:
            mcd["return_routed_experts"] = True
        if req.return_indexer_topk:
            mcd["return_indexer_topk"] = True
        model_config["model_config_dict"] = mcd

    # Trial root: organized by trial_name
    trial_root = HARBOR_ROOT / "trials"
    if req.trial_name:
        trial_root = trial_root / req.trial_name

    runtime_config = {
        "task_dir": task_dir,
        "trial_root": str(trial_root),
        "environment_type": _te_config.runtime.env_type,
    }
    # Forward resource overrides if set in the env_service yaml. These flow
    # through DockerHarborRuntime → harbor.environments.factory → DaytonaEnvironment
    # as `override_cpus / override_memory_mb / override_storage_mb` and are
    # applied per-sandbox-create (running sandboxes are unaffected).
    if _te_config.runtime.override_cpus is not None:
        runtime_config["override_cpus"] = _te_config.runtime.override_cpus
    if _te_config.runtime.override_memory_mb is not None:
        runtime_config["override_memory_mb"] = _te_config.runtime.override_memory_mb
    if _te_config.runtime.override_storage_mb is not None:
        runtime_config["override_storage_mb"] = _te_config.runtime.override_storage_mb

    # 3. Build gate (single-flight per task_name): the first trajectory of
    # each task does the image build via harbor.build() (e.g. docker compose
    # build for docker env_type); the other 7+ waiters block at the event
    # until the build is done, then proceed to their own start()/up -d which
    # uses the locally-cached image.
    build_root = trial_root / "_builds"
    build_env_type = _te_config.runtime.env_type
    async def build_fn():
        # Evict LRU snapshots BEFORE registering a new one (daytona env_type
        # only; the helper is a no-op when not over quota). Runs once per
        # task per env_service lifetime — subsequent /step calls for an
        # already-built task skip build_fn via BuildGate.
        # Declarative mode registers NO named snapshots, so there is no quota to
        # manage — skip eviction entirely (the snapshot-limit concern is moot).
        if build_env_type == "daytona" and os.environ.get("DAYTONA_DECLARATIVE", "0") != "1":
            await _ensure_snapshot_quota()
        rt = DockerHarborRuntime(
            task_dir=task_dir,
            trial_root=str(build_root),
            session_id=f"build_{task_name}",
            environment_type=build_env_type,
            **{k: runtime_config[k] for k in
               ("override_cpus", "override_memory_mb", "override_storage_mb")
               if k in runtime_config},
        )
        try:
            await rt.build()
        except SnapshotPlacementDeferred as _pd:
            # Case B: snapshot built fine (ACTIVE) but never became schedulable
            # within the canary timeout. The snapshot is VALID — just cold — so
            # we KEEP it (do NOT delete): the next encounter reuses it warm. The
            # group still skips this round (BuildGate records 'failed' so all
            # same-task waiters raise together — coherent group skip, no storm).
            logger.warning(
                "PLACEMENT DEFERRED for %s: %s — keeping snapshot to warm; "
                "group skips this round", task_name, _pd,
            )
            raise
        except BaseException as _be:
            # Case C (and any other build failure): the snapshot is broken ->
            # delete any partial/error snapshot so it neither lingers on quota nor
            # gets reused half-built. BuildGate records 'failed', so every
            # same-task waiter raises and all trajectories fail together.
            if build_env_type == "daytona":
                try:
                    from harbor.environments.daytona import DaytonaEnvironment
                    await DaytonaEnvironment.cleanup_snapshot(task_name)
                    logger.error(
                        "BUILD FAILED for %s: %s — deleted partial snapshot; "
                        "invalidating whole group", task_name, _be,
                    )
                except Exception as _ce:
                    logger.warning(
                        "post-build-fail snapshot cleanup failed for %s: %s",
                        task_name, _ce,
                    )
            raise
        finally:
            try:
                await rt.stop()
            except Exception:
                pass

    # ── BUILD/CREATE DECOUPLING ──────────────────────────────────────────────
    # _inflight_semaphore (MAX_SLOTS + INFLIGHT_LEAD) is the outermost gate:
    # backpressure so building never runs more than INFLIGHT_LEAD ahead of the
    # live slots. Inside it, PHASE 1 (build gate) runs WITHOUT a sandbox slot;
    # only PHASE 2 (create + run + eval) takes a _slot_semaphore permit, so the
    # 160 slots map 1:1 to live sandboxes instead of being squatted by builds.
    async with _inflight_semaphore:
        _bump_inflight(1)
        try:
            # ── PHASE 1: build gate — NO sandbox slot held ───────────────────
            # Single-flight per task (BuildGate); the builder is bounded by
            # _build_semaphore. A build failure returns here WITHOUT ever taking a
            # slot, so failed-build groups never burn sandbox capacity.
            try:
                await _build_gate.ensure_built(task_name, build_fn)
            except RuntimeError as e:
                return StepResponse(error=str(e))

            # ── PHASE 2: sandbox slot ONLY for create + run + eval ───────────
            async with _slot_semaphore:
                # Stagger entry into the create/eval phases so a freed-all-at-once
                # cohort (restart or batch boundary) cannot stampede Daytona's
                # create gate. Cheap at steady state; the cure for the herd.
                await _pace_admission()
                _active_count += 1
                # Per-task LRU tracking for snapshot eviction. Refresh last-used on
                # every /step; reference-count so eviction never deletes a snapshot
                # actively in use by a trajectory.
                _last_used_at[task_name] = time.monotonic()
                _active_per_task[task_name] = _active_per_task.get(task_name, 0) + 1
                try:
                    task = {**req.task, "task_path": task_dir}
                    te = TerminalEnvironment(
                        agent_config=agent_config,
                        model_config=model_config,
                        runtime_config=runtime_config,
                        env_config=env_config,
                    )
                    # ── PER-STEP THREAD+LOOP ISOLATION ─────────────────────
                    # Run te.step() in a dedicated worker thread with its own
                    # asyncio event loop (via asyncio.run). All async resources
                    # constructed inside te.step (DeepSeekV4SGLangModel httpx
                    # client, harbor AsyncDaytona, terminal-toolkit runtime) are
                    # bound to this per-step loop. A blocked tool / parse / SDK
                    # call in this step cannot freeze env_service's main loop or
                    # any other in-flight step.
                    #
                    # NOTE: te is constructed on the main loop but its __init__
                    # only stores dict configs (no async resources). The async
                    # resource construction happens inside te.step → _reset_runtime
                    # → _reset_agent, all on the per-step loop. Verified.
                    loop = asyncio.get_event_loop()
                    _uid = req.uid
                    _traj_i = req.traj_i
                    # P0 (STEP_USE_SUBPROCESS): run the step in a KILLABLE forkserver
                    # subprocess so a step that hangs in an uninterruptible Daytona/
                    # toolkit call is SIGKILLed at STEP_TIMEOUT — no un-cancellable
                    # thread is leaked (the zombie-thread leak that eventually starves
                    # the accept loop). Raises asyncio.TimeoutError on timeout (same as
                    # wait_for) so the except-blocks below are unchanged. Default (flag
                    # off) = the original in-thread isolated loop.
                    if _USE_STEP_SUBPROCESS:
                        from seta_env.services._step_subprocess import run_step_killable
                        _result_path = os.path.join("/tmp/env_step_results", f"{_uid}.pkl")
                        run_info, reward = await run_step_killable(
                            te, task, _uid, _traj_i, STEP_TIMEOUT_SECONDS, _result_path,
                        )
                    else:
                        def _run_step_in_isolated_loop():
                            return asyncio.run(te.step(task, uid=_uid, traj_i=_traj_i))
                        run_info, reward = await asyncio.wait_for(
                            loop.run_in_executor(_STEP_EXECUTOR, _run_step_in_isolated_loop),
                            timeout=STEP_TIMEOUT_SECONDS,
                        )
                    # If the model dumped a TITO state, surface it in the response so
                    # the miles custom-generate-fn adapter can build a Sample without
                    # going back to disk. Path matches what the runtime writes to
                    # (HARBOR_ROOT/trials/<trial_name>/<uid>/tito_state.json).
                    sample_payload: dict | None = None
                    try:
                        tito_path = trial_root / req.uid / "tito_state.json"
                        if tito_path.exists():
                            sample_payload = json.loads(tito_path.read_text())
                    except Exception as _e:
                        logger.warning("could not load tito_state for %s: %s", req.uid, _e)
                    return StepResponse(
                        run_info=run_info, reward=reward, sample=sample_payload
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    logger.error(
                        "step() TIMED OUT for %s after %.0fs (slot released; worker "
                        "thread left to unwind)", req.uid, STEP_TIMEOUT_SECONDS
                    )
                    return StepResponse(error=f"step timeout after {STEP_TIMEOUT_SECONDS:.0f}s")
                except Exception as e:
                    logger.error("step() failed for %s: %s", req.uid, e, exc_info=True)
                    return StepResponse(error=str(e))
                finally:
                    _active_count -= 1
                    _active_per_task[task_name] = max(0, _active_per_task.get(task_name, 0) - 1)
                    _last_used_at[task_name] = time.monotonic()
        finally:
            _bump_inflight(-1)


async def _download_dataset(dataset_name: str, hf_token: str = "") -> dict:
    """Download a dataset by name into DATASET_ROOT. Returns a status dict."""
    import shutil
    import tempfile

    dest = DATASET_ROOT / dataset_name
    if dest.exists() and any(dest.iterdir()):
        return {"status": "already_present", "path": str(dest), "success": True}

    datasets_yaml = Path(__file__).parent.parent / "dataset" / "datasets.yaml"
    if not datasets_yaml.exists():
        raise ValueError(f"datasets.yaml not found at {datasets_yaml}")

    datasets_cfg = yaml.safe_load(datasets_yaml.read_text()).get("datasets", {})
    if dataset_name not in datasets_cfg:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Available: {list(datasets_cfg.keys())}"
        )

    cfg = datasets_cfg[dataset_name]
    repo = cfg.get("repo")
    if not repo:
        raise ValueError(f"No repo URL for dataset {dataset_name!r}")
    subfolder = cfg.get("subfolder")

    if dataset_name not in _dataset_locks:
        _dataset_locks[dataset_name] = asyncio.Lock()

    async with _dataset_locks[dataset_name]:
        if dest.exists() and any(dest.iterdir()):
            return {"status": "already_present", "path": str(dest), "success": True}

        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        clone_url = repo
        clone_env = {**os.environ}
        hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        if hf_token and "huggingface.co" in repo:
            clone_url = repo.replace("https://", f"https://user:{hf_token}@")

        with tempfile.TemporaryDirectory() as tmpdir:
            clone_dest = f"{tmpdir}/repo"
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth=1", clone_url, clone_dest,
                env=clone_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"git clone failed: {stderr.decode()}")

            proc2 = await asyncio.create_subprocess_exec(
                "git", "lfs", "pull", cwd=clone_dest, env=clone_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, lfs_err = await proc2.communicate()
            if proc2.returncode != 0:
                logger.warning("git lfs pull failed (non-fatal): %s", lfs_err.decode()[:200])

            if subfolder:
                shutil.move(f"{clone_dest}/{subfolder}", str(dest))
            else:
                shutil.move(clone_dest, str(dest))

    return {"status": "downloaded", "path": str(dest), "success": True}


@app.post("/setup")
async def setup_dataset(req: SetupRequest, x_api_key: str = Header("")):
    """Download/activate dataset. Same pattern as node_manager."""
    _check_auth(x_api_key)
    try:
        return await _download_dataset(req.dataset_name, req.hf_token)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


class CleanupTaskRequest(BaseModel):
    task_name: str


@app.post("/cleanup_task")
async def cleanup_task(req: CleanupTaskRequest, x_api_key: str = Header("")):
    """Delete the named Daytona snapshot for ``task_name`` and forget it from
    the BuildGate registry.

    Required when training over a dataset with more unique tasks than the
    Daytona snapshot quota (Tier 3 = 100). Orchestrators should call this
    after a task's batch of trajectories completes, so the snapshot slot can
    be reused for the next task in the next batch. Without this, builds in
    later batches will fail with "Snapshot quota exceeded".

    Best-effort: never errors if snapshot doesn't exist or delete fails.
    """
    _check_auth(x_api_key)
    from harbor.environments.daytona import DaytonaEnvironment
    try:
        await DaytonaEnvironment.cleanup_snapshot(req.task_name)
    except Exception as e:
        logger.warning("cleanup_snapshot(%s) failed: %s", req.task_name, e)
    # Forget from BuildGate registry so re-encountering this task triggers
    # a fresh build (the snapshot is gone; the cached state would be stale).
    _build_gate._registry.pop(req.task_name, None)
    _build_gate._timestamps.pop(req.task_name, None)
    return {"task_name": req.task_name, "deleted": True}


@app.post("/cleanup")
async def cleanup(x_api_key: str = Header("")):
    """Full Docker cleanup: stop all, remove all, prune networks."""
    _check_auth(x_api_key)

    async def _docker(*args):
        p = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await p.communicate()
        return [c for c in out.decode().strip().split("\n") if c]

    # 1. Stop all running containers
    running = await _docker("ps", "-q")
    if running:
        await (await asyncio.create_subprocess_exec(
            "docker", "stop", *running,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )).communicate()

    # 2. Remove all containers
    all_containers = await _docker("ps", "-aq")
    if all_containers:
        await (await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", *all_containers,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )).communicate()

    # 3. Prune unused networks
    await (await asyncio.create_subprocess_exec(
        "docker", "network", "prune", "-f",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )).communicate()

    # 4. Clear build gate
    _build_gate.clear(older_than=0)

    return {
        "status": "ok",
        "containers_stopped": len(running),
        "containers_removed": len(all_containers),
        "networks_pruned": True,
    }


# ── GC loop ─────────────────────────────────────────────────────────────────


async def _gc_loop():
    while True:
        await asyncio.sleep(GC_INTERVAL_SEC)
        try:
            cleared = _build_gate.clear()
            if cleared:
                logger.info("GC: cleared %d expired build entries", cleared)
        except Exception as e:
            logger.warning("GC error: %s", e)
