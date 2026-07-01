"""TaskManager — FastAPI service for curriculum-learning task sampling.

Maintains per-task evaluation history and dynamically weights task sampling.
Multiple DP-head ranks pull tasks via HTTP.

Usage:
    python -m seta_env.datahubs.task_manager_service \
        --dataset-root /path/to/harbor/dataset --port 8765
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ResultItem(BaseModel):
    uid: str
    task_id: str
    score: float | None
    group_id: str


class TaskItem(BaseModel):
    task_id: str
    task_path: str
    instruction: str
    uid: str


class InitializeRequest(BaseModel):
    scores_csv: str


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryRecord:
    uid: str
    score: float | None
    group_id: str


@dataclass
class TaskStats:
    task_id: str
    recent_pass_at_n: float = 0.0
    recent_mean_score: float = 0.0
    recent_variance: float = 0.0
    n_groups: int = 0
    n_trajectories: int = 0
    n_errors: int = 0
    category: str = ""


@dataclass
class CategoryPool:
    name: str
    weight: float
    members: set[str] = field(default_factory=set)
    queue: deque[str] = field(default_factory=deque)

    def is_empty(self) -> bool:
        return len(self.members) == 0

    def build_queue(self) -> None:
        self.queue = deque(random.sample(list(self.members), len(self.members)))

    def add_task(self, task_id: str) -> None:
        self.members.add(task_id)
        self.queue.append(task_id)

    def remove_task(self, task_id: str) -> None:
        self.members.discard(task_id)
        try:
            self.queue.remove(task_id)
        except ValueError:
            pass


@dataclass
class TransitionEvent:
    timestamp: float
    task_id: str
    old_category: str
    new_category: str
    trigger: str  # "push" or "initialize"


@dataclass
class ServiceMetrics:
    total_pulls: int = 0
    total_pushes: int = 0
    total_trajectories: int = 0
    total_errors: int = 0
    service_start_time: float = 0.0
    last_pull_time: float = 0.0
    last_push_time: float = 0.0
    last_push_per_task: dict[str, float] = field(default_factory=dict)
    transitions: deque[TransitionEvent] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    category_snapshots: deque[tuple[float, dict[str, int]]] = field(
        default_factory=lambda: deque(maxlen=10000)
    )


# ---------------------------------------------------------------------------
# Sort helper — numeric-first then lexicographic
# ---------------------------------------------------------------------------


def _sort_key(name: str) -> tuple[int, int | str]:
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks_from_disk(dataset_root: str) -> list[dict]:
    """Load tasks from a harbor dataset directory.

    Each subdirectory with an ``instruction.md`` is treated as a task.
    Returns a list of dicts with ``task_id``, ``task_path``, ``instruction``,
    sorted in numeric-first then lexicographic order.
    """
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    tasks: list[dict] = []
    for task_dir in root.iterdir():
        if not task_dir.is_dir():
            continue
        instruction_file = task_dir / "instruction.md"
        if not instruction_file.exists():
            continue
        tasks.append(
            {
                "task_id": task_dir.name,
                "task_path": str(task_dir),
                "instruction": instruction_file.read_text().strip(),
            }
        )

    tasks.sort(key=lambda t: _sort_key(t["task_id"]))
    return tasks


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------


class CurriculumSampler:
    def __init__(self) -> None:
        self.pools: dict[str, CategoryPool] = {}
        self.task_category: dict[str, str] = {}

    def init_pools(self, w_zpd: float, w_uncertain: float, w_too_hard: float, w_mastered: float) -> None:
        self.pools = {
            "zpd": CategoryPool(name="zpd", weight=w_zpd),
            "uncertain": CategoryPool(name="uncertain", weight=w_uncertain),
            "too_hard": CategoryPool(name="too_hard", weight=w_too_hard),
            "mastered": CategoryPool(name="mastered", weight=w_mastered),
        }
        self.task_category = {}

    def assign_task(self, task_id: str, category: str) -> str | None:
        """Assign *task_id* to *category*. Returns the old category or None."""
        old = self.task_category.get(task_id)
        if old == category:
            return old
        if old and old in self.pools:
            self.pools[old].remove_task(task_id)
        if category in self.pools:
            self.pools[category].add_task(task_id)
        self.task_category[task_id] = category
        return old

    def remove_task(self, task_id: str) -> str | None:
        old = self.task_category.pop(task_id, None)
        if old and old in self.pools:
            self.pools[old].remove_task(task_id)
        return old

    def build_all_queues(self) -> None:
        for pool in self.pools.values():
            if not pool.is_empty():
                pool.build_queue()

    def pull(self) -> str | None:
        non_empty = [p for p in self.pools.values() if not p.is_empty()]
        if not non_empty:
            return None
        total_w = sum(p.weight for p in non_empty)
        weights = [p.weight / total_w for p in non_empty]
        chosen = random.choices(non_empty, weights=weights, k=1)[0]
        task_id = chosen.queue.popleft()
        if len(chosen.queue) == 0 and not chosen.is_empty():
            chosen.build_queue()
        return task_id


# ---------------------------------------------------------------------------
# TaskManagerApp — holds all mutable state
# ---------------------------------------------------------------------------


class TaskManagerApp:
    def __init__(self, dataset_root: str, strategy: str, **kwargs: float | int) -> None:
        self.dataset_root = dataset_root
        self.strategy = strategy
        self.pass_n_hi: float = kwargs.get("pass_n_hi", 1.0)
        self.var_thresh: float = kwargs.get("var_thresh", 0.05)
        self.window_size: int = int(kwargs.get("window_size", 4))
        self.w_zpd: float = kwargs.get("w_zpd", 4.0)
        self.w_uncertain: float = kwargs.get("w_uncertain", 4.0)
        self.w_too_hard: float = kwargs.get("w_too_hard", 1.0)
        self.w_mastered: float = kwargs.get("w_mastered", 0.2)

        # State — populated by cleanup()
        self.tasks: list[dict] = []
        self.task_map: dict[str, dict] = {}
        self._pull_counter: int = 0

        # Per-task rolling windows: task_id -> deque of groups (each group = list[TrajectoryRecord])
        self._task_windows: dict[str, deque[list[TrajectoryRecord]]] = {}
        self._task_stats: dict[str, TaskStats] = {}
        self._all_none_counts: dict[str, int] = {}  # consecutive all-None group counts

        # Sampling state
        self._sequential_queue: deque[str] = deque()
        self._cold_start_queue: deque[str] = deque()
        self._phase2: bool = False
        self._sampler: CurriculumSampler = CurriculumSampler()
        self._broken: set[str] = set()

        # Monitoring
        self.metrics = ServiceMetrics(service_start_time=time.time())

    # ---- cleanup ----

    def cleanup(self) -> dict:
        self.tasks = load_tasks_from_disk(self.dataset_root)
        self.task_map = {t["task_id"]: t for t in self.tasks}
        self._pull_counter = 0
        self._task_windows = {t["task_id"]: deque(maxlen=self.window_size) for t in self.tasks}
        self._task_stats = {}
        self._all_none_counts = {}
        self._broken = set()
        self._phase2 = False

        task_ids = [t["task_id"] for t in self.tasks]

        if self.strategy == "sequential":
            self._sequential_queue = deque(task_ids)
        else:
            self._cold_start_queue = deque(task_ids)
            self._sequential_queue = deque()
            self._sampler = CurriculumSampler()
            self._sampler.init_pools(self.w_zpd, self.w_uncertain, self.w_too_hard, self.w_mastered)

        # Reset metrics (keep service_start_time)
        self.metrics = ServiceMetrics(service_start_time=self.metrics.service_start_time)

        return {"status": "ok", "n_tasks": len(self.tasks)}

    # ---- initialize (warm start) ----

    def initialize(self, scores_csv: str) -> dict:
        if self.strategy != "weighted":
            return None  # signal 400

        df = pd.read_csv(scores_csv, header=None)
        seen_task_ids: set[str] = set()
        n_broken = 0

        for _, row in df.iterrows():
            task_id = str(row.iloc[0])
            seen_task_ids.add(task_id)

            if task_id not in self.task_map:
                continue  # skip tasks not in dataset

            scores: list[float | None] = []
            for val in row.iloc[1:]:
                if pd.isna(val):
                    scores.append(None)
                else:
                    scores.append(float(val))

            group_id = f"init_{task_id}"
            records = [
                TrajectoryRecord(uid=f"{task_id}_init_{i}", score=s, group_id=group_id)
                for i, s in enumerate(scores)
            ]
            self._task_windows[task_id].append(records)

            # Check all-None → broken
            if all(r.score is None for r in records):
                self._broken.add(task_id)
                n_broken += 1
            else:
                self._compute_stats(task_id)

        # Tasks in dataset but NOT in CSV → broken
        for task_id in self.task_map:
            if task_id not in seen_task_ids:
                self._broken.add(task_id)
                n_broken += 1

        # Classify all non-broken tasks with stats
        for task_id in self.task_map:
            if task_id in self._broken:
                if task_id in self._task_stats:
                    self._task_stats[task_id].category = "broken"
                else:
                    self._task_stats[task_id] = TaskStats(task_id=task_id, category="broken")
                continue
            if task_id in self._task_stats:
                cat = self._classify(task_id)
                self._sampler.assign_task(task_id, cat)
                self._task_stats[task_id].category = cat

        self._sampler.build_all_queues()
        self._phase2 = True
        self._cold_start_queue = deque()

        # Record transitions for monitoring
        now = time.time()
        for task_id, cat in self._sampler.task_category.items():
            self.metrics.transitions.append(
                TransitionEvent(timestamp=now, task_id=task_id, old_category="", new_category=cat, trigger="initialize")
            )

        cats = {
            "zpd": len(self._sampler.pools["zpd"].members),
            "uncertain": len(self._sampler.pools["uncertain"].members),
            "mastered": len(self._sampler.pools["mastered"].members),
            "too_hard": len(self._sampler.pools["too_hard"].members),
            "broken": len(self._broken),
        }

        # Snapshot
        self.metrics.category_snapshots.append((now, dict(cats)))

        return {
            "status": "ok",
            "n_initialized": len(seen_task_ids & set(self.task_map.keys())),
            "n_broken": n_broken,
            "categories": cats,
        }

    # ---- pull_task ----

    def pull_task(self) -> dict | None:
        self._pull_counter += 1
        uid_counter = self._pull_counter

        # Metrics
        self.metrics.total_pulls += 1
        self.metrics.last_pull_time = time.time()

        if self.strategy == "sequential":
            if not self._sequential_queue:
                self._sequential_queue = deque(t["task_id"] for t in self.tasks)
            task_id = self._sequential_queue.popleft()
        elif not self._phase2:
            # Weighted, Phase 1 (cold start)
            if not self._cold_start_queue:
                # Queue was already empty (transition happened on previous pull)
                return self._pull_phase2(uid_counter)
            task_id = self._cold_start_queue.popleft()
            # If that was the last one, transition to Phase 2 immediately
            if not self._cold_start_queue:
                self._transition_to_phase2()
        else:
            return self._pull_phase2(uid_counter)

        task = self.task_map[task_id]
        uid = f"{task_id}_{uid_counter:05d}"
        return {
            "task_id": task_id,
            "task_path": task["task_path"],
            "instruction": task["instruction"],
            "uid": uid,
        }

    def _pull_phase2(self, uid_counter: int) -> dict | None:
        task_id = self._sampler.pull()
        if task_id is None:
            return None
        task = self.task_map[task_id]
        uid = f"{task_id}_{uid_counter:05d}"
        return {
            "task_id": task_id,
            "task_path": task["task_path"],
            "instruction": task["instruction"],
            "uid": uid,
        }

    def _transition_to_phase2(self) -> None:
        """Cold start queue drained — classify all tasks with results and enter Phase 2."""
        for task_id in self.task_map:
            if task_id in self._broken:
                continue
            if task_id in self._task_stats:
                cat = self._classify(task_id)
                self._sampler.assign_task(task_id, cat)
                self._task_stats[task_id].category = cat
        self._sampler.build_all_queues()
        self._phase2 = True

        # Snapshot
        now = time.time()
        cats = self._category_counts()
        self.metrics.category_snapshots.append((now, cats))

    # ---- push_results ----

    def push_results(self, results: list[ResultItem]) -> dict:
        if not results:
            return {"status": "ok", "n_accepted": 0}

        now = time.time()
        self.metrics.total_pushes += 1
        self.metrics.last_push_time = now

        task_id = results[0].task_id
        group_id = results[0].group_id

        self.metrics.last_push_per_task[task_id] = now

        # Dedup: if a group with this group_id already exists in the window, skip entirely
        existing_group_ids: set[str] = set()
        if task_id in self._task_windows:
            for group in self._task_windows[task_id]:
                if group:
                    existing_group_ids.add(group[0].group_id)

        if group_id in existing_group_ids:
            return {"status": "ok", "n_accepted": 0}

        records: list[TrajectoryRecord] = []
        for r in results:
            records.append(TrajectoryRecord(uid=r.uid, score=r.score, group_id=group_id))

        # Metrics
        self.metrics.total_trajectories += len(records)
        self.metrics.total_errors += sum(1 for r in records if r.score is None)

        if not records:
            return {"status": "ok", "n_accepted": 0}

        n_accepted = len(records)

        # Append group to window
        if task_id not in self._task_windows:
            self._task_windows[task_id] = deque(maxlen=self.window_size)
        self._task_windows[task_id].append(records)

        # Check broken: all scores None in this group
        if all(r.score is None for r in records):
            self._all_none_counts[task_id] = self._all_none_counts.get(task_id, 0) + 1
            if self._all_none_counts[task_id] >= 2 and task_id not in self._broken:
                self._mark_broken(task_id, trigger="push")
        else:
            # Reset consecutive all-None counter on a non-all-None group
            self._all_none_counts[task_id] = 0

        # Compute stats and reclassify (if weighted and not broken)
        if task_id not in self._broken:
            self._compute_stats(task_id)

            if self.strategy == "weighted" and self._phase2:
                old_cat = self._sampler.task_category.get(task_id)
                new_cat = self._classify(task_id)
                if old_cat != new_cat:
                    self._sampler.assign_task(task_id, new_cat)
                    self._task_stats[task_id].category = new_cat
                    self.metrics.transitions.append(
                        TransitionEvent(
                            timestamp=now,
                            task_id=task_id,
                            old_category=old_cat or "",
                            new_category=new_cat,
                            trigger="push",
                        )
                    )
                # Handle tasks arriving after phase2 transition (were in-flight during cold start)
                elif old_cat is None:
                    self._sampler.assign_task(task_id, new_cat)
                    self._task_stats[task_id].category = new_cat

            elif self.strategy == "weighted" and not self._phase2:
                # Still in cold start — just classify for stats
                cat = self._classify(task_id)
                self._task_stats[task_id].category = cat

        # Category snapshot
        if self.strategy == "weighted":
            cats = self._category_counts()
            self.metrics.category_snapshots.append((now, cats))

        return {"status": "ok", "n_accepted": n_accepted}

    def _mark_broken(self, task_id: str, trigger: str = "push") -> None:
        old_cat = self._sampler.task_category.get(task_id, "")
        self._broken.add(task_id)
        self._sampler.remove_task(task_id)
        if task_id in self._task_stats:
            self._task_stats[task_id].category = "broken"
        else:
            self._task_stats[task_id] = TaskStats(task_id=task_id, category="broken")

        # Remove from cold start queue if still present
        try:
            self._cold_start_queue.remove(task_id)
        except ValueError:
            pass

        self.metrics.transitions.append(
            TransitionEvent(
                timestamp=time.time(),
                task_id=task_id,
                old_category=old_cat,
                new_category="broken",
                trigger=trigger,
            )
        )

    # ---- stats computation ----

    def _compute_stats(self, task_id: str) -> None:
        window = self._task_windows.get(task_id)
        if not window:
            return

        all_scores: list[float] = []
        group_pass_at_ns: list[float] = []
        group_variances: list[float] = []
        total_trajs = 0
        total_errors = 0

        for group in window:
            non_none = [r.score for r in group if r.score is not None]
            none_count = sum(1 for r in group if r.score is None)
            total_trajs += len(group)
            total_errors += none_count

            if non_none:
                all_scores.extend(non_none)
                pass_count = sum(1 for s in non_none if s == 1.0)
                group_pass_at_ns.append(pass_count / len(non_none))
                if len(non_none) >= 2:
                    group_variances.append(statistics.variance(non_none))
                else:
                    group_variances.append(0.0)
            else:
                # All None group — don't contribute to pass@n or variance
                group_pass_at_ns.append(0.0)
                group_variances.append(0.0)

        stats = TaskStats(
            task_id=task_id,
            recent_pass_at_n=statistics.mean(group_pass_at_ns) if group_pass_at_ns else 0.0,
            recent_mean_score=statistics.mean(all_scores) if all_scores else 0.0,
            recent_variance=statistics.mean(group_variances) if group_variances else 0.0,
            n_groups=len(window),
            n_trajectories=total_trajs,
            n_errors=total_errors,
            category=self._task_stats.get(task_id, TaskStats(task_id=task_id)).category,
        )
        self._task_stats[task_id] = stats

    def _classify(self, task_id: str) -> str:
        stats = self._task_stats.get(task_id)
        if not stats:
            return "too_hard"

        if stats.recent_pass_at_n >= self.pass_n_hi:
            return "mastered"
        elif stats.recent_pass_at_n > 0:
            return "zpd"
        else:
            # pass@n == 0
            if stats.recent_variance >= self.var_thresh:
                return "uncertain"
            else:
                return "too_hard"

    def _category_counts(self) -> dict[str, int]:
        return {
            "zpd": len(self._sampler.pools.get("zpd", CategoryPool(name="zpd", weight=0)).members),
            "uncertain": len(self._sampler.pools.get("uncertain", CategoryPool(name="uncertain", weight=0)).members),
            "mastered": len(self._sampler.pools.get("mastered", CategoryPool(name="mastered", weight=0)).members),
            "too_hard": len(self._sampler.pools.get("too_hard", CategoryPool(name="too_hard", weight=0)).members),
            "broken": len(self._broken),
        }

    # ---- stats endpoint ----

    def get_stats(self) -> dict:
        now = time.time()
        uptime = now - self.metrics.service_start_time if self.metrics.service_start_time else 0
        uptime_minutes = uptime / 60.0 if uptime > 0 else 1.0

        if self.strategy == "weighted":
            if self._phase2:
                phase = "weighted_phase2"
            else:
                phase = "weighted_phase1"
        else:
            phase = "sequential"

        # Build categories dict
        categories: dict[str, dict] = {}
        if self.strategy == "weighted":
            for cat_name in ["zpd", "uncertain", "mastered", "too_hard"]:
                pool = self._sampler.pools.get(cat_name)
                if pool:
                    categories[cat_name] = {"count": len(pool.members), "task_ids": sorted(pool.members, key=_sort_key)}
                else:
                    categories[cat_name] = {"count": 0, "task_ids": []}
            categories["broken"] = {"count": len(self._broken), "task_ids": sorted(self._broken, key=_sort_key)}
        else:
            categories = {}

        # Build tasks dict
        tasks_dict: dict[str, dict] = {}
        for tid, st in self._task_stats.items():
            tasks_dict[tid] = {
                "recent_pass_at_n": st.recent_pass_at_n,
                "recent_mean_score": st.recent_mean_score,
                "recent_variance": st.recent_variance,
                "n_groups": st.n_groups,
                "n_trajectories": st.n_trajectories,
                "n_errors": st.n_errors,
                "category": st.category,
                "last_push_time": self.metrics.last_push_per_task.get(tid, 0.0),
            }

        # Recent transitions
        recent_transitions = [
            {
                "timestamp": t.timestamp,
                "task_id": t.task_id,
                "old": t.old_category,
                "new": t.new_category,
                "trigger": t.trigger,
            }
            for t in self.metrics.transitions
        ]

        # Category history
        category_history = list(self.metrics.category_snapshots)

        return {
            "n_tasks": len(self.tasks),
            "phase": phase,
            "metrics": {
                "total_pulls": self.metrics.total_pulls,
                "total_pushes": self.metrics.total_pushes,
                "total_trajectories": self.metrics.total_trajectories,
                "total_errors": self.metrics.total_errors,
                "error_rate": (
                    self.metrics.total_errors / self.metrics.total_trajectories
                    if self.metrics.total_trajectories > 0
                    else 0.0
                ),
                "uptime_seconds": uptime,
                "pulls_per_minute": self.metrics.total_pulls / uptime_minutes,
                "pushes_per_minute": self.metrics.total_pushes / uptime_minutes,
            },
            "categories": categories,
            "recent_transitions": recent_transitions,
            "category_history": category_history,
            "tasks": tasks_dict,
        }


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def create_app(dataset_root: str, strategy: str = "weighted", **kwargs: float | int) -> FastAPI:
    app = FastAPI(title="TaskManager")
    state = TaskManagerApp(dataset_root, strategy, **kwargs)

    @app.post("/cleanup")
    async def cleanup():
        return state.cleanup()

    @app.post("/initialize")
    async def initialize(req: InitializeRequest):
        result = state.initialize(req.scores_csv)
        if result is None:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=400, content={"detail": "initialize is only valid in weighted mode"})
        return result

    @app.get("/pull_task")
    async def pull_task():
        result = state.pull_task()
        if result is None:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content={"detail": "no tasks available"})
        return result

    @app.post("/push_results")
    async def push_results(results: list[ResultItem]):
        return state.push_results(results)

    @app.get("/stats")
    async def stats():
        return state.get_stats()

    @app.get("/dashboard")
    async def dashboard():
        from fastapi.responses import HTMLResponse

        html_path = Path(__file__).parent / "dashboard.html"
        return HTMLResponse(content=html_path.read_text(), status_code=200)

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TaskManager curriculum-learning service")
    p.add_argument("--dataset-root", required=True, help="Path to Harbor dataset directory")
    p.add_argument("--strategy", default="weighted", choices=["weighted", "sequential"])
    p.add_argument("--pass-n-hi", type=float, default=1.0)
    p.add_argument("--var-thresh", type=float, default=0.05)
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--w-zpd", type=float, default=4.0)
    p.add_argument("--w-uncertain", type=float, default=4.0)
    p.add_argument("--w-too-hard", type=float, default=1.0)
    p.add_argument("--w-mastered", type=float, default=0.2)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        dataset_root=args.dataset_root,
        strategy=args.strategy,
        pass_n_hi=args.pass_n_hi,
        var_thresh=args.var_thresh,
        window_size=args.window_size,
        w_zpd=args.w_zpd,
        w_uncertain=args.w_uncertain,
        w_too_hard=args.w_too_hard,
        w_mastered=args.w_mastered,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
