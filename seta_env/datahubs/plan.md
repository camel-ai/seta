# TaskManager — Design & Implementation Spec

## Overview

A FastAPI service that maintains per-task evaluation history and dynamically weights task sampling for curriculum learning. Multiple DP-head ranks pull tasks via HTTP. No Redis, no shared memory, no DistributedSampler.

### Terminology: `score` vs reward

The field `score` throughout this doc is **not** an RL reward. It is the unit test pass ratio for a single trajectory: `tests_passed / total_tests`. A score of `1.0` means all tests passed. A score of `None` means the episode errored out (container crash, timeout, etc.) before evaluation could run. The actual RL reward used for policy optimization is computed elsewhere — the TaskManager only sees evaluation scores and uses them for curriculum sampling decisions.

```
task_manager_service.py   (standalone process, started before torchrun)
    ↑ HTTP
torchrun rank 0  (DP head) → TaskManagerClient → pull_task / push_results
torchrun rank 1  (DP head) → TaskManagerClient → pull_task / push_results
...
torchrun rank N  (non-DP-head, receives broadcast only, no client)
```

There are three components:

1. **TaskManagerService** — FastAPI app, runs as a standalone process
2. **TaskManagerClient** — synchronous HTTP client wrapper
3. **TaskManagerDataset** — PyTorch `IterableDataset` wrapping the client

---

## Component 1: TaskManagerService

File: `task_manager_service.py`

### CLI Arguments

| Parameter | Flag | Default | Description |
|---|---|---|---|
| `dataset_root` | `--dataset-root` | required | Path to Harbor dataset directory |
| `strategy` | `--strategy` | `weighted` | `weighted` (curriculum) or `sequential` (round-robin) |
| `pass_n_hi` | `--pass-n-hi` | `1.0` | pass@n threshold: at or above this → mastered |
| `var_thresh` | `--var-thresh` | `0.05` | score variance threshold: below this with pass@n=0 → too_hard |
| `window_size` | `--window-size` | `4` | rolling window size in groups per task |
| `w_zpd` | `--w-zpd` | `4` | sampling weight for zpd category |
| `w_uncertain` | `--w-uncertain` | `4` | sampling weight for uncertain category |
| `w_too_hard` | `--w-too-hard` | `1` | sampling weight for too_hard category |
| `w_mastered` | `--w-mastered` | `0.2` | sampling weight for mastered category |
| `port` | `--port` | `8765` | HTTP port |
| `host` | `--host` | `0.0.0.0` | bind address |

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/cleanup` | Reset all state, reload dataset from disk |
| `POST` | `/initialize` | Warm-start from prior eval results (weighted mode only, returns 400 in sequential) |
| `GET` | `/pull_task` | Return next task. Blocks until ready |
| `POST` | `/push_results` | Accept a list of trajectory results |
| `GET` | `/stats` | Return per-task stats and category assignments |

### Data Structures

#### TrajectoryRecord

```python
@dataclass
class TrajectoryRecord:
    uid: str              # unique pull identifier (from pull_task response)
    score: float | None   # unit test pass ratio; None = errored out
    group_id: str         # links trajectories from the same rollout group
```

#### Per-Task State

Each task maintains a rolling window of the last `window_size` **groups**. A group is a list of `TrajectoryRecord`s sharing the same `group_id`.

```python
# Key: task_id, Value: deque of groups, max length = window_size
# Each group is a list of TrajectoryRecord
_task_windows: dict[str, deque[list[TrajectoryRecord]]]
```

Incoming trajectories are accumulated into a staging dict keyed by `(task_id, group_id)`. When `push_results` is called, all trajectories in the payload are inserted. Since all trajectories in a group are pushed together in one call, the group is complete on arrival. Append the completed group to the task's window deque. If the deque exceeds `window_size`, the oldest group is evicted automatically.

Deduplicate on `uid`: if a `uid` already exists in the window, skip it. This makes pushes idempotent.

#### Derived Stats (recomputed on recalc)

Stats are computed over all groups in the task's window:

```python
@dataclass
class TaskStats:
    task_id: str
    recent_pass_at_n: float    # mean over groups of (count(score==1.0) / count(non-None))
    recent_mean_score: float  # mean of all non-None scores across window
    recent_variance: float     # mean over groups of var(non-None scores per group)
    n_groups: int              # number of groups in window (≤ window_size)
    n_trajectories: int        # total trajectory records in window
    n_errors: int              # count(score is None) across window
    category: str              # one of: zpd, uncertain, mastered, too_hard, broken
```

### Category Classification

5 categories. The first 4 are sampled with weights. The last is permanently excluded.

```
zpd:        0 < recent_pass_at_n < pass_n_hi          → w_zpd (default 4)
uncertain:  recent_pass_at_n == 0 AND variance >= var_thresh → w_uncertain (default 4)
mastered:   recent_pass_at_n >= pass_n_hi              → w_mastered (default 0.2)
too_hard:   recent_pass_at_n == 0 AND variance < var_thresh  → w_too_hard (default 1)
broken:     any group in window has ALL scores = None → weight 0, permanent
```

`broken` is triggered immediately on receipt of an all-None group, before any recalc cycle. Require 2 consecutive all-None groups before marking broken to avoid false positives from transient infrastructure failures.

There is no `unexplored` category in the sampler. Before a task has any results, it's either in the cold start queue (Phase 1) or simply not in any category queue yet (in-flight during the Phase 1→2 transition). Once results arrive, it gets classified into one of the 5 categories above.

### Sampling

#### Strategy: `sequential`

A single deque of all tasks in sorted order (numeric-first, then lexicographic — matching `load_harbor_dataset`). On pull: `popleft()`. When empty: refill in the same sorted order. Cycles indefinitely. Push records stats for `/stats` but never recalculates categories or changes sampling. `/initialize` returns 400.

#### Strategy: `weighted`

Two phases.

**Phase 1 — Cold start**

A sequential queue of all tasks in sorted order. On `pull_task`: `cold_start_queue.popleft()`. On `push_results`: record stats and check for broken normally (remove broken tasks from cold start queue if still present).

Transition to Phase 2: when `cold_start_queue` is empty, classify all tasks that have results so far, build category queues, set `phase2 = True`. Any tasks still in-flight (~32) are not in any category yet — they're already running and don't need to be sampled. When their results arrive, the push handler classifies them and appends them to the appropriate category queue.

**Phase 2 — Weighted sampling**

#### CategoryPool structure

Each of the 4 active categories (zpd, uncertain, too_hard, mastered) is a `CategoryPool`:

```python
@dataclass
class CategoryPool:
    name: str                # "zpd", "uncertain", "too_hard", "mastered"
    weight: float            # configured weight (e.g. 4.0 for zpd)
    members: set[str]        # task_ids currently classified into this category
    queue: deque[str]        # shuffled sampling order, popleft on each pull

    def is_empty(self) -> bool:
        return len(self.members) == 0
```

`broken` is not a `CategoryPool` — it's just a `set[str]` of excluded task_ids.

The sampler holds all 4 pools:

```python
class CurriculumSampler:
    pools: dict[str, CategoryPool]   # keyed by category name
    task_category: dict[str, str]    # task_id → current category name (for fast lookup)
```

#### Queue lifecycle rules

**Build**: `queue = deque(random.sample(list(members), len(members)))`. Only happens in two cases:
1. Initial build when entering Phase 2 (cold start transition or warm start)
2. When `queue` drains to empty after a `popleft()` — immediate refill from current `members`

**Pull**: `queue.popleft()`. If queue is empty after pop, rebuild immediately.

**Task enters pool** (on reclassification): `members.add(task_id)`, `queue.append(task_id)`. Goes to back of line — sampled this pass, but after existing members.

**Task leaves pool** (on reclassification): `members.discard(task_id)`, remove from `queue` if present. Use `deque.remove()` — O(n) but n is at most ~1000 tasks per category, called rarely.

**No rebuild on reclassification**. The queue only rebuilds on natural drain. This guarantees within-category coverage per pass.

Pull logic:
1. Compute effective weights over non-empty categories: `w_cat / sum(w_c for c in non_empty_categories)`
2. Sample a category by these weights
3. `task_id = category.queue.popleft()`
4. If queue is now empty and members is non-empty: `queue = deque(shuffle(list(members)))` (rebuild)
5. Return the task

Reclassification (triggered on every push, for the pushed task only):
1. Recompute stats for the pushed task
2. Classify it into a category
3. If category changed:
   - Remove from old category: `old.members.remove(task)`, `old.queue.remove(task)` if present
   - Add to new category: `new.members.add(task)`, `new.queue.append(task)` (back of line)
4. Update the non-empty categories set

No batch recalc interval needed. Classification is per-task and cheap (just comparing pass@n and variance against thresholds). Doing it on every push keeps categories maximally up-to-date with zero added complexity.

Queues are lazy — they never rebuild on reclassification. They only rebuild when they naturally drain to empty. This preserves within-category coverage: every task in a category is sampled once before any repeats.

### Concurrency

FastAPI's async event loop serializes handler calls. All state mutations (push, recalc, pull) happen in async handlers without thread delegation mid-operation, so no locks are needed. The bounded slot model (at most 32 concurrent tasks) means pull and push are naturally pairwise sequential per slot: a slot's next pull only happens after its push completes.

### Endpoint Details

#### `POST /cleanup`

Reset all state. Reload the task list from `dataset_root`. Re-initialize the sampler (cold start queue for weighted, sorted queue for sequential). Called by rank 0 before training starts.

Request body: none.
Response: `{"status": "ok", "n_tasks": int}`

#### `POST /initialize`

Warm-start from prior eval results. Only valid in `weighted` strategy (return 400 otherwise). Must be called after `/cleanup`.

Request body:
```json
{"scores_csv": "/path/to/eval_results.csv"}
```

The CSV file has no header. Each row is a task_id followed by the score of each trajectory:

```
42,1.0,0.5,,1.0
install-packages,0.0,0.0,0.0
setup-env,,,
```

Empty cells are treated as `None` (errored trajectory). The number of columns after task_id can vary per row (different tasks may have had different numbers of trajectories).

Processing:
1. Read CSV with `pandas.read_csv(path, header=None)`
2. First column is `task_id` (cast to `str`), remaining columns are scores (`float | None`)
3. For each row:
   - Treat all scores as a single synthetic group (group_id = `"init_{task_id}"`)
   - Store as TrajectoryRecords in the task's window
   - Compute stats and classify category
   - If all scores are None → mark broken immediately
4. Any task in the dataset but NOT in the CSV → mark broken immediately (missing from eval means the task is assumed non-functional)
5. Build category queues, set `phase2 = True`, skip cold start entirely

Response: `{"status": "ok", "n_initialized": int, "n_broken": int, "categories": {"zpd": int, "uncertain": int, "mastered": int, "too_hard": int, "broken": int}}`

#### `GET /pull_task`

Returns one task.

Response:
```json
{
  "task_id": "42",
  "task_path": "/data/harbor/tasks/42",
  "instruction": "contents of instruction.md...",
  "uid": "42_00017"
}
```

`uid` is `f"{task_id}_{monotonic_counter}"` — unique per pull, used for deduplication on push.

Task sort order: numeric-first then lexicographic. E.g. `["1", "2", "10", "100", "install-packages", "setup-env"]`.

#### `POST /push_results`

Accept trajectory results.

Request body:
```json
[
  {"uid": "42_00017", "task_id": "42", "score": 1.0, "group_id": "rank0_a1b2c3"},
  {"uid": "42_00017", "task_id": "42", "score": 0.5, "group_id": "rank0_a1b2c3"},
  {"uid": "42_00017", "task_id": "42", "score": null, "group_id": "rank0_a1b2c3"}
]
```

All items in one call share the same `group_id` and `task_id`. The `uid` is the same for all trajectories from the same pull (they all ran on the same pulled task).

Processing:
1. Deduplicate: skip any trajectory whose `uid + group_id + score` combo already exists
2. Group trajectories by `group_id` (should be one group per call)
3. Append group to `_task_windows[task_id]`
4. Check for broken: if all scores in the group are None, increment the all-None counter for this task. If counter reaches 2 → mark broken, remove from category queues
5. Recompute stats for this task, reclassify if category changed (surgical add/remove on queues)

Response: `{"status": "ok", "n_accepted": int}`

#### `GET /stats`

Response:
```json
{
  "n_tasks": 500,
  "phase": "weighted_phase2",
  "categories": {
    "zpd": ["task_12", "task_45", ...],
    "uncertain": [...],
    "mastered": [...],
    "too_hard": [...],
    "broken": [...]
  },
  "tasks": {
    "task_12": {
      "recent_pass_at_n": 0.4,
      "recent_mean_score": 0.35,
      "recent_variance": 0.12,
      "n_groups": 3,
      "n_trajectories": 48,
      "n_errors": 2,
      "category": "zpd"
    }
  }
}
```

---

## Component 2: TaskManagerClient

File: `task_manager_client.py`

Synchronous HTTP client. All methods use `requests` and block until the server responds.

```python
class TaskManagerClient:
    def __init__(self, base_url: str):
        """
        Args:
            base_url: e.g. "http://localhost:8765"
        """

    def cleanup(self) -> dict:
        """POST /cleanup. Returns {"status": "ok", "n_tasks": int}."""

    def initialize(self, scores_csv: str) -> dict:
        """
        POST /initialize.
        
        Args:
            scores_csv: path to CSV file with eval results
        
        Returns: {"status": "ok", "n_initialized": int, ...}
        """

    def pull_task(self) -> dict:
        """
        GET /pull_task.
        
        Returns: {"task_id": str, "task_path": str, "instruction": str, "uid": str}
        """

    def push_results(self, results: list[dict]) -> dict:
        """
        POST /push_results.
        
        Args:
            results: list of {"uid": str, "task_id": str, "score": float | None, "group_id": str}
        
        Returns: {"status": "ok", "n_accepted": int}
        """

    def stats(self) -> dict:
        """GET /stats. Returns full stats payload."""
```

All methods raise on HTTP errors (non-2xx status). No retry logic built in — the caller handles retries.

---

## Component 3: TaskManagerDataset

File: `task_manager_dataset.py`

A PyTorch `IterableDataset` that wraps `TaskManagerClient`. Each `__next__` blocks on `pull_task()`.

```python
class TaskManagerDataset(IterableDataset):
    def __init__(self, client: TaskManagerClient):
        """
        Args:
            client: a TaskManagerClient instance
        """

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """
        Calls client.pull_task() and returns the result dict.
        Returns: {"task_id": str, "task_path": str, "instruction": str, "uid": str}
        """
```

Usage in training:
```python
client = TaskManagerClient("http://localhost:8765")
dataset = TaskManagerDataset(client)
dataloader = StatefulDataLoader(dataset, batch_size=N)

# Each batch yields a list of N dicts with keys: task_id, task_path, instruction, uid
```

No `DistributedSampler` needed. The service distributes tasks atomically via HTTP. Only DP-head ranks instantiate the dataset/dataloader.

---

## Pydantic Models

Define these for request/response validation in the service.

```python
from pydantic import BaseModel

class ResultItem(BaseModel):
    uid: str
    task_id: str
    score: float | None
    group_id: str

class PushResultsRequest(BaseModel):
    results: list[ResultItem]
    # Note: the endpoint can also accept a bare list[ResultItem] as the body.
    # Choose one convention and stick with it.

class TaskItem(BaseModel):
    task_id: str
    task_path: str
    instruction: str
    uid: str
```

---

## Integration with arun_episode

After `rollout.run()` returns all trajectories:

```python
import uuid

async def arun_episode(self, engine, data):
    task_id = data["task_id"]
    group_id = f"{rank_id}_{uuid.uuid4().hex[:8]}"  # unique per group, no collisions

    results = await rollout.run(data, n_trajs=self.n_trajs)
    # results: list of (run_info, score)  where score: float | None

    await asyncio.to_thread(
        self.client.push_results,
        [
            {"uid": data["uid"], "task_id": task_id,
             "score": score, "group_id": group_id}
            for _, score in results
        ],
    )
```

---

## Test Plan

All tests in `test_task_manager.py`. Spin up a real server subprocess so client and dataset run exactly as in training. Use a small mock dataset (3–5 tasks) with controllable scores.

### Test 1: Sequential mode basics
- Verify tasks served in numeric-first then lexicographic order
- Verify second cycle repeats the same order
- Verify all tasks sampled equally over 300 pulls (uniform distribution)
- Verify `/initialize` returns 400

### Test 2: Cold start → Phase 2 transition
- Start in weighted mode with 3 tasks and concurrency of 1 (single slot)
- Pull and push all 3 tasks with known scores: easy=1.0, medium=0.4, hard=0.0
- Verify cold start queue drains after 3 pulls
- Verify Phase 2 activates with correct categories: easy→mastered, medium→zpd, hard→too_hard (or uncertain depending on variance)
- Pull 200 more tasks, verify sampling weights match expected distribution

### Test 3: Warm start via /initialize
- Create a CSV file: `easy,1.0,1.0,1.0\nmedium,0.4,0.0,0.4\nhard,0.0,0.0,0.0`
- Call `/cleanup` then `/initialize` with the CSV path
- Verify categories assigned immediately: easy→mastered, medium→zpd, hard→too_hard
- Verify `/stats` reflects synthetic records
- Verify first `pull_task` uses weighted sampling (no cold start)
- Add a 4th task to the dataset that is NOT in the CSV → verify it is immediately broken
- Verify a CSV row with all empty cells (all-None) → broken immediately

### Test 4: Broken detection requires 2 all-None groups
- Push one group with all scores=None for a task
- Verify task is NOT yet broken (only 1 all-None group)
- Push a second all-None group for the same task
- Verify task is now broken, weight=0, never sampled again
- Push a third group with valid scores — verify task stays broken

### Test 5: Category transitions
- Start with medium task in zpd (score=0.4)
- Push groups with score=1.0 → verify it transitions to mastered
- Push groups with score=0.0 → verify it transitions back (mastered → too_hard or uncertain)
- Verify queue membership updates correctly on each transition

### Test 6: Queue coverage guarantee
- Put 10 tasks in zpd category
- Pull all 10 → verify each appears exactly once (no repeats)
- Pull 10 more → verify reshuffled, each appears exactly once again
- Mid-pass (after pulling 5), push results that cause 2 new tasks to enter zpd
- Verify the 2 new tasks appear in the remaining pulls of this pass
- Verify none of the already-pulled 5 appear again until next pass

### Test 7: group_id and stats correctness
- Push 16 trajectories with same group_id, scores = [1.0]*8 + [0.0]*8
- Verify pass_at_n = 0.5 (8 full passes out of 16)
- Verify variance computed correctly within the group
- Push another group with different group_id, verify rolling window has 2 groups

### Test 8: Idempotent push
- Push a batch of results
- Push the exact same batch again
- Verify no duplicate records in the window, stats unchanged

### Test 9: TaskManagerDataset integration
- Create a dataset + dataloader with batch_size=2
- Pull one batch, verify it returns 2 TaskItem dicts
- Verify each dict has task_id, task_path, instruction, uid fields

### Test 10: Empty category handling
- Set up state where all tasks are mastered (only one non-empty category)
- Verify pull_task still works (samples from the only non-empty category)
- Mark all tasks broken → verify pull_task handles the "no tasks available" case gracefully (returns error or blocks)

---

## Monitoring

The service should track state over time for a dashboard. This requires two additions to the service internals:

### Additional State to Track

```python
@dataclass
class ServiceMetrics:
    # Counters (monotonically increasing)
    total_pulls: int = 0
    total_pushes: int = 0          # number of push_results calls
    total_trajectories: int = 0    # total individual trajectory records received
    total_errors: int = 0          # total None scores received

    # Timestamps
    service_start_time: float = 0.0
    last_pull_time: float = 0.0
    last_push_time: float = 0.0

    # Per-task timestamps
    last_push_per_task: dict[str, float] = field(default_factory=dict)

    # Transition log (ring buffer, keep last 1000)
    transitions: deque[TransitionEvent] = field(default_factory=lambda: deque(maxlen=1000))

    # Category size snapshots (list of (timestamp, {cat: count}) for plotting)
    category_snapshots: deque[tuple[float, dict[str, int]]] = field(default_factory=lambda: deque(maxlen=10000))

@dataclass
class TransitionEvent:
    timestamp: float
    task_id: str
    old_category: str
    new_category: str
    trigger: str              # "push" or "initialize"
```

### When to Record

- **On every pull**: increment `total_pulls`, update `last_pull_time`
- **On every push**: increment `total_pushes`, `total_trajectories`, `total_errors`, update `last_push_time` and `last_push_per_task[task_id]`
- **On every reclassification that changes a task's category**: append to `transitions`
- **On every push**: append a category snapshot `(time, {zpd: n, uncertain: n, ...})`

### Extended `/stats` Response

```json
{
  "n_tasks": 500,
  "phase": "weighted_phase2",
  "metrics": {
    "total_pulls": 3456,
    "total_pushes": 3400,
    "total_trajectories": 54400,
    "total_errors": 230,
    "error_rate": 0.0042,
    "uptime_seconds": 7200,
    "pulls_per_minute": 8.0,
    "pushes_per_minute": 7.9
  },
  "categories": {
    "zpd": {"count": 245, "task_ids": [...]},
    "uncertain": {"count": 89, "task_ids": [...]},
    "mastered": {"count": 40, "task_ids": [...]},
    "too_hard": {"count": 120, "task_ids": [...]},
    "broken": {"count": 6, "task_ids": [...]}
  },
  "recent_transitions": [
    {"timestamp": 1711612800.0, "task_id": "42", "old": "zpd", "new": "mastered", "trigger": "push"},
    {"timestamp": 1711612805.0, "task_id": "99", "old": "uncertain", "new": "zpd", "trigger": "push"}
  ],
  "category_history": [
    [1711612000.0, {"zpd": 300, "uncertain": 100, "mastered": 10, "too_hard": 85, "broken": 5}],
    [1711612060.0, {"zpd": 298, "uncertain": 99, "mastered": 12, "too_hard": 86, "broken": 5}]
  ],
  "tasks": {
    "task_12": {
      "recent_pass_at_n": 0.4,
      "recent_mean_score": 0.35,
      "recent_variance": 0.12,
      "n_groups": 3,
      "n_trajectories": 48,
      "n_errors": 2,
      "category": "zpd",
      "last_push_time": 1711612800.0
    }
  }
}
```

### Dashboard Panels

The dashboard is a standalone HTML page that polls `GET /stats` every 30 seconds.

**Panel 1 — Category distribution over time**: Stacked area chart from `category_history`. The key training signal: mastered count should grow, zpd/too_hard should shrink. Flatline = model not learning. Broken spike = infra problem.

**Panel 2 — Throughput**: Time series of `pulls_per_minute`, `pushes_per_minute`. Should be steady and roughly equal. Pulls >> pushes means slots are stalling. Both dropping means service unhealthy.

**Panel 3 — Error tracking**: `error_rate` over time, cumulative `total_errors`, list of recently broken tasks from `recent_transitions` filtered to `new == "broken"`. Error rate spike = infrastructure problem.

**Panel 4 — Curriculum health**: Mean `recent_pass_at_n` and `recent_mean_score` across all tasks over time (the learning curve). `recent_transitions` log showing category movements. Count of tasks stuck in the same category for many groups.

**Panel 5 — Per-task table**: Sortable by any column. Filter by category. Highlight rows where `last_push_time` is stale (> 10 min ago). Click to expand rolling window detail.

---

## Claude Code Implementation Prompt

Use the following prompt to have Claude Code implement this system. Copy the entire design doc into the project directory first, then run Claude Code with this prompt:

```
Read the design document at ./task_manager_design.md carefully. This is the
complete spec for a TaskManager curriculum learning system with 3 components:
service, client, and dataset.

Implement in this order, testing each phase before moving to the next:

PHASE 1 — Data model and core service skeleton
  Files: task_manager_service.py
  - Implement TrajectoryRecord, TaskStats, CategoryPool, CurriculumSampler dataclasses
  - Implement the FastAPI app with CLI argument parsing (argparse)
  - Implement /cleanup endpoint: load task list from dataset_root
    (each subdirectory is a task, read instruction.md from each)
  - Implement task sort order: numeric-first then lexicographic
  - Write tests for: task loading, sort order
  - Run tests, fix any failures before proceeding

PHASE 2 — Sequential strategy
  Files: task_manager_service.py, task_manager_client.py
  - Implement /pull_task for sequential mode (single cycling deque)
  - Implement /push_results (record stats, no reclassification)
  - Implement /stats endpoint
  - Implement TaskManagerClient (all methods, synchronous requests)
  - Implement /initialize returning 400 in sequential mode
  - Write tests: Test 1 (sequential basics) from the test plan
  - Use a real server subprocess in tests (not TestClient)
  - Run tests, fix any failures before proceeding

PHASE 3 — Weighted strategy: cold start (Phase 1 sampling)
  Files: task_manager_service.py
  - Implement cold start sequential queue for weighted mode
  - Implement broken detection (2 consecutive all-None groups)
  - Implement stats computation: per-group pass@n, variance, aggregation
  - Implement category classification (zpd, uncertain, mastered, too_hard, broken)
  - Write tests: Test 2 (cold start transition), Test 4 (broken detection)
  - Run tests, fix any failures before proceeding

PHASE 4 — Weighted strategy: Phase 2 sampling
  Files: task_manager_service.py
  - Implement CategoryPool with members set and queue deque
  - Implement weighted category sampling with non-empty normalization
  - Implement reclassification on every push (surgical queue add/remove)
  - Implement queue rebuild on natural drain
  - Implement Phase 1 → Phase 2 transition (on cold start queue empty)
  - Write tests: Test 5 (category transitions), Test 6 (queue coverage),
    Test 7 (group_id stats), Test 8 (idempotent push), Test 10 (empty categories)
  - Run tests, fix any failures before proceeding

PHASE 5 — Warm start
  Files: task_manager_service.py
  - Implement /initialize: read CSV with pandas, classify all tasks,
    missing tasks → broken, build category queues, skip to Phase 2
  - Update TaskManagerClient.initialize() to send CSV path
  - Write tests: Test 3 (warm start)
  - Run tests, fix any failures before proceeding

PHASE 6 — Dataset
  Files: task_manager_dataset.py
  - Implement TaskManagerDataset (IterableDataset wrapping client)
  - Write tests: Test 9 (dataset integration with dataloader)
  - Run tests, fix any failures before proceeding

PHASE 7 — Monitoring
  Files: task_manager_service.py
  - Add ServiceMetrics and TransitionEvent tracking
  - Record metrics on every pull, push, and reclassification
  - Extend /stats response with metrics, recent_transitions, category_history
  - Write a test that verifies metrics increment correctly after a sequence
    of pulls and pushes
  - Run tests, fix any failures before proceeding

IMPORTANT IMPLEMENTATION NOTES:
- All tests must use a real server subprocess (subprocess.Popen), not FastAPI
  TestClient — this matches how the service runs in production
- Create a small mock dataset directory (3-5 tasks) in a temp dir for tests,
  each task is a subdirectory with an instruction.md file
- Use pytest fixtures for server lifecycle (start before tests, kill after)
- Pydantic models for request/response validation
- The service is async (FastAPI) but all state mutations happen in the event
  loop without thread delegation — no locks needed
- TaskManagerClient is synchronous (uses requests library)
- group_id should use f"{rank_id}_{uuid.uuid4().hex[:8]}" not str(time.time())
- Deduplicate pushes on uid to make them idempotent
```