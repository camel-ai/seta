# Dynamic Task Sampling — Usage

## Components

**`task_manager_service.py`** — FastAPI service (one instance, standalone process).
Owns the dataset, weighted sampler, and reward history in memory. Exposes HTTP
endpoints for pulling tasks and reporting results. Recalculates per-task sampling
weights every N results received. Tasks that always error are immediately zeroed out.

**`task_manager_client.py`** — Thin sync HTTP client (one instance per DP-head rank).
Wraps the service endpoints. Used in two places: inside `TaskManagerDataset` to pull
tasks, and inside `arun_episode` to push results. Call `push_results` via
`asyncio.to_thread()` when inside an async context to avoid blocking the event loop.

**`task_dataset.py`** — `IterableDataset` backed by the client (one instance per DP-head rank).
Each `__next__` blocks on `pull_task()` until the service returns a task. Pass directly
to `StatefulDataLoader` — no `DistributedSampler` needed, the service distributes tasks
atomically across all ranks via HTTP.

---

## Setup before training / evaluation

### 1. Start the TaskManager service

Run this before launching `torchrun`. The service must be reachable from all DP-head ranks.

```bash
python -m seta_env.datahubs.task_manager_service \
    --dataset-root /path/to/harbor/dataset \
    --port 8765 \
    --acc-max 1.0 \
    --weight-update-interval 0 \   # 0 = len(dataset), recalc once per full pass
    --max-records-per-task 32
```

### 2. In train.py / eval.py — cleanup and wire the dataloader

```python
import asyncio
from seta_env.datahubs.task_manager_client import TaskManagerClient
from seta_env.datahubs.task_dataset import TaskManagerDataset
from torchdata.stateful_dataloader import StatefulDataLoader

rank = int(os.getenv("RANK", "0"))
client = TaskManagerClient("http://localhost:8765")

# Rank 0 resets service state; all ranks wait before pulling tasks.
if rank == 0:
    client.cleanup()
dist.barrier()

# Only DP-head ranks need the dataloader.
if train_engine.is_data_parallel_head():
    dataset = TaskManagerDataset(client)        # or TaskManagerDataset("http://localhost:8765")
    dataloader = StatefulDataLoader(dataset, batch_size=config.train_dataset.batch_size)
```

### 3. In arun_episode — push results after each episode

```python
async def arun_episode(self, engine, data):
    uid = data["uid"]
    task_id = data["task_id"]
    ...
    # reward is float on success, None if the episode errored out
    await asyncio.to_thread(
        self.client.push_results,
        [{"uid": uid, "task_id": task_id, "reward": reward}],
    )
```

### 4. Monitor sampling weights (optional)

```bash
curl http://localhost:8765/stats | python -m json.tool
```

Returns per-task `mean_reward`, `weight`, `n_results`, and `n_errors`.
