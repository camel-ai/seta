"""Continuous rollout worker for the seta env_service rollout (Miles train_async.py).

    --rollout-function-path common.env_service_rollout.generate_rollout_fully_async

Adapted from Miles' ``examples/fully_async/fully_async_rollout.py``. A background
thread keeps prompt groups in flight through ``generate_and_rm_group`` independently
of the trainer; each training step takes the next ``rollout_batch_size`` finished
groups. Differences from the Miles example:

* Concurrency is decoupled from the batch: ``ROLLOUT_CONCURRENCY`` groups are kept in
  flight (default ``rollout_batch_size``). Env_service trajectories take minutes and
  their durations vary widely, so running more groups than a step consumes keeps the
  sandboxes busy and lets a step take the groups that finish first.
* ``--dynamic-sampling-filter-path`` is applied (the Miles example ignores it on this
  path). A rejected group is dropped, not re-queued: the worker pulls a fresh prompt
  for the free slot.
* A group containing an aborted sample is dropped, not returned to the data buffer.

``--max-weight-staleness N`` recycles a finished group whose oldest weight version is
more than N versions behind the engines' current one (its samples are reset and
returned to the data buffer). The current version is read from the SGLang router's
``/model_info``; if that endpoint reports no version, nothing is recycled.
"""

import asyncio
import atexit
import logging
import os
import queue
import threading
import time

import aiohttp

from miles.rollout.data_source import DataSource
from miles.rollout.filter_hub.base_types import call_dynamic_filter
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def group_oldest_weight_version(group: list[Sample]) -> int | None:
    """Minimum weight version across all trajectories and turns of a group."""
    versions = [s.oldest_weight_version for s in group if s.oldest_weight_version is not None]
    return min(versions) if versions else None


class _CachedWeightVersion:
    """Throttled query of the current engine weight version via /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query: float = 0.0

    async def get(self, args) -> int | None:
        now = time.monotonic()
        if self._value is not None and (now - self._last_query) < self._ttl:
            return self._value
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._value = int(data["weight_version"])
                        self._last_query = now
        except Exception as e:
            logger.debug(f"Failed to query engine weight version: {e}")
        return self._value


_cached_version = _CachedWeightVersion()

_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer: DataSource):
    """Get or create the global worker."""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop the global worker."""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """Rollout worker on its own thread; outlives individual rollout-function calls."""

    def __init__(self, args, data_buffer: DataSource, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)
        self.worker_thread = None
        self.state = GenerateState(args)

    async def continuous_worker_loop(self):
        """Keep pulling prompt groups from the data buffer and generating them."""
        print("Continuous async rollout worker started")

        active_tasks = set()
        # Groups in flight, independent of the train batch (rollout_batch_size groups
        # per step).
        max_concurrent_tasks = int(os.environ.get("ROLLOUT_CONCURRENCY") or self.args.rollout_batch_size)
        group_id_counter = 0

        while self.running:
            try:
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # results are delivered by the done-callback
                        except Exception as e:
                            print(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks

                while len(active_tasks) < max_concurrent_tasks and self.running:
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        def make_callback(gid):
                            def task_done_callback(done_task):
                                result = done_task.result()
                                self.output_queue.put((gid, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        break

                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def get_completed_groups(self) -> list[tuple]:
        completed = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


async def generate_rollout_async(args, rollout_id: int, data_buffer: DataSource) -> list[list[Sample]]:
    """Collect rollout_batch_size finished groups from the global worker."""
    assert args.rollout_global_dataset

    worker = get_global_worker(args, data_buffer)

    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True
    stale_groups_recycled = 0
    staleness_values = []

    use_staleness_filter = getattr(args, "max_weight_staleness", None) is not None

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None) is not None
        else None
    )
    dynamic_dropped = 0
    dynamic_reasons: dict = {}

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")
    if use_staleness_filter:
        print(f"Staleness filter enabled: max_weight_staleness={args.max_weight_staleness}")

    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # warn after this long without a finished group

    while len(data) < target_data_size:
        completed = worker.get_completed_groups()

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        current_engine_version = None
        if use_staleness_filter:
            current_engine_version = await _cached_version.get(args)

        processed_any = False

        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                continue

            oldest = group_oldest_weight_version(group) if use_staleness_filter else None
            if oldest is not None and current_engine_version is not None:
                staleness = current_engine_version - oldest
                staleness_values.append(staleness)
                if staleness > args.max_weight_staleness:
                    try:
                        for s in group:
                            s.reset_for_retry()
                        data_buffer.add_samples([group])
                    except Exception as e:
                        logger.warning(f"Failed to recycle stale group {group_id}: {e}")
                    stale_groups_recycled += 1
                    logger.info(
                        f"Recycled stale group {group_id} "
                        f"(oldest_version={oldest}, current={current_engine_version}, "
                        f"staleness={staleness} > max={args.max_weight_staleness})"
                    )
                    continue

            if do_print:
                print(
                    f"First rollout sample: {[str(group[0].prompt) + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            if dynamic_filter is not None:
                out = call_dynamic_filter(dynamic_filter, args, group)
                if not out.keep:
                    dynamic_dropped += 1
                    dynamic_reasons[out.reason] = dynamic_reasons.get(out.reason, 0) + 1
                    continue

            data.append(group)
            processed_any = True

        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")
    if stale_groups_recycled > 0 or staleness_values:
        avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
        print(
            f"Staleness stats: recycled={stale_groups_recycled}, "
            f"avg_staleness={avg_staleness:.1f}, "
            f"max_staleness={max(staleness_values) if staleness_values else 0}"
        )
    if dynamic_dropped > 0:
        print(f"Dynamic-filter stats: dropped={dynamic_dropped} reasons={dynamic_reasons}", flush=True)

    if data:
        print(
            f"Finish rollout: {[str(data[-1][0].prompt) + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    return data


def generate_rollout_fully_async(args, rollout_id, data_buffer: DataSource, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


atexit.register(stop_global_worker)
