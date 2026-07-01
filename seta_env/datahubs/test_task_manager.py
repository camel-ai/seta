"""Tests for TaskManager service, client, and dataset.

All tests spin up a real server subprocess to match production behaviour.
A small mock dataset (3–5 tasks) is created in a temp directory.
"""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import pytest
import requests

from seta_env.datahubs.task_manager_client import TaskManagerClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_PORT = 18700  # base; each test class gets its own port


def _find_free_port(start: int = BASE_PORT) -> int:
    """Find a free port starting from *start*."""
    import socket

    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found")


def _create_mock_dataset(root: Path, task_names: list[str]) -> None:
    """Create minimal harbor-like task dirs with instruction.md."""
    for name in task_names:
        task_dir = root / name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "instruction.md").write_text(f"Do task {name}")


def _start_server(
    dataset_root: str,
    port: int,
    strategy: str = "weighted",
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start the service as a subprocess and wait until it's ready."""
    cmd = [
        sys.executable,
        "-m",
        "seta_env.datahubs.task_manager_service",
        "--dataset-root",
        dataset_root,
        "--port",
        str(port),
        "--strategy",
        strategy,
        "--host",
        "127.0.0.1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),  # repo root
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the server to be ready
    url = f"http://127.0.0.1:{port}/stats"
    for _ in range(60):
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(0.25)

    proc.kill()
    stdout, stderr = proc.communicate()
    raise RuntimeError(
        f"Server did not start in time.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
    )


def _stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_dataset_3():
    """Create a temp dataset with 3 tasks: '1', '2', 'alpha'."""
    with tempfile.TemporaryDirectory() as tmp:
        _create_mock_dataset(Path(tmp), ["1", "2", "alpha"])
        yield tmp


@pytest.fixture(scope="module")
def mock_dataset_4():
    """Create a temp dataset with 4 tasks: '1', '2', 'alpha', 'beta'."""
    with tempfile.TemporaryDirectory() as tmp:
        _create_mock_dataset(Path(tmp), ["1", "2", "alpha", "beta"])
        yield tmp


@pytest.fixture(scope="module")
def mock_dataset_10():
    """Create a temp dataset with 10 tasks: '1'..'10'."""
    with tempfile.TemporaryDirectory() as tmp:
        _create_mock_dataset(Path(tmp), [str(i) for i in range(1, 11)])
        yield tmp


# ---------------------------------------------------------------------------
# Test 1: Sequential mode basics
# ---------------------------------------------------------------------------


class TestSequentialMode:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(mock_dataset_3, self.port, strategy="sequential")
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_sort_order_and_cycling(self):
        """Tasks served in numeric-first then lexicographic order, cycling."""
        expected = ["1", "2", "alpha"]
        # First cycle
        first_cycle = [self.client.pull_task()["task_id"] for _ in range(3)]
        assert first_cycle == expected
        # Second cycle — same order
        second_cycle = [self.client.pull_task()["task_id"] for _ in range(3)]
        assert second_cycle == expected

    def test_uniform_distribution(self):
        """All tasks sampled equally over 300 pulls."""
        counts: Counter = Counter()
        for _ in range(300):
            task = self.client.pull_task()
            counts[task["task_id"]] += 1
        assert counts["1"] == 100
        assert counts["2"] == 100
        assert counts["alpha"] == 100

    def test_initialize_returns_400(self):
        """initialize is invalid in sequential mode."""
        with pytest.raises(requests.HTTPError) as exc_info:
            self.client.initialize("/nonexistent.csv")
        assert exc_info.value.response.status_code == 400


# ---------------------------------------------------------------------------
# Test 2: Cold start → Phase 2 transition
# ---------------------------------------------------------------------------


class TestColdStartTransition:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(
            mock_dataset_3,
            self.port,
            strategy="weighted",
            extra_args=["--window-size", "4", "--var-thresh", "0.01"],
        )
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_cold_start_to_phase2(self):
        # Pull all 3 tasks (cold start)
        pulled = []
        for _ in range(3):
            t = self.client.pull_task()
            pulled.append(t)

        # Expected cold start order
        assert [p["task_id"] for p in pulled] == ["1", "2", "alpha"]

        # Push results: 1=mastered, 2=zpd, alpha=too_hard
        # pass@n = count(score==1.0) / count(non-None), so zpd needs a mix of 1.0 and non-1.0
        gid = lambda: f"test_{uuid.uuid4().hex[:8]}"

        # Task 1: all 1.0 → mastered (pass@n=1.0)
        self.client.push_results([
            {"uid": pulled[0]["uid"], "task_id": "1", "score": 1.0, "group_id": gid()},
        ])

        # Task 2: mix of 1.0 and 0.0 → zpd (0 < pass@n < 1.0)
        g = gid()
        self.client.push_results([
            {"uid": pulled[1]["uid"], "task_id": "2", "score": 1.0, "group_id": g},
            {"uid": pulled[1]["uid"], "task_id": "2", "score": 0.0, "group_id": g},
        ])

        # Task alpha: all 0.0 → too_hard (pass@n=0, variance=0 < 0.01)
        g = gid()
        self.client.push_results([
            {"uid": pulled[2]["uid"], "task_id": "alpha", "score": 0.0, "group_id": g},
        ])

        # Verify phase 2 is active
        stats = self.client.stats()
        assert stats["phase"] == "weighted_phase2"

        # Check categories
        categories = stats["categories"]
        assert "1" in categories["mastered"]["task_ids"]
        assert "2" in categories["zpd"]["task_ids"]
        assert "alpha" in categories["too_hard"]["task_ids"]

        # Pull 200 more tasks — verify weighted sampling
        counts: Counter = Counter()
        for _ in range(200):
            t = self.client.pull_task()
            counts[t["task_id"]] += 1

        # zpd (w=4) and too_hard (w=1) should dominate over mastered (w=0.2)
        # task 2 (zpd, w=4) should be pulled much more than task 1 (mastered, w=0.2)
        assert counts["2"] > counts["1"], f"zpd should be sampled more than mastered: {counts}"


# ---------------------------------------------------------------------------
# Test 3: Warm start via /initialize
# ---------------------------------------------------------------------------


class TestWarmStart:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_4):
        self.port = _find_free_port()
        self.dataset_root = mock_dataset_4
        self.proc = _start_server(
            mock_dataset_4,
            self.port,
            strategy="weighted",
            extra_args=["--var-thresh", "0.01"],
        )
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_warm_start(self):
        # Create CSV: 1=mastered, 2=zpd, alpha=too_hard, beta NOT in CSV → broken
        # pass@n = count(score==1.0)/count(non-None)
        # 2 needs pass@n > 0 → needs at least one score=1.0
        csv_path = os.path.join(self.dataset_root, "eval_results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["1", 1.0, 1.0, 1.0])       # pass@n=1.0 → mastered
            w.writerow(["2", 1.0, 0.0, 0.0])        # pass@n=1/3 → zpd
            w.writerow(["alpha", 0.0, 0.0, 0.0])    # pass@n=0, var=0 → too_hard

        result = self.client.initialize(csv_path)
        assert result["status"] == "ok"
        assert result["n_initialized"] == 3
        assert result["categories"]["broken"] >= 1  # beta is broken

        stats = self.client.stats()
        assert stats["phase"] == "weighted_phase2"
        assert "1" in stats["categories"]["mastered"]["task_ids"]
        assert "2" in stats["categories"]["zpd"]["task_ids"]
        assert "alpha" in stats["categories"]["too_hard"]["task_ids"]
        assert "beta" in stats["categories"]["broken"]["task_ids"]

        # First pull should use weighted sampling (no cold start)
        task = self.client.pull_task()
        assert task["task_id"] in ["1", "2", "alpha"]  # not beta (broken)

    def test_all_none_csv_row_is_broken(self):
        self.client.cleanup()
        csv_path = os.path.join(self.dataset_root, "eval_results2.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["1", 1.0, 1.0, 1.0])
            w.writerow(["2", 1.0, 0.0, 0.0])
            w.writerow(["alpha", "", "", ""])  # all None → broken

        result = self.client.initialize(csv_path)
        assert "alpha" in [
            tid
            for tid in self.client.stats()["categories"]["broken"]["task_ids"]
        ]


# ---------------------------------------------------------------------------
# Test 4: Broken detection requires 2 all-None groups
# ---------------------------------------------------------------------------


class TestBrokenDetection:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(
            mock_dataset_3,
            self.port,
            strategy="weighted",
            extra_args=["--var-thresh", "0.01"],
        )
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_broken_requires_two_all_none(self):
        # Pull all 3 to drain cold start
        tasks = [self.client.pull_task() for _ in range(3)]

        # Push valid results for task 1 and alpha
        self.client.push_results([
            {"uid": tasks[0]["uid"], "task_id": "1", "score": 1.0, "group_id": "g1"},
        ])
        self.client.push_results([
            {"uid": tasks[2]["uid"], "task_id": "alpha", "score": 0.5, "group_id": "g3"},
        ])

        # Push first all-None group for task 2
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": None, "group_id": "g_none1"},
        ])

        stats = self.client.stats()
        # Task 2 should NOT be broken yet
        assert "2" not in stats["categories"]["broken"]["task_ids"]

        # Push second all-None group → now broken
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": None, "group_id": "g_none2"},
        ])

        stats = self.client.stats()
        assert "2" in stats["categories"]["broken"]["task_ids"]

        # Push a third group with valid scores — still broken
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 0.8, "group_id": "g_valid"},
        ])
        stats = self.client.stats()
        assert "2" in stats["categories"]["broken"]["task_ids"]


# ---------------------------------------------------------------------------
# Test 5: Category transitions
# ---------------------------------------------------------------------------


class TestCategoryTransitions:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(
            mock_dataset_3,
            self.port,
            strategy="weighted",
            extra_args=["--window-size", "2", "--var-thresh", "0.01"],
        )
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_transitions(self):
        # Drain cold start
        tasks = [self.client.pull_task() for _ in range(3)]

        # 1 → mastered, 2 → zpd (mix of 1.0 and 0.0), alpha → too_hard
        self.client.push_results([
            {"uid": tasks[0]["uid"], "task_id": "1", "score": 1.0, "group_id": "g1"},
        ])
        # Task 2: pass@n = 1/2 = 0.5 → zpd
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 1.0, "group_id": "g2"},
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 0.0, "group_id": "g2"},
        ])
        self.client.push_results([
            {"uid": tasks[2]["uid"], "task_id": "alpha", "score": 0.0, "group_id": "g3"},
        ])

        stats = self.client.stats()
        assert "2" in stats["categories"]["zpd"]["task_ids"]

        # Push two groups with all 1.0 to transition task 2 to mastered
        # window_size=2, so after 2 groups of all 1.0, mean pass@n = 1.0
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 1.0, "group_id": "g2b"},
        ])
        # window: [group(1.0,0.0)→pass@n=0.5], [group(1.0)→pass@n=1.0] → mean=0.75 → still zpd
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 1.0, "group_id": "g2c"},
        ])
        # window: [group(1.0)→pass@n=1.0], [group(1.0)→pass@n=1.0] → mean=1.0 → mastered
        stats = self.client.stats()
        assert "2" in stats["categories"]["mastered"]["task_ids"]

        # Push two groups with score=0.0 → transitions to too_hard
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 0.0, "group_id": "g2d"},
        ])
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 0.0, "group_id": "g2e"},
        ])
        stats = self.client.stats()
        assert "2" in stats["categories"]["too_hard"]["task_ids"]


# ---------------------------------------------------------------------------
# Test 6: Queue coverage guarantee
# ---------------------------------------------------------------------------


class TestQueueCoverage:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_10):
        self.port = _find_free_port()
        self.proc = _start_server(
            mock_dataset_10,
            self.port,
            strategy="weighted",
            extra_args=["--var-thresh", "0.01"],
        )
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_coverage_within_category(self):
        # Drain cold start: 10 tasks
        cold_tasks = [self.client.pull_task() for _ in range(10)]

        # Push all as zpd: need 0 < pass@n < 1.0, so mix of 1.0 and 0.0
        for t in cold_tasks:
            gid = f"g_{t['task_id']}"
            self.client.push_results([
                {"uid": t["uid"], "task_id": t["task_id"], "score": 1.0, "group_id": gid},
                {"uid": t["uid"], "task_id": t["task_id"], "score": 0.0, "group_id": gid},
            ])

        stats = self.client.stats()
        assert stats["phase"] == "weighted_phase2"
        # All should be zpd
        assert stats["categories"]["zpd"]["count"] == 10

        # Pull 10 — each should appear exactly once (single category = 100% weight)
        first_pass = [self.client.pull_task()["task_id"] for _ in range(10)]
        assert sorted(first_pass) == sorted(str(i) for i in range(1, 11))

        # Pull 10 more — reshuffled but each appears once
        second_pass = [self.client.pull_task()["task_id"] for _ in range(10)]
        assert sorted(second_pass) == sorted(str(i) for i in range(1, 11))

        # Verify different order (reshuffled) — could fail with p=1/10! but negligible
        # Skip this assertion as it's probabilistic


# ---------------------------------------------------------------------------
# Test 7: group_id and stats correctness
# ---------------------------------------------------------------------------


class TestGroupStats:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(
            mock_dataset_3,
            self.port,
            strategy="weighted",
            extra_args=["--window-size", "4"],
        )
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_stats_computation(self):
        # Drain cold start
        tasks = [self.client.pull_task() for _ in range(3)]
        # Push valid results for 2 and alpha so they don't stay empty
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 0.5, "group_id": "other1"},
        ])
        self.client.push_results([
            {"uid": tasks[2]["uid"], "task_id": "alpha", "score": 0.5, "group_id": "other2"},
        ])

        # Push 16 trajectories for task 1: 8 × 1.0 + 8 × 0.0
        group1_id = "grp_16traj"
        results = []
        for i in range(8):
            results.append({"uid": tasks[0]["uid"], "task_id": "1", "score": 1.0, "group_id": group1_id})
        for i in range(8):
            results.append({"uid": tasks[0]["uid"], "task_id": "1", "score": 0.0, "group_id": group1_id})
        self.client.push_results(results)

        stats = self.client.stats()
        task1 = stats["tasks"]["1"]
        # pass@n for this group: 8 scores of 1.0 out of 16 non-None = 0.5
        assert abs(task1["recent_pass_at_n"] - 0.5) < 0.01
        assert task1["n_groups"] == 1
        assert task1["n_trajectories"] == 16

        # Push another group with different group_id
        group2_id = "grp_second"
        self.client.push_results([
            {"uid": tasks[0]["uid"], "task_id": "1", "score": 1.0, "group_id": group2_id},
            {"uid": tasks[0]["uid"], "task_id": "1", "score": 1.0, "group_id": group2_id},
        ])

        stats = self.client.stats()
        task1 = stats["tasks"]["1"]
        assert task1["n_groups"] == 2
        # group1: pass@n=0.5, group2: pass@n=1.0 → mean = 0.75
        assert abs(task1["recent_pass_at_n"] - 0.75) < 0.01


# ---------------------------------------------------------------------------
# Test 8: Idempotent push
# ---------------------------------------------------------------------------


class TestIdempotentPush:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(mock_dataset_3, self.port, strategy="weighted")
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_duplicate_push(self):
        tasks = [self.client.pull_task() for _ in range(3)]

        batch = [
            {"uid": tasks[0]["uid"], "task_id": "1", "score": 0.5, "group_id": "dup_g1"},
            {"uid": tasks[0]["uid"], "task_id": "1", "score": 0.8, "group_id": "dup_g1"},
        ]

        # Also push something for task 2 and alpha to complete cold start
        self.client.push_results([
            {"uid": tasks[1]["uid"], "task_id": "2", "score": 0.5, "group_id": "g2"},
        ])
        self.client.push_results([
            {"uid": tasks[2]["uid"], "task_id": "alpha", "score": 0.5, "group_id": "g3"},
        ])

        # First push
        r1 = self.client.push_results(batch)
        assert r1["n_accepted"] == 2

        stats1 = self.client.stats()

        # Second push — exact same batch
        r2 = self.client.push_results(batch)
        assert r2["n_accepted"] == 0  # all duplicates

        stats2 = self.client.stats()
        # Stats unchanged
        assert stats1["tasks"]["1"]["n_trajectories"] == stats2["tasks"]["1"]["n_trajectories"]


# ---------------------------------------------------------------------------
# Test 9: TaskManagerDataset integration
# ---------------------------------------------------------------------------


class TestDatasetIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(mock_dataset_3, self.port, strategy="sequential")
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_dataset_with_dataloader(self):
        from torch.utils.data import DataLoader

        from seta_env.datahubs.task_dataset import TaskManagerDataset

        dataset = TaskManagerDataset(self.client)
        # Can't use standard DataLoader with num_workers>0 easily for this,
        # but batch_size=1 with manual iteration works
        # Pull 2 items manually
        it = iter(dataset)
        item1 = next(it)
        item2 = next(it)

        for item in [item1, item2]:
            assert "task_id" in item
            assert "task_path" in item
            assert "instruction" in item
            assert "uid" in item

        assert item1["task_id"] == "1"
        assert item2["task_id"] == "2"


# ---------------------------------------------------------------------------
# Test 10: Empty category handling
# ---------------------------------------------------------------------------


class TestEmptyCategories:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(
            mock_dataset_3,
            self.port,
            strategy="weighted",
            extra_args=["--window-size", "2"],
        )
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_single_nonempty_category(self):
        # Drain cold start, all mastered
        tasks = [self.client.pull_task() for _ in range(3)]
        for t in tasks:
            self.client.push_results([
                {"uid": t["uid"], "task_id": t["task_id"], "score": 1.0, "group_id": f"g_{t['task_id']}"},
            ])

        stats = self.client.stats()
        assert stats["phase"] == "weighted_phase2"
        assert stats["categories"]["mastered"]["count"] == 3

        # Pull still works from the single non-empty category
        task = self.client.pull_task()
        assert task["task_id"] in ["1", "2", "alpha"]

    def test_all_broken_returns_503(self):
        # Drain cold start
        tasks = [self.client.pull_task() for _ in range(3)]

        # Push all-None twice for each task to make them broken
        for t in tasks:
            self.client.push_results([
                {"uid": t["uid"], "task_id": t["task_id"], "score": None, "group_id": f"n1_{t['task_id']}"},
            ])
            self.client.push_results([
                {"uid": t["uid"], "task_id": t["task_id"], "score": None, "group_id": f"n2_{t['task_id']}"},
            ])

        stats = self.client.stats()
        assert stats["categories"]["broken"]["count"] == 3

        # Pull should raise (503)
        with pytest.raises(requests.HTTPError) as exc_info:
            self.client.pull_task()
        assert exc_info.value.response.status_code == 503


# ---------------------------------------------------------------------------
# Test 11: Monitoring metrics
# ---------------------------------------------------------------------------


class TestMonitoringMetrics:
    @pytest.fixture(autouse=True)
    def setup(self, mock_dataset_3):
        self.port = _find_free_port()
        self.proc = _start_server(mock_dataset_3, self.port, strategy="weighted")
        self.client = TaskManagerClient(f"http://127.0.0.1:{self.port}")
        self.client.cleanup()
        yield
        _stop_server(self.proc)

    def test_metrics_increment(self):
        stats0 = self.client.stats()
        assert stats0["metrics"]["total_pulls"] == 0
        assert stats0["metrics"]["total_pushes"] == 0

        # Pull 3 tasks
        tasks = [self.client.pull_task() for _ in range(3)]

        stats1 = self.client.stats()
        assert stats1["metrics"]["total_pulls"] == 3

        # Push results
        for t in tasks:
            self.client.push_results([
                {"uid": t["uid"], "task_id": t["task_id"], "score": 0.5, "group_id": f"g_{t['task_id']}"},
            ])

        stats2 = self.client.stats()
        assert stats2["metrics"]["total_pushes"] == 3
        assert stats2["metrics"]["total_trajectories"] == 3
        assert stats2["metrics"]["total_errors"] == 0
        assert stats2["metrics"]["uptime_seconds"] > 0
        assert "category_history" in stats2
        assert "recent_transitions" in stats2
