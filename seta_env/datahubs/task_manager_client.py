"""TaskManagerClient — synchronous HTTP client for TaskManagerService."""

from __future__ import annotations

import requests


class TaskManagerClient:
    """Synchronous HTTP client wrapping the TaskManager service endpoints.

    All methods block until the server responds and raise on HTTP errors.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def cleanup(self) -> dict:
        """POST /cleanup. Returns ``{"status": "ok", "n_tasks": int}``."""
        resp = self.session.post(f"{self.base_url}/cleanup")
        resp.raise_for_status()
        return resp.json()

    def initialize(self, scores_csv: str) -> dict:
        """POST /initialize with a CSV path for warm-start.

        Args:
            scores_csv: path to CSV file with prior eval results.

        Returns:
            ``{"status": "ok", "n_initialized": int, ...}``
        """
        resp = self.session.post(
            f"{self.base_url}/initialize",
            json={"scores_csv": scores_csv},
        )
        resp.raise_for_status()
        return resp.json()

    def pull_task(self) -> dict:
        """GET /pull_task.

        Returns:
            ``{"task_id": str, "task_path": str, "instruction": str, "uid": str}``
        """
        resp = self.session.get(f"{self.base_url}/pull_task")
        resp.raise_for_status()
        return resp.json()

    def push_results(self, results: list[dict]) -> dict:
        """POST /push_results.

        Args:
            results: list of dicts with keys ``uid``, ``task_id``, ``score``, ``group_id``.

        Returns:
            ``{"status": "ok", "n_accepted": int}``
        """
        resp = self.session.post(f"{self.base_url}/push_results", json=results)
        resp.raise_for_status()
        return resp.json()

    def stats(self) -> dict:
        """GET /stats. Returns the full stats payload."""
        resp = self.session.get(f"{self.base_url}/stats")
        resp.raise_for_status()
        return resp.json()
