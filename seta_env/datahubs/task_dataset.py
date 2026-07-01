"""TaskManagerDataset — PyTorch IterableDataset wrapping TaskManagerClient."""

from __future__ import annotations

from torch.utils.data import IterableDataset

from seta_env.datahubs.task_manager_client import TaskManagerClient


class TaskManagerDataset(IterableDataset):
    """An ``IterableDataset`` where each ``__next__`` blocks on ``pull_task()``.

    Usage::

        client = TaskManagerClient("http://localhost:8765")
        dataset = TaskManagerDataset(client)
        dataloader = StatefulDataLoader(dataset, batch_size=N)
    """

    def __init__(self, client: TaskManagerClient) -> None:
        super().__init__()
        self.client = client

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """Pull the next task from the service.

        Returns:
            ``{"task_id": str, "task_path": str, "instruction": str, "uid": str}``
        """
        return self.client.pull_task()
