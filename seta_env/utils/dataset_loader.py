"""Load a harbor-format dataset folder into a HuggingFace Dataset.

Each subdirectory of the dataset folder must be a valid harbor task, i.e.
it must contain at minimum:
    task.toml       — task config (metadata, environment, verifier settings)
    instruction.md  — the natural-language task instruction

The returned Dataset has three columns consumed by TerminalEnvironment / GRPORollout:
    task_name   str  — the subdirectory name (e.g. "0", "42", "install-packages")
    task_path   str  — absolute path to the task directory
    instruction str  — contents of instruction.md

Usage:
    from seta_env.utils.dataset_loader import load_harbor_dataset
    ds = load_harbor_dataset("dataset/seta-env-harbor")
    # use directly or wrap with create_dataloader
"""

from pathlib import Path

from datasets import Dataset
from harbor.models.task.task import Task


def load_harbor_dataset(dataset_path: str | Path) -> Dataset:
    """Load all tasks from a harbor dataset folder.

    Directories that are not valid harbor tasks (missing task.toml /
    instruction.md) are skipped with a warning rather than raising.

    Args:
        dataset_path: Path to the dataset folder containing one subdirectory
                      per task.

    Returns:
        A HuggingFace ``Dataset`` with columns ``task_name``, ``task_path``,
        and ``instruction``, sorted by directory name.
    """
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    records = []
    # Sort numerically where possible, then lexicographically for mixed names.
    def _sort_key(p: Path):
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)

    for task_dir in sorted(dataset_path.iterdir(), key=_sort_key):
        if not task_dir.is_dir():
            continue
        try:
            task = Task(task_dir)
            records.append({
                "task_name": task.name,
                "task_path": str(task_dir),
                "instruction": task.instruction.strip(),
            })
        except Exception as e:
            print(f"[dataset_loader] skipping {task_dir.name}: {e}")

    print(f"[dataset_loader] loaded {len(records)} tasks from {dataset_path}")
    return Dataset.from_list(records)
