"""HuggingFace integration utilities for the evolution pipeline."""

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional

from evol_config import Config
from io_utils import list_input_tasks, is_harbor_task_complete, read_synth_info

logger = logging.getLogger(__name__)


def _get_hf_token(config: Config) -> Optional[str]:
    return os.environ.get(config.huggingface.token_env)


def download_input_task(task_id: str, input_dir: str, config: Config) -> bool:
    """Download a single task folder from input_repo on demand."""
    task_dir = Path(input_dir) / task_id
    if task_dir.exists() and is_harbor_task_complete(str(task_dir)):
        return True

    repo = config.huggingface.input_repo
    if not repo:
        return False

    token = _get_hf_token(config)
    task_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo, repo_type="dataset",
            allow_patterns=[f"{task_id}/**"],
            local_dir=str(input_dir), token=token,
        )
        logger.info("Downloaded %s from %s", task_id, repo)
        return task_dir.exists()
    except Exception as e:
        logger.error("Failed to download %s: %s", task_id, e)
        return False


def upload_output_data(config: Config, dry_run: bool = False) -> int:
    """Upload PASS variants from all evolve round output_dirs to output_repo."""
    repo = config.huggingface.output_repo
    if not repo:
        logger.info("No output_repo configured; skipping upload.")
        return 0

    token = _get_hf_token(config)
    total = 0

    for rnd in config.evolve.rounds:
        output_dir = rnd.output_dir
        base = Path(output_dir)
        if not base.is_dir():
            continue

        pass_dirs = []
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            info = read_synth_info(output_dir, entry.name)
            if info and info.get("status") == "done" and info.get("verdict") == "PASS":
                if is_harbor_task_complete(str(entry)):
                    pass_dirs.append(entry.name)

        if not pass_dirs:
            continue

        logger.info("%d PASS variant(s) in %s for upload to %s", len(pass_dirs), output_dir, repo)
        if dry_run:
            for d in pass_dirs:
                print(f"  [dry-run] Would upload: {d}")
            total += len(pass_dirs)
            continue

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.upload_large_folder(
                folder_path=str(base), repo_id=repo, repo_type="dataset",
                allow_patterns=[f"{d}/**" for d in pass_dirs],
            )
            total += len(pass_dirs)
        except Exception as e:
            logger.error("Upload failed for %s: %s", output_dir, e)

    return total


def generate_filter_csvs(input_dir: str, n_parts: int, output_dir: str) -> List[str]:
    """Partition task IDs from input_dir into N filter CSVs."""
    tasks = list_input_tasks(input_dir)
    if not tasks:
        logger.warning("No tasks found in %s", input_dir)
        return []

    os.makedirs(output_dir, exist_ok=True)
    import random
    rng = random.Random(42)
    tasks_shuffled = list(tasks)
    rng.shuffle(tasks_shuffled)

    chunk_size = max(1, len(tasks_shuffled) // n_parts)
    paths: List[str] = []
    for i in range(n_parts):
        start = i * chunk_size
        chunk = tasks_shuffled[start:] if i == n_parts - 1 else tasks_shuffled[start:start + chunk_size]
        csv_path = os.path.join(output_dir, f"part_{i + 1:02d}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id"])
            for tid in sorted(chunk):
                writer.writerow([tid])
        paths.append(csv_path)
    return paths
