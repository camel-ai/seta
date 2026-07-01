import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFolder


SOURCE_REPO = "camel-ai/seta-env-seed2synth-synth"
TARGET_REPO = "camel-ai/seta-env-v2"

# Download tasks in batches to avoid HuggingFace 429 rate limits
BATCH_SIZE = 50
BATCH_PAUSE_SECS = 10
MAX_WORKERS = 4


def merge_sources(src_dir: Path, out_dir: Path) -> None:
    """Merge task folders across sources, prefixing task_id with source name.

    Input layout:
        src_dir/<source>/<task_id>/...harbor task files...

    Output layout:
        out_dir/<source>__<task_id>/...harbor task files...
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for source_dir in sorted(src_dir.iterdir()):
        if not source_dir.is_dir() or source_dir.name.startswith("."):
            continue
        source_name = source_dir.name

        for task_dir in sorted(source_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("."):
                continue
            task_id = task_dir.name
            merged_name = f"{source_name}__{task_id}"
            dest = out_dir / merged_name

            shutil.move(str(task_dir), dest)
            print(f"  {source_name}/{task_id} -> {merged_name}")


def main():
    api = HfApi()

    # Step 1: List task folders already present in target
    print(f"Listing existing tasks in {TARGET_REPO}...")
    existing = {
        e.path
        for e in api.list_repo_tree(
            repo_id=TARGET_REPO, repo_type="dataset", recursive=False
        )
        if isinstance(e, RepoFolder)
    }
    print(f"  {len(existing)} tasks already in target")

    # Step 2: List (source, task_id) pairs in source repo
    print(f"Listing tasks in {SOURCE_REPO}...")
    sources = [
        e.path
        for e in api.list_repo_tree(
            repo_id=SOURCE_REPO, repo_type="dataset", recursive=False
        )
        if isinstance(e, RepoFolder)
    ]
    src_pairs: list[tuple[str, str]] = []
    for source in sources:
        for e in api.list_repo_tree(
            repo_id=SOURCE_REPO,
            repo_type="dataset",
            path_in_repo=source,
            recursive=False,
        ):
            if isinstance(e, RepoFolder):
                src_pairs.append((source, e.path.split("/")[-1]))
    print(f"  {len(src_pairs)} tasks in source")

    # Step 3: Diff
    missing = [(s, t) for (s, t) in src_pairs if f"{s}__{t}" not in existing]
    print(f"{len(missing)} new tasks to sync")
    if not missing:
        print(f"Already up to date. https://huggingface.co/datasets/{TARGET_REPO}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_dir = tmp / "source"
        out_dir = tmp / "merged"

        # Step 4: Batch download — download in chunks to avoid 429 rate limits
        batches = [
            missing[i : i + BATCH_SIZE]
            for i in range(0, len(missing), BATCH_SIZE)
        ]
        print(
            f"Downloading {len(missing)} task folders from {SOURCE_REPO} "
            f"in {len(batches)} batches of up to {BATCH_SIZE}..."
        )
        for batch_idx, batch in enumerate(batches):
            allow_patterns = [f"{s}/{t}/**" for (s, t) in batch]
            print(
                f"  Batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch)} tasks)..."
            )
            snapshot_download(
                repo_id=SOURCE_REPO,
                repo_type="dataset",
                local_dir=str(src_dir),
                allow_patterns=allow_patterns,
                max_workers=MAX_WORKERS,
            )
            if batch_idx < len(batches) - 1:
                print(f"  Pausing {BATCH_PAUSE_SECS}s before next batch...")
                time.sleep(BATCH_PAUSE_SECS)

        # Step 5: Merge (rename <source>/<task_id> -> <source>__<task_id>)
        print("Merging sources...")
        merge_sources(src_dir, out_dir)

        # Step 6: Upload (upload_large_folder handles many files with
        # parallel workers, resumability, and multi-commit batching)
        print(f"Uploading {len(missing)} new tasks to {TARGET_REPO}...")
        api.upload_large_folder(
            folder_path=str(out_dir),
            repo_id=TARGET_REPO,
            repo_type="dataset",
        )

    print(f"Done. https://huggingface.co/datasets/{TARGET_REPO}")


if __name__ == "__main__":
    main()
