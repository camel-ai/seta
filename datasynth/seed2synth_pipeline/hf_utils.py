#!/usr/bin/env python3
"""HuggingFace utilities for downloading seed data, uploading synth results, and splitting tasks.

Functions:
- download_metadata():    download metadata.csv for specified sources
- download_seed_task():   download a single seed task folder
- upload_synth_data():    upload done+PASS synth tasks to HF (resumable, verifies after)
- verify_synth_data():    compare local done+PASS tasks vs remote, file by file
- check_seed_vs_synth():  compare seed vs synth on HF, generate filter CSVs

Can be used as a library or as a CLI script.

────────────────────────────────────────────────────────────────────────────
CLI usage examples
────────────────────────────────────────────────────────────────────────────

  # Download metadata.csv for every configured source
  python hf_utils.py --config configs/kaggle_base.yaml download-metadata

  # Download metadata for a specific source
  python hf_utils.py --config configs/kaggle_base.yaml download-metadata \\
      --sources kaggle_notebook

  # Download a single seed task folder
  python hf_utils.py --config configs/kaggle_base.yaml download-task \\
      --source kaggle_notebook --task-id kanncaa1_data-sciencetutorial-for-beginners

  # Dry-run: see which done+PASS tasks would be uploaded for one source
  python hf_utils.py --config configs/kaggle_base.yaml upload-synth \\
      --sources kaggle_notebook --dry-run

  # Real upload (resumable, chunked, verifies after upload completes)
  python hf_utils.py --config configs/kaggle_base.yaml upload-synth \\
      --sources kaggle_notebook

  # Just verify what's on the remote against local done+PASS tasks
  # (file-by-file size + presence check, no uploading)
  python hf_utils.py --config configs/kaggle_base.yaml verify-synth \\
      --sources kaggle_notebook

  # Compare seed vs synth on HF and split pending tasks into N filter CSVs
  python hf_utils.py --config configs/kaggle_base.yaml check-and-split \\
      --sources kaggle_notebook --n-parts 4 --output-dir filters/

Environment:
  HF_TOKEN (or whatever ``config.huggingface.token_env`` points to) must be
  set for any operation that talks to HuggingFace.

Notes:
  - upload-synth uses HfApi.upload_large_folder under the hood: resumable,
    parallel, automatic commit batching (stays well under the HF 128/hour
    commits-per-repo limit), and retries transient errors.
  - upload-synth never overwrites: any task folder already present on the
    remote is skipped.
  - upload-synth runs verify_synth_data() after the upload completes, so
    interrupted/timed-out runs that actually succeeded server-side are
    still reported correctly (no false "FAILED" output).
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Set

try:
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Install with: pip install huggingface-hub")
    sys.exit(1)

from seed2synth_config import Config

logger = logging.getLogger(__name__)


def download_metadata(
    config: Config,
    sources: Optional[List[str]] = None,
    skip_existing: bool = True,
) -> Dict[str, Path]:
    """Download metadata.csv for specified sources from HF seed repo.

    Args:
        config: Config object
        sources: list of source names (default: config.sources)
        skip_existing: if True, skip sources where metadata.csv already exists locally

    Returns:
        dict mapping source_name → local metadata.csv path
    """
    if sources is None:
        sources = config.sources

    hf_token = os.environ.get(config.huggingface.token_env)
    seed_data_dir = Path(config.paths.seed_data_dir)

    results = {}

    for source in sources:
        metadata_local = seed_data_dir / source / "metadata.csv"
        metadata_local.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and metadata_local.exists():
            logger.info(f"[{source}] metadata.csv already exists locally, skipping download")
            results[source] = metadata_local
            continue

        try:
            logger.info(f"[{source}] downloading metadata.csv from HF...")
            hf_hub_download(
                repo_id=config.huggingface.seed_repo,
                filename=f"{source}/metadata.csv",
                local_dir=str(seed_data_dir),
                repo_type="dataset",
                token=hf_token,
            )
            logger.info(f"[{source}] ✓ metadata.csv downloaded")
            results[source] = metadata_local
        except Exception as e:
            logger.error(f"[{source}] failed to download metadata.csv: {e}")

    return results


def download_seed_task(
    source: str,
    task_id: str,
    seed_data_dir: Path,
    config: Config,
) -> bool:
    """Download a single seed task folder from HF.

    Args:
        source: source name
        task_id: task ID (folder name)
        seed_data_dir: local seed_data root directory
        config: Config object

    Returns:
        True if successful, False otherwise
    """
    task_dir = seed_data_dir / source / task_id
    if task_dir.exists() and list(task_dir.glob("*")):
        # Already downloaded
        return True

    hf_token = os.environ.get(config.huggingface.token_env)
    task_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.debug(f"Downloading {source}/{task_id} from HF...")
        snapshot_download(
            repo_id=config.huggingface.seed_repo,
            repo_type="dataset",
            allow_patterns=[f"{source}/{task_id}/**"],
            local_dir=str(seed_data_dir),
            token=hf_token,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to download {source}/{task_id}: {e}")
        return False


def upload_synth_data(
    config: Config,
    synth_data_dir: Optional[Path] = None,
    dry_run: bool = False,
    sources: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Upload done+PASS synth tasks to HF synth repo, skipping any already present.

    Only uploads tasks whose synth_info.json has status=done and verdict=PASS
    (case-insensitive). Tasks whose folder already exists on the remote synth
    repo are skipped — never overwritten.

    Args:
        config: Config object
        synth_data_dir: synth_data root (default: config.paths.synth_data_dir)
        dry_run: if True, show what would be uploaded without doing it
        sources: optional restriction to specific source subdirs

    Returns:
        dict mapping source_name → list of uploaded task_ids
    """
    if synth_data_dir is None:
        synth_data_dir = Path(config.paths.synth_data_dir)

    hf_token = os.environ.get(config.huggingface.token_env)
    if not hf_token and not dry_run:
        logger.warning(f"{config.huggingface.token_env} not set, cannot upload to HF")
        return {}

    hf_api = HfApi(token=hf_token)
    repo_id = config.huggingface.synth_repo

    uploaded: Dict[str, List[str]] = {}

    for source_dir in sorted(synth_data_dir.iterdir()):
        if not source_dir.is_dir() or source_dir.name.startswith("."):
            continue

        source = source_dir.name
        if sources is not None and source not in sources:
            continue

        uploaded[source] = []

        # 1. Collect local done+PASS task ids
        local_pass_tasks: List[str] = []
        for task_dir in sorted(source_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            synth_info_path = task_dir / "synth_info.json"
            if not synth_info_path.exists():
                continue
            try:
                with open(synth_info_path) as f:
                    synth_info = json.load(f)
            except Exception as e:
                logger.error(f"Error reading {synth_info_path}: {e}")
                continue

            status = str(synth_info.get("status", "")).lower()
            verdict = str(synth_info.get("verdict", "")).lower()
            if status == "done" and verdict == "pass":
                local_pass_tasks.append(task_dir.name)

        if not local_pass_tasks:
            logger.info(f"[{source}] no done+PASS tasks locally, skipping")
            continue

        # 2. List task folders that already exist on the remote (one call)
        remote_existing: Set[str] = set()
        try:
            tree = hf_api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=source,
                recursive=False,
            )
            for item in tree:
                # directories only — task folders
                if getattr(item, "type", None) == "directory" or item.__class__.__name__ == "RepoFolder":
                    remote_existing.add(Path(item.path).name)
        except Exception as e:
            # Path may not exist yet on remote — that's fine, nothing to skip
            logger.debug(f"[{source}] could not list remote tree ({e}); assuming empty")

        logger.info(
            f"[{source}] {len(local_pass_tasks)} local done+PASS, "
            f"{len(remote_existing)} already on remote"
        )

        # 3. Compute pending = local PASS - already on remote
        pending = [tid for tid in local_pass_tasks if tid not in remote_existing]
        skipped = len(local_pass_tasks) - len(pending)
        if skipped:
            logger.info(f"[{source}] skipping {skipped} tasks already on remote")

        if not pending:
            logger.info(f"[{source}] nothing to upload")
            continue

        if dry_run:
            for task_id in pending:
                logger.info(
                    f"[DRY-RUN] would upload {source_dir / task_id} → "
                    f"{repo_id}:{source}/{task_id}"
                )
            uploaded[source] = list(pending)
            continue

        # 4. Upload all pending tasks via upload_large_folder.
        #    This is HF's recommended path for big/many-file uploads:
        #      - resumable: re-runs continue where they left off
        #      - chunked + parallel: tolerates per-file timeouts
        #      - automatic commit batching (well under the 128/hr limit)
        #      - retries transient errors internally
        #
        #    upload_large_folder has no `path_in_repo` arg — it mirrors
        #    `folder_path` at the repo root. To preserve the on-remote layout
        #    `<source>/<task>/...`, we point folder_path at synth_data_dir
        #    (the parent of source_dir) and scope allow_patterns to
        #    `<source>/<task>/**` for each pending task.
        allow_patterns = [f"{source}/{tid}/**" for tid in pending]
        upload_exc: Optional[BaseException] = None
        try:
            logger.info(
                f"[{source}] uploading {len(pending)} pending tasks via "
                f"upload_large_folder (resumable, chunked)..."
            )
            hf_api.upload_large_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(synth_data_dir),
                allow_patterns=allow_patterns,
                print_report=True,
            )
            logger.info(f"[{source}] upload_large_folder returned cleanly")
        except Exception as e:
            # The HF client may raise on read-timeouts even when the server
            # actually accepted the commit. Don't trust the exception alone —
            # let the verification step below decide what really landed.
            upload_exc = e
            logger.warning(
                f"[{source}] upload_large_folder raised: {e}. "
                f"Will verify against remote before reporting failure."
            )

        # 5. Post-upload verification: file-by-file check vs remote.
        logger.info(f"[{source}] verifying remote contents...")
        verify_result = _verify_source(
            hf_api=hf_api,
            repo_id=repo_id,
            source=source,
            source_dir=source_dir,
            task_ids=local_pass_tasks,
        )
        ok_tasks = verify_result["ok"]
        bad_tasks = verify_result["incomplete"] + verify_result["missing"]
        uploaded[source] = sorted(set(ok_tasks) & set(pending))

        if bad_tasks:
            logger.error(
                f"[{source}] verification found {len(bad_tasks)} "
                f"incomplete/missing tasks on remote:"
            )
            for t in bad_tasks[:20]:
                logger.error(f"    - {t}")
            if len(bad_tasks) > 20:
                logger.error(f"    ... and {len(bad_tasks) - 20} more")
        else:
            logger.info(
                f"[{source}] ✓ verification passed: all "
                f"{len(local_pass_tasks)} done+PASS tasks present and complete"
            )
            if upload_exc is not None:
                logger.info(
                    f"[{source}] (the earlier upload exception was a false "
                    f"alarm — server-side commit landed)"
                )

    # Summary
    print("\n" + "=" * 70)
    print("UPLOAD SUMMARY" + (" (DRY-RUN)" if dry_run else ""))
    print("=" * 70)
    for src, tids in uploaded.items():
        print(f"  {src}: {len(tids)} tasks uploaded (verified)")
    print("=" * 70 + "\n")

    return uploaded


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _verify_source(
    hf_api: "HfApi",
    repo_id: str,
    source: str,
    source_dir: Path,
    task_ids: List[str],
) -> Dict[str, List[str]]:
    """Compare local task dirs against the remote ``<source>/`` tree.

    For each task in ``task_ids``, walks the local task dir, lists every
    regular file, and checks that the same relative path exists on the remote
    and (where the remote reports a size) that the file sizes match.

    Returns a dict with three sorted lists:
        {
            "ok":         tasks fully present and matching,
            "incomplete": tasks present but with missing files / size mismatches,
            "missing":    tasks with no folder on the remote at all,
        }
    """
    from collections import defaultdict

    # 1. List the full remote tree under <source>/ in one recursive call.
    remote_files: Dict[str, Optional[int]] = {}
    try:
        for item in hf_api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=source,
            recursive=True,
        ):
            if item.__class__.__name__ == "RepoFile":
                remote_files[item.path] = getattr(item, "size", None)
    except Exception as e:
        logger.warning(
            f"[{source}] could not list remote tree for verification ({e}); "
            f"treating remote as empty"
        )

    # 2. Bucket remote files by task id (skipping anything at the source root,
    #    e.g. summary.csv).
    remote_by_task: Dict[str, Dict[str, Optional[int]]] = defaultdict(dict)
    prefix = f"{source}/"
    for path, size in remote_files.items():
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):]
        if "/" not in rest:
            continue  # file directly under <source>/, not part of any task
        task_id, rel = rest.split("/", 1)
        remote_by_task[task_id][rel] = size

    ok: List[str] = []
    incomplete: List[str] = []
    missing: List[str] = []

    for task in task_ids:
        if task not in remote_by_task:
            missing.append(task)
            continue

        local_files: Dict[str, int] = {}
        tdir = source_dir / task
        for p in tdir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(tdir).as_posix()
                local_files[rel] = p.stat().st_size

        rem = remote_by_task[task]
        missing_files = [rel for rel in local_files if rel not in rem]
        size_mismatches = [
            rel for rel, lsize in local_files.items()
            if rem.get(rel) is not None and rem[rel] != lsize
        ]

        if missing_files or size_mismatches:
            incomplete.append(task)
            for rel in missing_files[:3]:
                logger.debug(f"[{source}/{task}] missing on remote: {rel}")
            for rel in size_mismatches[:3]:
                logger.debug(
                    f"[{source}/{task}] size mismatch: {rel} "
                    f"local={local_files[rel]} remote={rem[rel]}"
                )
        else:
            ok.append(task)

    return {"ok": sorted(ok), "incomplete": sorted(incomplete), "missing": sorted(missing)}


def verify_synth_data(
    config: Config,
    synth_data_dir: Optional[Path] = None,
    sources: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Verify that every local done+PASS task is fully present on the remote.

    Walks each local task dir and compares its file set (and per-file size,
    where the remote reports it) against the remote ``<source>/`` tree on
    ``config.huggingface.synth_repo``.

    Args:
        config: Config object
        synth_data_dir: synth_data root (default: config.paths.synth_data_dir)
        sources: optional restriction to specific source subdirs

    Returns:
        ``{source: {"ok": [...], "incomplete": [...], "missing": [...]}}``
        Exit-status-friendly: callers can treat any non-empty
        ``incomplete``/``missing`` as a failure.
    """
    if synth_data_dir is None:
        synth_data_dir = Path(config.paths.synth_data_dir)

    hf_token = os.environ.get(config.huggingface.token_env)
    hf_api = HfApi(token=hf_token)
    repo_id = config.huggingface.synth_repo

    report: Dict[str, Dict[str, List[str]]] = {}

    for source_dir in sorted(synth_data_dir.iterdir()):
        if not source_dir.is_dir() or source_dir.name.startswith("."):
            continue
        source = source_dir.name
        if sources is not None and source not in sources:
            continue

        # Collect local done+PASS tasks for this source
        local_pass: List[str] = []
        for task_dir in sorted(source_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            si = task_dir / "synth_info.json"
            if not si.exists():
                continue
            try:
                info = json.loads(si.read_text())
            except Exception as e:
                logger.error(f"Error reading {si}: {e}")
                continue
            status = str(info.get("status", "")).lower()
            verdict = str(info.get("verdict", "")).lower()
            if status == "done" and verdict == "pass":
                local_pass.append(task_dir.name)

        if not local_pass:
            logger.info(f"[{source}] no local done+PASS tasks; nothing to verify")
            report[source] = {"ok": [], "incomplete": [], "missing": []}
            continue

        logger.info(
            f"[{source}] verifying {len(local_pass)} local done+PASS tasks "
            f"against {repo_id}..."
        )
        result = _verify_source(
            hf_api=hf_api,
            repo_id=repo_id,
            source=source,
            source_dir=source_dir,
            task_ids=local_pass,
        )
        report[source] = result

    # Print report
    print("\n" + "=" * 70)
    print("VERIFICATION REPORT")
    print("=" * 70)
    any_bad = False
    for source, result in report.items():
        n_ok = len(result["ok"])
        n_inc = len(result["incomplete"])
        n_miss = len(result["missing"])
        total = n_ok + n_inc + n_miss
        flag = "✓" if (n_inc + n_miss) == 0 else "✗"
        print(f"  {flag} {source}: {n_ok}/{total} ok, "
              f"{n_inc} incomplete, {n_miss} missing")
        if n_inc:
            for t in result["incomplete"][:10]:
                print(f"      INCOMPLETE: {t}")
            if n_inc > 10:
                print(f"      ... and {n_inc - 10} more")
        if n_miss:
            for t in result["missing"][:10]:
                print(f"      MISSING:    {t}")
            if n_miss > 10:
                print(f"      ... and {n_miss - 10} more")
        if n_inc or n_miss:
            any_bad = True
    print("=" * 70)
    print(("FAIL: some tasks are missing or incomplete on remote"
           if any_bad else "PASS: every local done+PASS task is fully on remote"))
    print("=" * 70 + "\n")

    return report


def check_seed_vs_synth(
    config: Config,
    sources: Optional[List[str]] = None,
    n_parts: int = 1,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Compare seed vs synth on HF, identify pending tasks, and generate filter CSVs.

    Args:
        config: Config object
        sources: list of sources to check (default: config.sources)
        n_parts: split pending tasks into N equal parts
        output_dir: where to write filter CSVs (default: filters/)

    Returns:
        list of paths to generated filter CSVs
    """
    if sources is None:
        sources = config.sources

    if output_dir is None:
        output_dir = Path("filters")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata CSVs for all sources (cached)
    download_metadata(config, sources=sources, skip_existing=True)

    seed_data_dir = Path(config.paths.seed_data_dir)
    hf_api = HfApi()
    hf_token = os.environ.get(config.huggingface.token_env)

    all_pending = {}  # source → list of task_ids

    for source in sources:
        # Read local metadata.csv to get all seed tasks
        metadata_path = seed_data_dir / source / "metadata.csv"
        if not metadata_path.exists():
            logger.warning(f"[{source}] metadata.csv not found, skipping")
            continue

        all_seed_tasks = set()
        try:
            with open(metadata_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("filtered", "").lower() != "true":
                        all_seed_tasks.add(row.get("task_id", "").strip())
        except Exception as e:
            logger.error(f"[{source}] error reading metadata.csv: {e}")
            continue

        logger.info(f"[{source}] found {len(all_seed_tasks)} non-filtered seed tasks")

        # Check which tasks are already done+pass on HF synth repo
        done_tasks = set()
        try:
            logger.info(f"[{source}] checking HF synth repo for done tasks...")
            # List all task folders under {source}/ on HF synth repo
            tree = hf_api.list_repo_tree(
                repo_id=config.huggingface.synth_repo,
                repo_type="dataset",
                path_in_repo=source,
                token=hf_token,
            )

            for item in tree:
                if hasattr(item, 'name') and item.name != "summary.csv":
                    task_id = item.name
                    synth_info_hf = item / "synth_info.json"  # synthetic path
                    # Try to check if task exists on HF
                    try:
                        hf_hub_download(
                            repo_id=config.huggingface.synth_repo,
                            filename=f"{source}/{task_id}/synth_info.json",
                            repo_type="dataset",
                            token=hf_token,
                        )
                        # If we can download it, consider it done (actual verdict check would happen here)
                        done_tasks.add(task_id)
                    except:
                        pass  # Task not yet synth'ed on HF
        except Exception as e:
            logger.debug(f"[{source}] error checking HF synth repo: {e}")

        if done_tasks:
            logger.info(f"[{source}] found {len(done_tasks)} already-done tasks on HF")

        # Compute pending = seed - done - filtered
        pending = sorted(all_seed_tasks - done_tasks)
        logger.info(f"[{source}] {len(pending)} tasks pending")

        all_pending[source] = pending

    # Split pending tasks into N parts and write filter CSVs
    generated_filters = []

    total_pending = sum(len(v) for v in all_pending.values())
    if total_pending == 0:
        logger.warning("No pending tasks to split")
        return generated_filters

    tasks_per_part = max(1, total_pending // n_parts)

    part_idx = 1
    current_part_tasks = []

    for source in sources:
        for task_id in all_pending.get(source, []):
            current_part_tasks.append((source, task_id))

            if len(current_part_tasks) >= tasks_per_part and part_idx < n_parts:
                # Write current part
                filter_path = output_dir / f"part_{part_idx:02d}.csv"
                with open(filter_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["source", "task_id"])
                    writer.writeheader()
                    for src, tid in current_part_tasks:
                        writer.writerow({"source": src, "task_id": tid})

                logger.info(f"Wrote {len(current_part_tasks)} tasks to {filter_path}")
                generated_filters.append(filter_path)

                current_part_tasks = []
                part_idx += 1

    # Write final part with remaining tasks
    if current_part_tasks:
        filter_path = output_dir / f"part_{part_idx:02d}.csv"
        with open(filter_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["source", "task_id"])
            writer.writeheader()
            for src, tid in current_part_tasks:
                writer.writerow({"source": src, "task_id": tid})

        logger.info(f"Wrote {len(current_part_tasks)} tasks to {filter_path}")
        generated_filters.append(filter_path)

    print("\n" + "=" * 70)
    print(f"SPLIT SUMMARY: {total_pending} pending tasks → {len(generated_filters)} parts")
    print("=" * 70)
    for i, filter_path in enumerate(generated_filters, 1):
        with open(filter_path) as f:
            lines = len(f.readlines()) - 1  # exclude header
        print(f"  part_{i:02d}.csv: {lines} tasks")
    print("=" * 70 + "\n")

    return generated_filters


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="HuggingFace utilities for Seed2Synth")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # download-metadata
    sub = subparsers.add_parser("download-metadata", help="Download metadata.csv for sources")
    sub.add_argument("--sources", help="Comma-separated source names (default: all in config)")

    # download-task
    sub = subparsers.add_parser("download-task", help="Download a single seed task")
    sub.add_argument("--source", required=True)
    sub.add_argument("--task-id", required=True)

    # upload-synth
    sub = subparsers.add_parser("upload-synth", help="Upload done+PASS synth tasks to HF")
    sub.add_argument("--dry-run", action="store_true")
    sub.add_argument("--sources", help="Comma-separated source names (default: all subdirs)")

    # verify-synth
    sub = subparsers.add_parser(
        "verify-synth",
        help="Verify local done+PASS tasks vs the remote synth repo (no upload)",
    )
    sub.add_argument("--sources", help="Comma-separated source names (default: all subdirs)")

    # check-and-split
    sub = subparsers.add_parser("check-and-split", help="Compare seed vs synth, generate filter CSVs")
    sub.add_argument("--sources", help="Comma-separated source names (default: all in config)")
    sub.add_argument("--n-parts", type=int, default=4, help="Split into N parts (default: 4)")
    sub.add_argument("--output-dir", default="filters", help="Output directory for filter CSVs")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    try:
        config = Config.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Route to command handler
    if args.command == "download-metadata":
        sources = None
        if args.sources:
            sources = [s.strip() for s in args.sources.split(",")]
        download_metadata(config, sources=sources)

    elif args.command == "download-task":
        download_seed_task(args.source, args.task_id, Path(config.paths.seed_data_dir), config)

    elif args.command == "upload-synth":
        sources = None
        if args.sources:
            sources = [s.strip() for s in args.sources.split(",")]
        upload_synth_data(config, dry_run=args.dry_run, sources=sources)

    elif args.command == "verify-synth":
        sources = None
        if args.sources:
            sources = [s.strip() for s in args.sources.split(",")]
        report = verify_synth_data(config, sources=sources)
        any_bad = any(r["incomplete"] or r["missing"] for r in report.values())
        sys.exit(1 if any_bad else 0)

    elif args.command == "check-and-split":
        sources = None
        if args.sources:
            sources = [s.strip() for s in args.sources.split(",")]
        check_seed_vs_synth(
            config,
            sources=sources,
            n_parts=args.n_parts,
            output_dir=Path(args.output_dir),
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    # Fix import for CLI usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from seed2synth_config import load_config
    Config.from_yaml = staticmethod(lambda f: load_config(f))

    main()
