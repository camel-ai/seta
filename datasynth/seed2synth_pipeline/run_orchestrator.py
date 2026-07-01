#!/usr/bin/env python3
"""Seed2Synth Orchestrator - Config-Driven Entry Point

A simplified, portable interface to the entire Seed2Synth pipeline.
All configuration is done in YAML — no hardcoded paths or CLI arguments.

Quick Start:
    1. Copy the example config:
       cp configs/config.example.yaml config.yaml

    2. Edit config.yaml (set model URL, worker counts, HF token env var, etc.)

    3. Run:
       python run_orchestrator.py config.yaml                # Full pipeline
       python run_orchestrator.py config.yaml --dry-run      # Preview only
       python run_orchestrator.py config.yaml --synth-only   # Skip rollout
       python run_orchestrator.py config.yaml --rollout-only # Skip synthesis

Environment Variables:
    HF_TOKEN=<token>           # For HuggingFace downloads/uploads
    SGLANG_URL=http://...      # Model server (can also set in config.yaml)

Flags:
    --dry-run              Show what would be queued (no execution)
    --synth-only           Run only synthesis (skip rollout)
    --rollout-only         Run only rollout (skip synthesis)
    --generate-summary     Regenerate summary.csv from synth_info.json files
    --n-synth-workers N    Override config's worker count
    --n-rollout-workers N  Override config's rollout worker count
    --debug                Verbose logging

For more details, see README.md or inspect configs/config.example.yaml.
"""

import argparse
import asyncio
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from seed2synth_config import load_config, validate_config, print_config
from seed2synth_orchestrator import (
    Seed2SynthOrchestrator,
    collect_synth_queue,
    find_rollout_gaps,
    get_harbor_tasks,
    get_synth_task_ids,
)
from hf_utils import download_metadata, download_seed_task

logger = logging.getLogger(__name__)


def filter_csv_by_task_list(csv_path, task_list_file):
    """Filter a CSV to only include task IDs from a filter file.

    Filter file format: source, task_id (CSV with header)
    Returns path to temporary filtered CSV.
    """
    # Read filter (source, task_id pairs)
    filter_set = set()
    with open(task_list_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row.get("source", "").strip()
            task_id = row.get("task_id", "").strip()
            if source and task_id:
                filter_set.add((source, task_id))

    # Read original CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    # Filter to matching tasks
    filtered_rows = []
    current_source = Path(csv_path).parent.name  # infer from CSV path
    for row in all_rows:
        task_id = row.get("task_id", row.get("folder", "")).strip()
        if (current_source, task_id) in filter_set:
            filtered_rows.append(row)

    # Create temp CSV
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)
    temp_file.close()

    logger.info(f"Filtered {len(all_rows)} tasks to {len(filtered_rows)} from filter CSV")
    return temp_file.name


def build_source_configs(config) -> list:
    """Build source_configs list from Config object and metadata CSVs.

    Format: [{"source": "unix_linux_se", "csv_path": "...metadata.csv"}, ...]
    """
    seed_data_dir = Path(config.paths.seed_data_dir)
    source_configs = []

    for source in config.sources:
        metadata_path = seed_data_dir / source / "metadata.csv"
        if metadata_path.exists():
            source_configs.append({
                "source": source,
                "csv_path": str(metadata_path),
            })
        else:
            logger.warning(f"metadata.csv not found for {source}, skipping")

    return source_configs


async def main():
    parser = argparse.ArgumentParser(
        description="Seed2Synth orchestrator (config-driven)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_orchestrator.py config.yaml
    python run_orchestrator.py config.yaml --dry-run
    python run_orchestrator.py config.yaml --synth-only --n-synth-workers 5
    python run_orchestrator.py config.yaml --rollout-only
    python run_orchestrator.py config.yaml --generate-summary
        """,
    )

    parser.add_argument("config", nargs="?", default="config.yaml",
                        help="Config YAML file (default: config.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be queued, no execution")
    parser.add_argument("--synth-only", action="store_true",
                        help="Run only synth loop")
    parser.add_argument("--rollout-only", action="store_true",
                        help="Run only rollout loop")
    parser.add_argument("--generate-summary", action="store_true",
                        help="Scan synth_info.json files and generate summary.csv")
    parser.add_argument("--n-synth-workers", type=int, default=None,
                        help="Override synth workers from config")
    parser.add_argument("--n-rollout-workers", type=int, default=None,
                        help="Override rollout workers from config")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("camel").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Load and validate config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    issues = validate_config(config)
    if issues:
        print("\nCONFIG WARNINGS/ERRORS:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)

    print_config(config)

    # Override config from CLI args
    if args.n_synth_workers:
        config.pipeline.n_synth_workers = args.n_synth_workers
    if args.n_rollout_workers:
        config.rollout.n_rollout_workers = args.n_rollout_workers

    # PHASE 1: PREPARATION
    logger.info("=" * 70)
    logger.info("PHASE 1: PREPARATION")
    logger.info("=" * 70)

    seed_data_dir = Path(config.paths.seed_data_dir)
    synth_data_dir = Path(config.paths.synth_data_dir)

    # Download metadata CSVs
    logger.info("Downloading metadata CSVs from HF...")
    download_metadata(config, skip_existing=True)

    # Build source_configs with metadata paths
    source_configs = build_source_configs(config)
    if not source_configs:
        print("ERROR: No valid metadata.csv files found", file=sys.stderr)
        sys.exit(1)

    # Apply filter CSV if specified
    if config.filter_csv:
        logger.info(f"Applying filter from {config.filter_csv}...")
        filtered_source_configs = []
        for cfg in source_configs:
            filtered_csv = filter_csv_by_task_list(cfg["csv_path"], config.filter_csv)
            filtered_cfg = cfg.copy()
            filtered_cfg["csv_path"] = filtered_csv
            filtered_source_configs.append(filtered_cfg)
        source_configs = filtered_source_configs

    logger.info(f"Prepared {len(source_configs)} sources for synthesis")

    # Generate summary if requested
    if args.generate_summary:
        logger.info("Generating summary.csv from synth_info.json files...")
        for source_dir in synth_data_dir.iterdir():
            if not source_dir.is_dir() or source_dir.name.startswith("."):
                continue

            source = source_dir.name
            summary_rows = []

            for task_dir in source_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                synth_info_path = task_dir / "synth_info.json"
                if synth_info_path.exists():
                    import json
                    try:
                        with open(synth_info_path) as f:
                            info = json.load(f)
                        # Build row compatible with old summary.csv format
                        row = {
                            "task_id": info.get("task_id"),
                            "source": info.get("source"),
                            "category": "",  # TODO: get from metadata.csv
                            "title": "",
                            "status": info.get("status"),
                            "verdict": info.get("verdict"),
                            "stage": info.get("stage"),
                            "idea_time_s": info.get("idea_time_s", ""),
                            "datapoint_time_s": info.get("datapoint_time_s", ""),
                            "total_synth_time_s": info.get("total_synth_time_s", ""),
                            "timestamp": info.get("timestamp"),
                        }
                        summary_rows.append(row)
                    except Exception as e:
                        logger.error(f"Error reading {synth_info_path}: {e}")

            if summary_rows:
                summary_path = source_dir / "summary.csv"
                fieldnames = list(summary_rows[0].keys())
                with open(summary_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(summary_rows)
                logger.info(f"Wrote {len(summary_rows)} tasks to {summary_path}")

        return

    # Dry-run: just show what would be queued
    if args.dry_run:
        logger.info("=" * 70)
        logger.info("DRY-RUN: Task Queue")
        logger.info("=" * 70)

        q = collect_synth_queue(
            source_configs,
            str(seed_data_dir),
            str(synth_data_dir),
            max_tasks=None,
            skip_timeout=config.pipeline.skip_timeout,
        )
        print(f"\nSynth queue: {len(q)} tasks")
        for t in q[:20]:  # Show first 20
            print(f"  [{t['source']}] {t['task_id']:30s} {t['category']:20s} {t['title'][:40]}")
        if len(q) > 20:
            print(f"  ... and {len(q) - 20} more")

        if config.rollout.enabled:
            gaps = find_rollout_gaps(
                str(synth_data_dir),
                config.paths.rollout_dir,
                config.rollout.model_config_name,
            )
            print(f"\nRollout gaps ({config.rollout.model_config_name}): {len(gaps)} tasks")
            for t in gaps[:10]:
                print(f"  {t['task_name']}")
            if len(gaps) > 10:
                print(f"  ... and {len(gaps) - 10} more")
        return

    # PHASE 2-4: Synth, Rollout, Upload
    logger.info("=" * 70)
    logger.info("PHASE 2-4: SYNTHESIS & ROLLOUT")
    logger.info("=" * 70)

    orch = Seed2SynthOrchestrator(
        source_configs=source_configs,
        seed_data_dir=str(seed_data_dir),
        synth_data_dir=str(synth_data_dir),
        rollout_dir=config.paths.rollout_dir,
        model_url=config.rollout.model_url,
        model_config_name=config.rollout.model_config_name,
        model_name=config.rollout.model_name,
        n_synth_workers=config.pipeline.n_synth_workers,
        n_rollout_workers=config.rollout.n_rollout_workers if config.rollout.enabled else 0,
        n_trajs=config.rollout.n_trajs,
        max_tasks=None,
        synth_stage=config.pipeline.stage,
        thinking=config.rollout.thinking,
        skip_timeout=config.pipeline.skip_timeout,
        hf_config={
            "seed_repo": config.huggingface.seed_repo,
            "synth_repo": config.huggingface.synth_repo,
            "token_env": config.huggingface.token_env,
        },
    )

    if args.synth_only:
        results = {"synth_results": await orch.run_synth_loop(), "rollout_results": []}
    elif args.rollout_only:
        results = {"synth_results": [], "rollout_results": await orch.run_rollout_loop()}
    elif config.rollout.enabled:
        results = await orch.run()
    else:
        results = {"synth_results": await orch.run_synth_loop(), "rollout_results": []}

    orch.print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
