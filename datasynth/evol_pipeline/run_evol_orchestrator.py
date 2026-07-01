#!/usr/bin/env python3
"""CLI entry point for the evolution pipeline.

Usage:
    python run_evol_orchestrator.py config.yaml                    # all stages
    python run_evol_orchestrator.py config.yaml evolve             # evolve only
    python run_evol_orchestrator.py config.yaml rollout            # rollout only
    python run_evol_orchestrator.py config.yaml verify             # verify only
    python run_evol_orchestrator.py config.yaml evolve --dry-run
    python run_evol_orchestrator.py config.yaml --generate-summary
    python run_evol_orchestrator.py config.yaml --upload
"""

import argparse
import asyncio
import logging

from evol_config import load_config, validate_config, print_config
from evol_orchestrator import EvolOrchestrator, collect_evol_queue, write_summary_csv
from hf_utils import generate_filter_csvs, upload_output_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress noisy logs
for _noisy in ("huggingface_hub", "huggingface_hub.file_download",
               "huggingface_hub._commit_api", "huggingface_hub.utils",
               "filelock",
               "camel.base_model", "camel.models", "camel.agents",
               "camel.camel.agents.chat_agent",
               "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evolution Pipeline Orchestrator")
    p.add_argument("config", help="Path to YAML config file")
    p.add_argument("stage", nargs="?", default="all",
                   choices=["evolve", "rollout", "verify", "all"],
                   help="Pipeline stage to run (default: all)")
    p.add_argument("--dry-run", action="store_true", help="Show queue without executing")
    p.add_argument("--n-workers", type=int, help="Override n_workers for evolve stage")
    p.add_argument("--generate-summary", action="store_true", help="Regenerate summary.csv")
    p.add_argument("--generate-filters", action="store_true", help="Generate filter CSVs")
    p.add_argument("--n-parts", type=int, default=4, help="Number of filter CSV parts")
    p.add_argument("--upload", action="store_true", help="Upload PASS variants to output_repo")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.n_workers:
        config.evolve.n_workers = args.n_workers

    print_config(config)

    issues = validate_config(config)
    if issues:
        print("\nConfig validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()

    # --- Generate summary ---
    if args.generate_summary:
        for rnd in config.evolve.rounds:
            csv_path = write_summary_csv(rnd.output_dir)
            print(f"Summary CSV: {csv_path}")
        return

    # --- Generate filter CSVs ---
    if args.generate_filters:
        input_dir = config.evolve.rounds[0].input_dir if config.evolve.rounds else "."
        paths = generate_filter_csvs(input_dir, args.n_parts, output_dir="configs/filters")
        print(f"Generated {len(paths)} filter CSVs:")
        for p in paths:
            print(f"  {p}")
        return

    # --- Upload ---
    if args.upload:
        n = upload_output_data(config)
        print(f"Uploaded {n} variant(s).")
        return

    # --- Dry run ---
    if args.dry_run:
        for i, rnd in enumerate(config.evolve.rounds):
            label = rnd.name or f"round_{i+1}"
            queue = collect_evol_queue(rnd, config)
            print(f"\n[DRY RUN] [{label}] Would process {len(queue)} tasks:")
            for item in queue[:20]:
                print(f"  {item['task_id']} <- {item['input_task_path']}")
            if len(queue) > 20:
                print(f"  ... and {len(queue) - 20} more")
        return

    # --- Full run ---
    orchestrator = EvolOrchestrator(config)
    results = await orchestrator.run(stage=args.stage)
    orchestrator.print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
