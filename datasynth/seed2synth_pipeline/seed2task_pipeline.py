"""
Seed2Task Pipeline v2: converts a seed data folder into a complete Harbor task.

Differences from v1 (seed2task_pipeline.py):
  - Input is a seed data folder (e.g. seed_data/unix_linux_se/13/) not a single JSON
  - Source type is auto-detected from main.json['source']
  - Output follows Harbor format (task.toml, environment/Dockerfile, solution/solve.sh, instruction.md)
  - Supports sources: nl2bash, stackoverflow, unix_linux_se

Pipeline stages:
  1. Seed2IdeaAgent   — reads seed folder, produces draft_spec.md
  2. DatapointAgent   — reads draft_spec.md, builds full Harbor task, self-reviews and writes judge_report.md

Usage:
  # Single seed folder (source auto-detected)
  python seed2task_pipeline_2.py --seed-folder /path/to/seed_data/unix_linux_se/13

  # With explicit output base
  python seed2task_pipeline_2.py --seed-folder /path/to/seed_data/unix_linux_se/13 \\
      --output-base /path/to/output

  # Idea-only mode (stop after draft_spec.md)
  python seed2task_pipeline_2.py --seed-folder /path/to/seed_data/unix_linux_se/13 \\
      --stage idea-only

  # Batch: comma-separated folders
  python seed2task_pipeline_2.py \\
      --seed-folders /path/to/unix_linux_se/13,/path/to/nl2bash/42
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from pipeline_base import (
    Seed2IdeaAgent, DatapointAgent,
    EvolvedTask, DatapointResult, SynthResult,
)
from io_utils import (
    load_seed_folder, validate_seed_folder, validate_output_path,
    is_harbor_task_complete,
)

# Paths relative to this file
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(_PIPELINE_DIR, "agents")
DATAPOINT_GUIDE_DIR = os.path.join(AGENTS_DIR, "datapoint_agent_guide")


class Seed2TaskPipeline:
    """
    Pipeline v2: seed data folder → complete Harbor task.

    Directory layout:

    output_base/
    └── <source_type>/
        └── <folder_name>/           # named after seed folder basename
            ├── draft_spec.md        # Seed2IdeaAgent output
            ├── idea_agent_log.txt
            ├── task.toml            # Harbor task files (DatapointAgent output)
            ├── instruction.md
            ├── environment/
            │   └── Dockerfile
            ├── solution/
            │   └── solve.sh
            ├── tests/
            ├── datapoint_agent_log.txt
            └── judge_report.md
    """

    def __init__(
        self,
        idea_agent: Seed2IdeaAgent,
        datapoint_agent: DatapointAgent,
        output_base: str,
    ):
        self.idea_agent = idea_agent
        self.datapoint_agent = datapoint_agent
        self.output_base = output_base

    # ── Stage 1: Seed2Idea ────────────────────────────────────────────────

    async def run_idea_agent(
        self,
        seed_data_folder: str,
        source_type: str,
        task_path: str,
    ) -> EvolvedTask:
        task_name = os.path.basename(task_path)
        print(f"[{task_name}] Running Seed2Idea Agent (source={source_type})...")
        result = await self.idea_agent.generate(
            seed_data_folder=seed_data_folder,
            source_type=source_type,
            output_path=task_path,
        )
        print(f"[{task_name}] Seed2Idea Agent complete.")
        return result

    # ── Stage 2: Datapoint creation ───────────────────────────────────────

    async def run_datapoint_agent(
        self,
        task_name: str,
        task_path: str,
        seed_data_folder: str = "",
    ) -> DatapointResult:
        print(f"[{task_name}] Running Datapoint Agent...")
        result = await self.datapoint_agent.create(
            task_id_evol=task_name,
            evol_task_path=task_path,
            guide_dir=DATAPOINT_GUIDE_DIR,
            seed_data_folder=seed_data_folder,
        )
        print(
            f"[{task_name}] Datapoint creation complete. "
            f"oracle_passed={result.oracle_passed}, empty_failed={result.empty_failed}"
        )
        return result

    # ── Stage 3: Parse self-review from judge_report.md ──────────────────

    def _parse_judge_report(self, task_name: str, task_path: str) -> SynthResult:
        """Read judge_report.md written by the datapoint agent and return a SynthResult."""
        report_path = os.path.join(task_path, "judge_report.md")
        verdict = "FAIL"
        feedback = ""
        issues = []
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                content = f.read()
            if "## Verdict: PASS" in content:
                verdict = "PASS"
            if "## Feedback for Datapoint Agent" in content:
                idx = content.index("## Feedback for Datapoint Agent")
                feedback = content[idx:]
            for line in content.splitlines():
                if ": FAIL" in line:
                    issues.append(line.strip())
        else:
            issues.append("judge_report.md not written by datapoint agent")
        return SynthResult(
            task_id_evol=task_name,
            verdict=verdict,
            feedback=feedback,
            issues=issues,
            metadata={"evol_task_path": task_path, "report": report_path},
        )

    # ── Main orchestration ────────────────────────────────────────────────

    async def run(
        self,
        seed_data_folder: str,
        stage: str = "full",
    ) -> Optional[SynthResult]:
        """
        Run the full pipeline for one seed data folder.

        Args:
            seed_data_folder: Path to a seed folder containing main.json
            stage: "full" for complete pipeline, "idea-only" to stop after draft_spec.md

        Returns:
            SynthResult if full pipeline ran, None if idea-only or on error.
        """
        # 1. Validate seed folder and auto-detect source
        validate_seed_folder(seed_data_folder)
        source_type, _ = load_seed_folder(seed_data_folder)

        folder_name = Path(seed_data_folder).name  # e.g. "13"
        task_name = f"{source_type}_{folder_name}"
        task_path = os.path.join(self.output_base, source_type, folder_name)

        validate_output_path(task_path, seed_data_folder)
        os.makedirs(task_path, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Seed2TaskPipeline: task={task_name}, source={source_type}, stage={stage}")
        print(f"Seed folder : {seed_data_folder}")
        print(f"Output path : {task_path}")
        print(f"{'=' * 60}")

        # 2. Run Seed2IdeaAgent (skip if draft_spec.md already exists)
        draft_path = os.path.join(task_path, "draft_spec.md")
        if os.path.exists(draft_path):
            print(f"[{task_name}] draft_spec.md already exists — skipping idea agent.")
            idea_elapsed = 0.0
        else:
            t_idea = time.monotonic()
            await self.run_idea_agent(seed_data_folder, source_type, task_path)
            idea_elapsed = round(time.monotonic() - t_idea, 1)
            print(f"[{task_name}] Idea agent: {idea_elapsed}s")

        if not os.path.exists(draft_path):
            print(f"[{task_name}] ERROR: draft_spec.md was not created. Aborting.")
            return None

        # Check for early ditch verdict
        with open(draft_path) as f:
            draft_content = f.read()

        if draft_content.strip().startswith("# EARLY_DITCH:"):
            # Extract ditch reason
            ditch_line = draft_content.split('\n')[0]
            ditch_reason = ditch_line.replace("# EARLY_DITCH:", "").strip()
            print(f"[{task_name}] EARLY_DITCH: {ditch_reason}")
            # Return a result with DITCH verdict
            return SynthResult(
                task_id_evol=task_name,
                verdict="DITCH",
                feedback=f"Early ditch: {ditch_reason}",
                issues=[],
                metadata={
                    "evol_task_path": task_path,
                    "ditch_reason": ditch_reason,
                    "idea_time_s": idea_elapsed,
                }
            )

        if stage == "idea-only":
            print(f"[{task_name}] Idea-only mode: stopping after draft_spec.md.")
            return None

        # For full stage, verify draft_spec.md was created and is valid (not EARLY_DITCH)
        # before running datapoint agent
        if stage == "full":
            if not os.path.exists(draft_path):
                print(f"[{task_name}] ERROR: draft_spec.md missing for full stage. Skipping datapoint agent.")
                return None

            if draft_content.strip().startswith("# EARLY_DITCH:"):
                print(f"[{task_name}] Skipping datapoint agent: task was ditched in idea stage.")
                return None

        # 3. Datapoint agent (builds task + self-reviews + writes judge_report.md)
        t_dp = time.monotonic()
        await self.run_datapoint_agent(task_name, task_path, seed_data_folder=seed_data_folder)
        dp_elapsed = round(time.monotonic() - t_dp, 1)
        print(f"[{task_name}] Datapoint agent: {dp_elapsed}s")

        judge_result = self._parse_judge_report(task_name, task_path)
        judge_result.metadata["idea_time_s"] = idea_elapsed
        judge_result.metadata["datapoint_time_s"] = dp_elapsed

        print(f"[{task_name}] Self-review verdict: {judge_result.verdict}")
        if judge_result.issues:
            print(f"[{task_name}] Issues: {judge_result.issues}")

        print(f"\n{'=' * 60}")
        print(f"Pipeline complete for {task_name}. Verdict: {judge_result.verdict}")
        if is_harbor_task_complete(task_path):
            print(f"Harbor task files present in: {task_path}")
        print(f"{'=' * 60}")
        return judge_result


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import asyncio

    from agents.claude_agents import (
        ClaudeSeed2IdeaAgent, ClaudeDatapointAgent,
    )

    DEFAULT_OUTPUT_BASE = os.path.join(_PIPELINE_DIR, "output", "seed2task_v2")

    parser = argparse.ArgumentParser(
        description="Run Seed2Task pipeline v2 on a seed data folder."
    )

    folder_group = parser.add_mutually_exclusive_group(required=True)
    folder_group.add_argument(
        "--seed-folder", type=str, default=None,
        help="Path to a single seed data folder (must contain main.json)."
    )
    folder_group.add_argument(
        "--seed-folders", type=str, default=None,
        help="Comma-separated paths to multiple seed data folders."
    )

    parser.add_argument(
        "--stage", type=str, default="full",
        choices=["full", "idea-only"],
        help="Pipeline stage: 'full' or 'idea-only' (default: full)."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Number of seed folders to process concurrently (default: 1 = sequential)."
    )
    parser.add_argument(
        "--output-base", type=str, default=DEFAULT_OUTPUT_BASE,
        help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE})."
    )
    args = parser.parse_args()

    # Resolve seed folders list
    if args.seed_folder:
        seed_folders = [args.seed_folder]
    else:
        seed_folders = [f.strip() for f in args.seed_folders.split(",") if f.strip()]

    # Validate all folders before starting
    for folder in seed_folders:
        if not os.path.isdir(folder):
            print(f"ERROR: Seed folder not found or not a directory: {folder}")
            sys.exit(1)

    print(f"Seed folders : {seed_folders}")
    print(f"Stage        : {args.stage}")
    print(f"Output base  : {args.output_base}")
    print(f"Parallelism  : {args.parallel}")

    async def run_all():
        import asyncio as _asyncio

        pipeline = Seed2TaskPipeline(
            idea_agent=ClaudeSeed2IdeaAgent(),
            datapoint_agent=ClaudeDatapointAgent(),
            output_base=args.output_base,
        )

        semaphore = _asyncio.Semaphore(args.parallel)

        async def run_one(i: int, folder: str):
            async with semaphore:
                print(f"\n{'=' * 60}")
                print(f"[{i + 1}/{len(seed_folders)}] Folder: {folder}")
                print(f"{'=' * 60}")
                return await pipeline.run(seed_data_folder=folder, stage=args.stage)

        raw = await _asyncio.gather(
            *(run_one(i, folder) for i, folder in enumerate(seed_folders)),
            return_exceptions=True,
        )

        results = []
        for folder, outcome in zip(seed_folders, raw):
            if isinstance(outcome, Exception):
                print(f"[ERROR] {folder}: {outcome}")
            elif outcome is not None:
                results.append(outcome)

        if results:
            passed = sum(1 for r in results if r.verdict == "PASS")
            print(f"\n{'=' * 60}")
            print(f"SUMMARY: {passed}/{len(results)} tasks PASSED")
            print(f"{'=' * 60}")

    asyncio.run(run_all())
