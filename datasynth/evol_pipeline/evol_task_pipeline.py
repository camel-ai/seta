"""Per-task evolution pipeline: evol agent → datapoint agent (with self-assessment).

Mirrors seed2synth_pipeline/seed2task_pipeline.py but for evolution:
1. Evol agent reads input task → writes draft_spec.md per variant
2. Datapoint agent reads draft_spec.md → builds Harbor task → self-reviews → judge_report.md
"""

import os
import shutil
from typing import List, Optional

from pipeline_base import (
    EvolAgent, DatapointAgent,
    TaskContext, EvolutionOption, EvolvedTask, DatapointResult, SynthResult,
)
from io_utils import generate_variant_ids, load_harbor_files

import pathlib

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDE_DIR = os.path.join(_PIPELINE_DIR, "agents", "datapoint_agent_guide")


class EvolTaskPipeline:
    """Run evolution + datapoint creation for a single input task.

    Writes all output directly to output_dir/{variant_id}/.
    """

    def __init__(
        self,
        evol_agent: EvolAgent,
        datapoint_agent: DatapointAgent,
        output_dir: str,
        rollout_dir: str,
    ):
        self.evol_agent = evol_agent
        self.datapoint_agent = datapoint_agent
        self.output_dir = os.path.abspath(output_dir)
        self.rollout_dir = os.path.abspath(rollout_dir)

    # ------------------------------------------------------------------
    # Evol step
    # ------------------------------------------------------------------

    async def run_evol_agent(
        self,
        task_context: TaskContext,
        evol_opt: EvolutionOption,
        variant_paths: List[str],
    ) -> EvolvedTask:
        print(f"[{task_context.task_id}] Running Evol Agent ({evol_opt.parameters.get('target')})...")
        evolved = await self.evol_agent.evolve(
            task_context, evol_opt=evol_opt, variant_paths=variant_paths,
        )
        print(f"[{task_context.task_id}] Evolution complete.")
        return evolved

    # ------------------------------------------------------------------
    # Datapoint step
    # ------------------------------------------------------------------

    async def run_datapoint_agent(
        self,
        task_id_evol: str,
        evol_task_path: str,
        input_task_path: str,
        evol_target: str,
    ) -> DatapointResult:
        print(f"[{task_id_evol}] Running Datapoint Agent...")
        result = await self.datapoint_agent.create(
            task_id_evol=task_id_evol,
            evol_task_path=evol_task_path,
            guide_dir=GUIDE_DIR,
            input_task_path=input_task_path,
            evol_target=evol_target,
            rollout_dir=self.rollout_dir,
        )
        print(f"[{task_id_evol}] Datapoint creation complete.")
        return result

    # ------------------------------------------------------------------
    # Parse self-assessment verdict
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_judge_report(task_id_evol: str, evol_task_path: str) -> SynthResult:
        """Parse judge_report.md written by the datapoint agent.

        Checks for '## Verdict: PASS' string (same convention as seed2synth).
        """
        report_path = os.path.join(evol_task_path, "judge_report.md")
        verdict = "FAIL"
        feedback = ""
        issues: List[str] = []

        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "## Verdict: PASS" in content:
                verdict = "PASS"

            if "## Feedback" in content:
                idx = content.index("## Feedback")
                feedback = content[idx:]

            for line in content.splitlines():
                if ": FAIL" in line:
                    issues.append(line.strip())
        else:
            issues.append("judge_report.md not written by datapoint agent")

        return SynthResult(
            task_id_evol=task_id_evol,
            verdict=verdict,
            feedback=feedback,
            issues=issues,
            metadata={"evol_task_path": evol_task_path, "report": report_path},
        )

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    async def run(
        self,
        task_id: str,
        input_task_path: str,
        evol_opt: EvolutionOption,
    ) -> List[SynthResult]:
        """Full pipeline for one input task: evol → datapoint+self-assess.

        Returns one SynthResult per non-filtered variant.
        """
        input_task_path = os.path.abspath(input_task_path)
        evol_target = evol_opt.parameters.get("target", "INCREASE_DIFFICULTY")

        print(f"\n=== EvolTaskPipeline: task_id={task_id}, strategy={evol_target} ===")

        # 1. Load input task files
        task_files = load_harbor_files(pathlib.Path(input_task_path))
        task_context = TaskContext(
            task_id=task_id,
            task_files=task_files,
            metadata={"input_task_path": input_task_path},
        )

        # 2. Generate variant IDs and pre-create directories
        variant_ids = generate_variant_ids(
            task_id, evol_opt.strategy, evol_opt.max_variants,
        )
        variant_paths: List[str] = []
        for vid in variant_ids:
            vpath = os.path.abspath(os.path.join(self.output_dir, vid))
            os.makedirs(vpath, exist_ok=True)
            variant_paths.append(vpath)

        # 3. Run evol agent → draft_spec.md (skip if all variants already have one)
        need_evol = any(
            not os.path.exists(os.path.join(vp, "draft_spec.md"))
            and not os.path.exists(os.path.join(vp, "FILTERED"))
            for vp in variant_paths
        )
        if need_evol:
            await self.run_evol_agent(task_context, evol_opt, variant_paths)
        else:
            print(f"[{task_id}] All variants have draft_spec.md, skipping evol agent.")

        # 4. For each non-filtered variant: datapoint agent + parse verdict
        results: List[SynthResult] = []
        for vid, vpath in zip(variant_ids, variant_paths):
            # Check if filtered
            if os.path.exists(os.path.join(vpath, "FILTERED")):
                print(f"\n[{vid}] Marked FILTERED by evol agent. Skipping.")
                continue

            # Check if draft_spec.md was written
            if not os.path.exists(os.path.join(vpath, "draft_spec.md")):
                print(f"\n[{vid}] No draft_spec.md found. Skipping.")
                continue

            # Skip datapoint agent if judge_report.md already exists (fully done)
            if os.path.exists(os.path.join(vpath, "judge_report.md")):
                print(f"\n[{vid}] judge_report.md exists, resuming from verdict.")
            else:
                await self.run_datapoint_agent(
                    vid, vpath, input_task_path, evol_target,
                )

            # Parse the self-assessment verdict
            synth_result = self._parse_judge_report(vid, vpath)
            results.append(synth_result)

            print(f"[{vid}] Verdict: {synth_result.verdict}")
            if synth_result.issues:
                print(f"[{vid}] Issues: {synth_result.issues}")

        print(f"\n=== EvolTaskPipeline complete for {task_id}. "
              f"{len(results)} variant(s) processed. ===")
        return results
