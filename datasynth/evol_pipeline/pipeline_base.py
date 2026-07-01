"""Base dataclasses and abstract agent interfaces for the evolution pipeline.

This module defines the core data structures and agent contracts used throughout
the evol_pipeline. It is fully standalone — no imports from other pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvolutionOption:
    """Configuration for a single evolution strategy."""
    strategy: str                           # "depth" or "breadth"
    parameters: Dict[str, Any]              # {"target": "INCREASE_DIFFICULTY"} etc.
    max_variants: int = 1                   # how many variants to generate per input task


@dataclass
class TaskContext:
    """Loaded input task with its files and metadata."""
    task_id: str
    task_files: Dict[str, str]              # {relative_path: file_content}
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolvedTask:
    """Result from the evolution agent — pointers to created variant dirs."""
    task_id: str
    variant_paths: List[str]                # absolute paths to variant directories
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatapointResult:
    """Result from the datapoint creation stage."""
    task_id_evol: str
    oracle_passed: bool = False             # harbor run --agent oracle passes (1.0)
    empty_failed: bool = False              # harbor run --agent empty fails (0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthResult:
    """Final verdict for one evolved variant (from datapoint self-assessment)."""
    task_id_evol: str
    verdict: str = "FAIL"                   # "PASS" or "FAIL"
    feedback: str = ""                      # actionable feedback if FAIL
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract agent interfaces
# ---------------------------------------------------------------------------

class EvolAgent(ABC):
    """Reads an input task and produces draft_spec.md for each variant."""

    @abstractmethod
    async def evolve(
        self,
        task_context: TaskContext,
        evol_opt: EvolutionOption,
        variant_paths: List[str],
        **kwargs,
    ) -> EvolvedTask:
        """Generate evolution draft specs.

        Args:
            task_context: loaded input task files + metadata
            evol_opt: strategy configuration (target, max_variants)
            variant_paths: pre-created directories to write draft_spec.md into

        Returns:
            EvolvedTask with paths to populated variant directories.
        """
        ...


class DatapointAgent(ABC):
    """Reads draft_spec.md and builds a complete Harbor task with self-assessment."""

    @abstractmethod
    async def create(
        self,
        task_id_evol: str,
        evol_task_path: str,
        guide_dir: str,
        *,
        input_task_path: Optional[str] = None,
        evol_target: Optional[str] = None,
        **kwargs,
    ) -> DatapointResult:
        """Build Harbor task from draft_spec.md.

        The agent also runs oracle/empty validation and writes judge_report.md.

        Args:
            task_id_evol: variant identifier (e.g. "402__d1")
            evol_task_path: directory containing draft_spec.md (and where Harbor
                files will be written)
            guide_dir: path to datapoint_agent_guide/ with agent.md
            input_task_path: original input task for evolution fidelity comparison
            evol_target: strategy applied (e.g. "INCREASE_DIFFICULTY") for
                context in the self-review

        Returns:
            DatapointResult with oracle/empty validation flags.
        """
        ...
