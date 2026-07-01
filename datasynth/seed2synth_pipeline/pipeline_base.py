from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import pathlib

@dataclass
class TaskContext:
    """Holds all connection information for a specific task."""
    task_id: str
    seed_files: Dict[str, str] = field(default_factory=dict)
    # rollouts is a list of dicts: [{'trajectory': path, 'test_results': path}, ...]
    rollouts: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata for tracking
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    """Result from the analysis stage."""
    task_id: str
    analysis_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvolvedTask:
    """Result from the evolution stage (draft_spec.md written per variant)."""
    task_id: str
    variant_paths: List[str] = field(default_factory=list)  # paths to created variant dirs
    new_files: Dict[str, str] = field(default_factory=dict)  # kept for compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvolutionOption:
    """Options for evolution strategy."""
    strategy: str  # e.g., "depth", "breadth", "complexity_increase"
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_variants: int = 3  # maximum number of variants to pre-create for evol agent

@dataclass
class DatapointResult:
    """Result from the datapoint creation stage."""
    task_id_evol: str
    oracle_passed: bool = False     # oracle harbor run passes all tests
    empty_failed: bool = False      # empty harbor run fails all tests
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SynthResult:
    """Result from the judge evaluation stage."""
    task_id_evol: str
    verdict: str = "FAIL"           # "PASS" or "FAIL"
    feedback: str = ""              # actionable feedback for datapoint agent on FAIL
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AnalysisAgent(ABC):
    """Abstract interface for agents that analyze execution traces."""

    @abstractmethod
    async def analyze(self, task_context: TaskContext, evol_opt: EvolutionOption = None, **kwargs) -> AnalysisResult:
        """Analyze the task execution and return insights."""
        pass

class EvolAgent(ABC):
    """Abstract interface for agents that evolve tasks based on analysis."""

    @abstractmethod
    async def evolve(self, task_context: TaskContext, analysis_result: AnalysisResult, evol_opt: EvolutionOption = None, variant_paths: List[str] = None, **kwargs) -> EvolvedTask:
        """Evolve the task: plan N variants via DAG, research via web, write draft_spec.md per variant."""
        pass

class DirectEvolAgent(ABC):
    """Abstract interface for evolution agents that work directly from seed task files (no analysis report)."""

    @abstractmethod
    async def evolve(
        self,
        task_context: TaskContext,
        evol_opt: EvolutionOption,
        variant_paths: List[str],
        **kwargs
    ) -> EvolvedTask:
        """Evolve the task by reading seed files directly and choosing from embedded strategies."""
        pass

class Seed2IdeaAgent(ABC):
    """Abstract interface for agents that convert raw seed data into draft_spec.md."""

    @abstractmethod
    async def generate(
        self,
        seed_data_folder: str,
        source_type: str,
        output_path: str,
        **kwargs
    ) -> EvolvedTask:
        """Given a seed data folder and source type, generate draft_spec.md in output_path."""
        pass


class DatapointAgent(ABC):
    """Abstract interface for agents that create full task folders from draft_spec.md."""

    @abstractmethod
    async def create(self, task_id_evol: str, evol_task_path: str, guide_dir: str, **kwargs) -> DatapointResult:
        """Create a complete Harbor task folder from draft_spec.md."""
        pass

class JudgeAgent(ABC):
    """Abstract interface for agents that evaluate task quality."""

    @abstractmethod
    async def judge(self, task_id_evol: str, evol_task_path: str, datapoint_result: DatapointResult, **kwargs) -> SynthResult:
        """Evaluate task coherence, test quality, long horizon, and file completeness."""
        pass
