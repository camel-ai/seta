"""Configuration management for the evolution pipeline.

Provides config dataclasses, YAML loading with ``${ENV_VAR}`` interpolation,
and validation.  All paths are resolved to absolute at load time.

Three independent stages, each with its own config section:
  - ``evolve``  — multi-round evolution of seed tasks
  - ``rollout`` — agent trajectory rollout (scans folders, fills gaps)
  - ``verify``  — rollout failure analysis via trajectory judge
"""

import os
import re
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HuggingFaceConfig:
    """HuggingFace repos for input download and output upload."""
    input_repo: Optional[str] = "camel-ai/seta-env-v2"
    output_repo: Optional[str] = "camel-ai/seta-env-evol"
    token_env: str = "HF_TOKEN"


@dataclass
class StrategyConfig:
    """Single evolution strategy for one round."""
    evol_strategy: str = "depth"
    evol_target: str = "INCREASE_DIFFICULTY"
    max_variants: int = 1


@dataclass
class EvolveRoundConfig:
    """One round of evolution."""
    name: str = ""
    input_dir: str = "outputs/seeds"
    output_dir: str = "outputs/evol_data"
    strategy: StrategyConfig = field(default_factory=StrategyConfig)


@dataclass
class EvolveConfig:
    """Evolution stage — multi-round, sequential."""
    n_workers: int = 4
    skip_timeout: bool = False
    task_timeout_s: int = 3600
    rounds: List[EvolveRoundConfig] = field(default_factory=lambda: [
        EvolveRoundConfig(),
    ])


@dataclass
class RolloutModelConfig:
    """Model configuration for rollout.

    Maps to ``seta_env.utils.configs.ModelConfig`` at runtime.
    """
    model_config_name: str = "default"
    model_platform: str = "moonshot"
    model_type: str = "kimi-k2.5"
    url: Optional[str] = None
    api_key: Optional[str] = None
    tito_enabled: bool = False
    tito_validate: bool = False
    model_config_dict: Dict[str, Any] = field(default_factory=lambda: {
        "max_tokens": 4096, "stream": False, "temperature": 1.0,
    })


@dataclass
class RolloutConfig:
    """Rollout stage — scans task folders, fills gaps.

    ``tasks_dirs`` lists the task folders to roll out. Each auto-derives
    its rollout directory as ``<tasks_dir>_rollout`` unless overridden.
    ``models`` lists the model configurations to use; each model's results
    are namespaced under ``<rollout_dir>/<model_config_name>/``.

    Agent, runtime, and env settings are baked in (see ``ROLLOUT_*``
    constants below) — only the model varies per config.
    """
    n_workers: int = 2
    n_trajs: int = 1
    tasks_dirs: List[str] = field(default_factory=list)
    models: List[RolloutModelConfig] = field(default_factory=list)


# Baked-in agent/runtime/env config (same across all rollout runs).
# Only the model config varies per run.
ROLLOUT_AGENT_CONFIG: Dict[str, Any] = {
    "agent": "train_agent",
    "prompt": "sys_prompt_base",
    "max_total_tokens": 28672,
    "max_completion_tokens": 4096,
    "max_iteration": 30,
    "tool_names": [
        "shell_exec", "shell_view", "shell_wait",
        "shell_write_to_process", "shell_kill_process",
        "shell_write_content_to_file",
    ],
    "thinking": False,
}

ROLLOUT_RUNTIME_CONFIG: Dict[str, Any] = {
    "env_type": "docker",
    "trial_root": "",
    "toolkit": "docker",
}

ROLLOUT_ENV_CONFIG: Dict[str, Any] = {
    "reward_fn": "pass_ratio",
    "task_timeouts": {
        "_reset_env": 300.0,
        "_reset_agent": 120.0,
        "agent_astep": 900.0,
        "_evaluate_completion_sync": 600.0,
        "_cleanup": None,
    },
}


@dataclass
class VerifyConfig:
    """Verification stage — classifies rollout failures.

    Scans ``tasks_dirs`` (auto-deriving ``<dir>_rollout`` for each) and
    verifies tasks whose pass rate is at or below ``max_pass_rate``.
    """
    tasks_dirs: List[str] = field(default_factory=list)
    model_config_name: str = "default"
    max_pass_rate: float = 0.0


@dataclass
class UploadConfig:
    """HuggingFace upload settings."""
    enabled: bool = False
    interval_minutes: int = 30


@dataclass
class Config:
    """Top-level pipeline configuration.

    Three stages (evolve / rollout / verify), each independently invocable.
    ``filter_csv`` is a top-level gate that applies to all stages, filtering
    on the root seed task_id.
    """
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    filter_csv: Optional[str] = None
    evolve: EvolveConfig = field(default_factory=EvolveConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rollout directory convention
# ---------------------------------------------------------------------------

def rollout_dir_for(tasks_dir: str) -> str:
    """Derive the rollout directory for a given tasks folder.

    Convention: ``<tasks_dir>_rollout``.
    """
    return tasks_dir.rstrip("/") + "_rollout"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate_env_vars(value: str) -> str:
    """Replace ``${VAR_NAME}`` and ``${oc.env:VAR_NAME}`` with environment variable values."""

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        return os.environ.get(var, match.group(0))

    return re.sub(r"\$\{(?:oc\.env:)?([A-Za-z_][A-Za-z0-9_]*)\}", _replace, value)


def _interpolate_dict(d: Any) -> Any:
    """Recursively interpolate env vars in a nested dict/list/str."""
    if isinstance(d, dict):
        return {k: _interpolate_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_interpolate_dict(v) for v in d]
    if isinstance(d, str):
        return _interpolate_env_vars(d)
    return d


def _resolve_path(path_str: str, base_dir: Path) -> str:
    """Resolve a path to absolute.  Relative paths are resolved against *base_dir*."""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _pick(raw: dict, keys: set) -> dict:
    """Return only the entries from *raw* whose keys are in *keys*."""
    return {k: v for k, v in raw.items() if k in keys}


# ---------------------------------------------------------------------------
# Load / Validate / Print
# ---------------------------------------------------------------------------

_DEPRECATED_TOP_LEVEL_KEYS = {
    "strategy": "evolve.rounds[].strategy",
    "pipeline": "evolve",
    "paths": "evolve.rounds[].input_dir / output_dir",
    "seed_rollout": "rollout (unified)",
    "evolved_rollout": "rollout (unified)",
    "rollout_verification": "verify",
}


def _reject_old_format(raw: dict) -> None:
    """Hard-fail if the YAML uses pre-restructure block names."""
    found = [k for k in _DEPRECATED_TOP_LEVEL_KEYS if k in raw]
    if not found:
        return
    rename_lines = "\n  ".join(
        f"- `{k}` → `{_DEPRECATED_TOP_LEVEL_KEYS[k]}`" for k in found
    )
    raise ValueError(
        "Config uses an old pipeline YAML format. The following keys were "
        "renamed:\n  " + rename_lines + "\n\nSee `configs/config.example.yaml` "
        "for the current 3-stage layout (evolve / rollout / verify)."
    )


def _parse_strategy(raw: dict) -> StrategyConfig:
    return StrategyConfig(
        **_pick(raw, {"evol_strategy", "evol_target", "max_variants"}),
    )


def _parse_model(raw: dict) -> RolloutModelConfig:
    return RolloutModelConfig(
        **_pick(raw, {"model_config_name", "model_platform", "model_type",
                      "url", "api_key", "tito_enabled", "tito_validate",
                      "model_config_dict"}),
    )


def _parse_round(raw: dict, pipeline_root: Path) -> EvolveRoundConfig:
    strat_raw = raw.get("strategy", {})
    return EvolveRoundConfig(
        name=raw.get("name", ""),
        input_dir=_resolve_path(raw.get("input_dir", "outputs/seeds"), pipeline_root),
        output_dir=_resolve_path(raw.get("output_dir", "outputs/evol_data"), pipeline_root),
        strategy=_parse_strategy(strat_raw),
    )


def load_config(config_file: str = "config.yaml") -> Config:
    """Load a YAML config with env-var interpolation and path resolution.

    Relative paths are resolved against the pipeline root directory.
    """
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    raw = _interpolate_dict(raw)
    _reject_old_format(raw)

    # Resolve relative paths against the pipeline root directory
    # (i.e. the evol_pipeline/ dir), not the config file's dir.
    # If config lives in a subdirectory (e.g. configs/), go up one level.
    pipeline_root = config_path.parent
    if pipeline_root.name in ("configs", "config"):
        pipeline_root = pipeline_root.parent

    # -- huggingface --------------------------------------------------------
    hf_raw = raw.get("huggingface", {})
    hf = HuggingFaceConfig(
        **_pick(hf_raw, {"input_repo", "output_repo", "token_env"}),
    )

    # -- filter_csv (top-level, applies to all stages) ----------------------
    filter_csv = raw.get("filter_csv")
    if filter_csv:
        filter_csv = _resolve_path(filter_csv, config_path.parent)

    # -- evolve stage -------------------------------------------------------
    evol_raw = raw.get("evolve", {})
    rounds_raw = evol_raw.get("rounds", [])
    rounds = [_parse_round(r, pipeline_root) for r in rounds_raw]
    evolve = EvolveConfig(
        n_workers=evol_raw.get("n_workers", 4),
        skip_timeout=evol_raw.get("skip_timeout", False),
        task_timeout_s=evol_raw.get("task_timeout_s", 3600),
        rounds=rounds if rounds else [EvolveRoundConfig()],
    )

    # -- rollout stage ------------------------------------------------------
    roll_raw = raw.get("rollout", {})
    tasks_dirs_raw = roll_raw.get("tasks_dirs", [])
    tasks_dirs = [_resolve_path(d, pipeline_root) for d in tasks_dirs_raw]

    models_raw = roll_raw.get("models", [])
    models = [_parse_model(m) for m in models_raw]

    rollout = RolloutConfig(
        n_workers=roll_raw.get("n_workers", 2),
        n_trajs=roll_raw.get("n_trajs", 1),
        tasks_dirs=tasks_dirs,
        models=models,
    )

    # -- verify stage -------------------------------------------------------
    ver_raw = raw.get("verify", {})
    ver_tasks_dirs_raw = ver_raw.get("tasks_dirs", [])
    ver_tasks_dirs = [_resolve_path(d, pipeline_root) for d in ver_tasks_dirs_raw]

    verify = VerifyConfig(
        tasks_dirs=ver_tasks_dirs,
        model_config_name=ver_raw.get("model_config_name", "default"),
        max_pass_rate=ver_raw.get("max_pass_rate", 0.0),
    )

    # -- upload -------------------------------------------------------------
    upload_raw = raw.get("upload", {})
    upload = UploadConfig(**_pick(upload_raw, {"enabled", "interval_minutes"}))

    config = Config(
        huggingface=hf,
        filter_csv=filter_csv,
        evolve=evolve,
        rollout=rollout,
        verify=verify,
        upload=upload,
    )
    logger.info("Loaded config from %s", config_path)
    return config


def validate_config(config: Config) -> List[str]:
    """Return a list of warnings/errors (empty if valid)."""
    issues: List[str] = []

    # -- filter_csv ---------------------------------------------------------
    if config.filter_csv and not Path(config.filter_csv).exists():
        issues.append(f"filter_csv not found: {config.filter_csv}")

    # -- evolve rounds ------------------------------------------------------
    valid_targets = {"INCREASE_DIFFICULTY", "DECREASE_DIFFICULTY",
                     "SLIGHT_INCREASE", "SLIGHT_DECREASE",
                     "CHANGE_CONTEXT", "INCREASE_DIFFICULTY_AND_CHANGE_CONTEXT"}
    valid_strategies = {"depth", "breadth"}

    for i, rnd in enumerate(config.evolve.rounds):
        prefix = f"evolve.rounds[{i}]"
        if rnd.strategy.evol_target not in valid_targets:
            issues.append(
                f"{prefix}.strategy.evol_target: unknown '{rnd.strategy.evol_target}' "
                f"(valid: {valid_targets})"
            )
        if rnd.strategy.evol_strategy not in valid_strategies:
            issues.append(
                f"{prefix}.strategy.evol_strategy: unknown '{rnd.strategy.evol_strategy}' "
                f"(valid: {valid_strategies})"
            )
        if rnd.strategy.max_variants < 1:
            issues.append(
                f"{prefix}.strategy.max_variants must be >= 1, got {rnd.strategy.max_variants}"
            )

    if config.evolve.n_workers < 1:
        issues.append(f"evolve.n_workers must be >= 1, got {config.evolve.n_workers}")

    # -- rollout ------------------------------------------------------------
    if config.rollout.tasks_dirs:
        if config.rollout.n_workers < 1:
            issues.append(f"rollout.n_workers must be >= 1, got {config.rollout.n_workers}")
        if config.rollout.n_trajs < 1:
            issues.append(f"rollout.n_trajs must be >= 1, got {config.rollout.n_trajs}")
        if not config.rollout.models:
            issues.append("rollout.tasks_dirs specified but no models configured")

        # Check model_config_name uniqueness
        names = [m.model_config_name for m in config.rollout.models]
        if len(names) != len(set(names)):
            seen = set()
            dupes = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
            issues.append(
                f"rollout.models: duplicate model_config_name(s): {dupes}"
            )

    # -- verify -------------------------------------------------------------
    if config.verify.tasks_dirs:
        if not (0.0 <= config.verify.max_pass_rate <= 1.0):
            issues.append(
                f"verify.max_pass_rate must be in [0, 1], "
                f"got {config.verify.max_pass_rate}"
            )

    # -- HF token -----------------------------------------------------------
    if config.huggingface.input_repo or config.upload.enabled:
        token = os.environ.get(config.huggingface.token_env)
        if not token:
            issues.append(
                f"HF token not found in env var {config.huggingface.token_env}. "
                f"Set it if you need HF download/upload."
            )

    return issues


def print_config(config: Config) -> None:
    """Pretty-print configuration to stdout."""
    print("\n" + "=" * 70)
    print("EVOLUTION PIPELINE CONFIGURATION")
    print("=" * 70)

    print(f"\nHuggingFace:")
    print(f"  input_repo:  {config.huggingface.input_repo or '(disabled)'}")
    print(f"  output_repo: {config.huggingface.output_repo or '(disabled)'}")
    print(f"  token_env:   {config.huggingface.token_env}")

    if config.filter_csv:
        print(f"\nFilter CSV: {config.filter_csv}")
        print(f"  (filters all stages on root seed task_id)")

    print(f"\nEvolve:")
    print(f"  n_workers:      {config.evolve.n_workers}")
    print(f"  task_timeout_s: {config.evolve.task_timeout_s}")
    print(f"  skip_timeout:   {config.evolve.skip_timeout}")
    print(f"  rounds:         {len(config.evolve.rounds)}")
    for i, rnd in enumerate(config.evolve.rounds):
        label = rnd.name or f"round_{i+1}"
        print(f"    [{label}]")
        print(f"      input_dir:      {rnd.input_dir}")
        print(f"      output_dir:     {rnd.output_dir}")
        print(f"      evol_strategy:  {rnd.strategy.evol_strategy}")
        print(f"      evol_target:    {rnd.strategy.evol_target}")
        print(f"      max_variants:   {rnd.strategy.max_variants}")

    print(f"\nRollout:")
    if config.rollout.tasks_dirs:
        print(f"  n_workers: {config.rollout.n_workers}")
        print(f"  n_trajs:   {config.rollout.n_trajs}")
        print(f"  tasks_dirs:")
        for d in config.rollout.tasks_dirs:
            print(f"    - {d}")
            print(f"      rollout_dir: {rollout_dir_for(d)}")
        print(f"  models:")
        for m in config.rollout.models:
            print(f"    - {m.model_config_name}: {m.model_platform}/{m.model_type}")
            if m.url:
                print(f"      url: {m.url}")
    else:
        print(f"  (no tasks_dirs configured)")

    print(f"\nVerify:")
    if config.verify.tasks_dirs:
        print(f"  model_config_name: {config.verify.model_config_name}")
        print(f"  max_pass_rate:     {config.verify.max_pass_rate}")
        print(f"  tasks_dirs:")
        for d in config.verify.tasks_dirs:
            print(f"    - {d}")
            print(f"      rollout_dir: {rollout_dir_for(d)}")
    else:
        print(f"  (no tasks_dirs configured)")

    print(f"\nUpload:")
    print(f"  enabled: {config.upload.enabled}")
    if config.upload.enabled:
        print(f"  interval: {config.upload.interval_minutes} min")

    print("=" * 70 + "\n")
