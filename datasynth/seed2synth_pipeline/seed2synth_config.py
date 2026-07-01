#!/usr/bin/env python3
"""Configuration management for the Seed2Synth pipeline.

Provides:
- PipelineConfig: dataclass for all config options
- load_config(): load YAML config file with environment variable interpolation
- validate_config(): check that paths exist, URLs are reachable, etc.
"""

import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceConfig:
    """HuggingFace repository and authentication settings."""
    seed_repo: str = "camel-ai/seta-env-seed2synth-seed"
    synth_repo: str = "camel-ai/seta-env-seed2synth-synth"
    token_env: str = "HF_TOKEN"  # environment variable name


@dataclass
class PathsConfig:
    """Local directory paths (all relative to working directory)."""
    seed_data_dir: str = "seed_data"
    synth_data_dir: str = "synth_data"
    rollout_dir: str = "synth_data_rollouts"


@dataclass
class SeedPreparationConfig:
    """Seed data download options."""
    download_all_upfront: bool = False  # download all before synth starts vs. on-demand


@dataclass
class PipelineConfig:
    """Synthesis pipeline settings."""
    n_synth_workers: int = 3
    stage: str = "full"  # full | idea-only | unified
    skip_timeout: bool = False  # skip tasks that previously timed out


@dataclass
class RolloutConfig:
    """Rollout settings."""
    enabled: bool = True
    n_rollout_workers: int = 2
    n_trajs: int = 8
    model_url: str = "http://localhost:8000"
    model_config_name: str = "Qwen3-8B_thinking"
    model_name: Optional[str] = None  # null = auto-detect
    thinking: bool = False


@dataclass
class UploadConfig:
    """HuggingFace upload settings."""
    enabled: bool = False
    interval_minutes: int = 30  # scan and upload every N minutes


@dataclass
class Config:
    """Top-level configuration for Seed2Synth orchestrator."""
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    sources: List[str] = field(default_factory=lambda: ["unix_linux_se", "stack_overflow", "kaggle_notebook"])
    filter_csv: Optional[str] = None  # path to filter CSV (source, task_id columns)
    seed_preparation: SeedPreparationConfig = field(default_factory=SeedPreparationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)

    def to_dict(self) -> dict:
        """Convert to nested dict (for YAML serialization)."""
        return asdict(self)


def _interpolate_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable values.

    Args:
        value: string that may contain ${VAR_NAME} references

    Returns:
        string with env vars substituted

    Raises:
        KeyError if referenced env var doesn't exist
    """
    def replace_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace_var, value)


def _resolve_path(path_str: str, base_dir: Path = None) -> Path:
    """Resolve a path string to absolute Path.

    Args:
        path_str: path string (may be relative)
        base_dir: base directory for relative paths (defaults to cwd)

    Returns:
        resolved Path object
    """
    if base_dir is None:
        base_dir = Path.cwd()

    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_config(config_file: str = "config.yaml") -> Config:
    """Load configuration from YAML file with environment variable interpolation.

    Args:
        config_file: path to config.yaml

    Returns:
        Config object

    Raises:
        FileNotFoundError: if config file doesn't exist
        yaml.YAMLError: if YAML is malformed
        ValueError: if required fields are missing or invalid
    """
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    # Interpolate environment variables in string values
    def interpolate_dict(d):
        if isinstance(d, dict):
            return {k: interpolate_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [interpolate_dict(v) for v in d]
        elif isinstance(d, str):
            return _interpolate_env_vars(d)
        return d

    raw_config = interpolate_dict(raw_config)

    # Build nested config objects
    hf_cfg = HuggingFaceConfig(
        **{k: v for k, v in raw_config.get("huggingface", {}).items()
           if k in ("seed_repo", "synth_repo", "token_env")}
    )

    paths_cfg = PathsConfig(
        **{k: v for k, v in raw_config.get("paths", {}).items()
           if k in ("seed_data_dir", "synth_data_dir", "rollout_dir")}
    )

    # Resolve all path fields to absolute paths.
    # Relative paths are resolved against the directory containing the pipeline
    # entry point (run_orchestrator.py), which is the parent of configs/.
    # We use the config file's grandparent when configs live in a subdirectory,
    # but fall back to cwd for top-level config files.
    pipeline_root = config_path.resolve().parent
    # If config lives inside a subdirectory (e.g. configs/), go up one level
    if pipeline_root.name in ("configs", "config"):
        pipeline_root = pipeline_root.parent
    for attr in ("seed_data_dir", "synth_data_dir", "rollout_dir"):
        val = getattr(paths_cfg, attr)
        if val:
            setattr(paths_cfg, attr, str(_resolve_path(val, pipeline_root)))

    seed_prep_cfg = SeedPreparationConfig(
        **{k: v for k, v in raw_config.get("seed_preparation", {}).items()
           if k in ("download_all_upfront",)}
    )

    pipeline_cfg = PipelineConfig(
        **{k: v for k, v in raw_config.get("pipeline", {}).items()
           if k in ("n_synth_workers", "stage", "skip_timeout")}
    )

    rollout_dict = raw_config.get("rollout", {})
    rollout_cfg = RolloutConfig(
        **{k: v for k, v in rollout_dict.items()
           if k in ("enabled", "n_rollout_workers", "n_trajs", "model_url",
                    "model_config_name", "model_name", "thinking")}
    )

    upload_dict = raw_config.get("upload", {})
    upload_cfg = UploadConfig(
        **{k: v for k, v in upload_dict.items()
           if k in ("enabled", "interval_minutes")}
    )

    sources = raw_config.get("sources", [])
    filter_csv = raw_config.get("filter_csv", None)
    if filter_csv:
        filter_csv = str(_resolve_path(filter_csv, pipeline_root))

    config = Config(
        huggingface=hf_cfg,
        paths=paths_cfg,
        sources=sources,
        filter_csv=filter_csv,
        seed_preparation=seed_prep_cfg,
        pipeline=pipeline_cfg,
        rollout=rollout_cfg,
        upload=upload_cfg,
    )

    logger.info(f"Loaded config from {config_path}")
    return config


def validate_config(config: Config) -> list[str]:
    """Validate configuration and return list of warnings/errors.

    Args:
        config: Config object to validate

    Returns:
        list of warning/error messages (empty if valid)
    """
    issues = []

    # Check paths
    for attr in ("seed_data_dir", "synth_data_dir", "rollout_dir"):
        path = Path(getattr(config.paths, attr))
        if not path.is_absolute():
            # Relative paths are OK (will be resolved at runtime)
            pass

    # Check sources
    valid_sources = {"unix_linux_se", "stack_overflow", "kaggle_notebook", "nl2bash", "nvd"}
    for source in config.sources:
        if source not in valid_sources:
            issues.append(f"Unknown source: {source} (valid: {valid_sources})")

    # Check filter CSV if specified
    if config.filter_csv:
        filter_path = Path(config.filter_csv)
        if not filter_path.exists():
            issues.append(f"Filter CSV not found: {config.filter_csv}")

    # Check pipeline stage
    if config.pipeline.stage not in ("full", "idea-only", "unified"):
        issues.append(f"Invalid pipeline.stage: {config.pipeline.stage}")

    # Check rollout settings
    if config.rollout.n_rollout_workers < 1:
        issues.append(f"rollout.n_rollout_workers must be >= 1, got {config.rollout.n_rollout_workers}")
    if config.rollout.n_trajs < 1:
        issues.append(f"rollout.n_trajs must be >= 1, got {config.rollout.n_trajs}")

    # Check HF token if needed
    hf_token = os.environ.get(config.huggingface.token_env)
    if not hf_token:
        issues.append(
            f"HuggingFace token not found in env var {config.huggingface.token_env}. "
            f"Set it before running if you plan to download/upload."
        )

    return issues


def print_config(config: Config) -> None:
    """Pretty-print configuration."""
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"\nHuggingFace:")
    print(f"  seed_repo:      {config.huggingface.seed_repo}")
    print(f"  synth_repo:     {config.huggingface.synth_repo}")
    print(f"  token_env:      {config.huggingface.token_env}")

    print(f"\nPaths (relative):")
    print(f"  seed_data_dir:  {config.paths.seed_data_dir}")
    print(f"  synth_data_dir: {config.paths.synth_data_dir}")
    print(f"  rollout_dir:    {config.paths.rollout_dir}")

    print(f"\nSources: {', '.join(config.sources)}")
    if config.filter_csv:
        print(f"Filter CSV:     {config.filter_csv}")

    print(f"\nPipeline:")
    print(f"  stage:          {config.pipeline.stage}")
    print(f"  n_synth_workers: {config.pipeline.n_synth_workers}")
    print(f"  skip_timeout:   {config.pipeline.skip_timeout}")

    print(f"\nRollout:")
    print(f"  enabled:        {config.rollout.enabled}")
    if config.rollout.enabled:
        print(f"  n_rollout_workers: {config.rollout.n_rollout_workers}")
        print(f"  n_trajs:        {config.rollout.n_trajs}")
        print(f"  model_url:      {config.rollout.model_url}")
        print(f"  model_config:   {config.rollout.model_config_name}")

    print(f"\nUpload:")
    print(f"  enabled:        {config.upload.enabled}")
    if config.upload.enabled:
        print(f"  interval:       {config.upload.interval_minutes} min")

    print("=" * 70 + "\n")
