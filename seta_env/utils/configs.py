"""Structured configuration for seta_env.

Hierarchical config mirroring the runtime class hierarchy:

    EnvConfig           — reward function and timeouts
    ModelConfig         — LLM inference backend
    AgentConfig         — agent behaviour and tool selection
    RuntimeConfig       — environment backend + trial output path
    TerminalEnvConfig   — composes all above; enough to construct GRPORollout
    EvalConfig          — adds eval-loop concerns (workers, seed, dataset)

Each level is independently usable. TerminalEnvConfig is the core config
shared between standalone eval and AReaL training.

Typical usage
-------------
::

    from seta_env.utils.configs import EvalConfig, load_eval_config

    cfg, _ = load_eval_config(sys.argv[1:], EvalConfig)
    tasks  = load_tasks(cfg)

Override fields on the command line::

    python eval.py --config configs/eval_default.yaml \\
        terminal_env.model.model_type=Qwen/Qwen3-32B \\
        terminal_env.runtime.env_type=remote_docker \\
        workers=8
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, TypeVar

ConfigT = TypeVar("ConfigT")


# ══════════════════════════════════════════════════════════════════════════════
# ModelConfig — mirrors ModelFactory.create() signature exactly
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """LLM inference backend.

    Field names mirror camel ModelFactory.create() exactly:
    model_platform, model_type, api_key, url, model_config_dict.

    model_platform uses the camel ModelPlatformType enum value (lowercase),
    e.g. "sglang", "openai_compatible_model", "aws_bedrock".
    """

    model_platform: str = field(
        default="sglang",
        metadata={
            "help": (
                "camel ModelPlatformType value (lowercase). "
                "Common values: sglang | openai_compatible_model | aws_bedrock."
            ),
        },
    )
    model_type: str = field(
        default="",
        metadata={
            "help": "Model identifier: HuggingFace path, model name, or Bedrock model ID.",
        },
    )
    url: str = field(
        default="",
        metadata={
            "help": (
                "OpenAI-compatible endpoint URL. "
                "Required for sglang and openai_compatible_model. "
                "Example: http://localhost:30000/v1"
            ),
        },
    )
    api_key: str = field(
        default="",
        metadata={
            "help": "API key for the model backend.",
        },
    )
    tito_enabled: bool = field(
        default=False,
        metadata={"help": "Enable TITO (Token-In-Token-Out) caching."},
    )
    tito_validate: bool = field(
        default=False,
        metadata={"help": "Validate TITO cache hits (disable after validation)."},
    )
    # Passed directly to the model API as **kwargs.
    # Token limit keys vary by provider:
    #   - OpenAI / Azure GPT-5+: use "max_completion_tokens"
    #   - sglang / older OpenAI / Kimi: use "max_tokens"
    # Do NOT set both — some APIs reject the unsupported key.
    # Each eval config should explicitly set the correct one.
    model_config_dict: dict = field(
        default_factory=lambda: {"stream": False},
        metadata={"help": "model_config_dict passed directly to ModelFactory.create()."},
    )


# ══════════════════════════════════════════════════════════════════════════════
# AgentConfig
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """Agent behaviour and tool selection."""

    agent: str = field(
        default="train_agent",
        metadata={
            "help": "Agent class key in the seta_env agent registry.",
            "choices": ["train_agent"],
        },
    )
    prompt: str = field(
        default="sys_prompt_base",
        metadata={
            "help": (
                "Prompt file name (without .md) under seta_env/agent/prompts/. "
                "Example: sys_prompt_default"
            ),
        },
    )
    max_total_tokens: int = field(
        default=28672,
        metadata={"help": "Total context-window token budget for the agent per trajectory."},
    )
    max_completion_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens the model may generate in a single turn."},
    )
    max_iteration: int = field(
        default=30,
        metadata={"help": "Maximum number of agent tool-call iterations per trajectory."},
    )
    tool_names: List[str] = field(
        default_factory=lambda: [
            "shell_exec",
            "shell_view",
            "shell_wait",
            "shell_write_to_process",
            "shell_kill_process",
            "shell_write_content_to_file",
        ],
        metadata={"help": "Tool names the agent is allowed to call."},
    )
    thinking: bool = field(
        default=True,
        metadata={
            "help": "Set False to append /no_think to the system prompt (disables chain-of-thought).",
        },
    )
    max_parallel_tool_calls: int = field(
        default=0,
        metadata={
            "help": (
                "Cap on parallel tool calls per assistant turn. Excess calls are "
                "rejected with an explicit tool-result message. 0 disables the cap."
            ),
        },
    )
    parallel_internal_tools: bool = field(
        default=False,
        metadata={
            "help": (
                "Dispatch internal tool calls within a turn via asyncio.gather "
                "instead of a sequential await-loop. Per-resource concurrency "
                "safety lives in each toolkit (e.g. TerminalToolkit's per-session "
                "asyncio.Lock)."
            ),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# RuntimeConfig — Docker / Daytona environment backend
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeConfig:
    """Docker / Daytona environment runtime.

    Controls which execution backend GRPORollout uses, how it connects
    to the slot-pool scheduler for remote_docker mode, and where trial
    outputs are written.
    """

    env_type: str = field(
        default="docker",
        metadata={
            "help": "Execution backend for task environments.",
            "choices": ["remote_docker", "docker", "daytona"],
        },
    )
    toolkit: str = field(
        default="auto",
        metadata={
            "help": (
                "Terminal toolkit implementation. "
                "'auto' selects TerminalToolkitDocker for local docker, "
                "TerminalToolkit (tmux) otherwise. "
                "'tmux' forces tmux-based toolkit. "
                "'docker' forces Docker API toolkit (local docker only)."
            ),
            "choices": ["auto", "tmux", "docker"],
        },
    )
    trial_root: str = field(
        default="outputs/trials",
        metadata={
            "help": "Root directory for per-task trial outputs (logs, perf traces).",
        },
    )
    scheduler_url: str = field(
        default="http://127.0.0.1:8000",
        metadata={
            "help": "Slot-pool scheduler URL. Used only when env_type=remote_docker.",
        },
    )
    node_api_key: str = field(
        default="harbor-node-dev-key",
        metadata={
            "help": "API key for the remote node manager. Used only when env_type=remote_docker.",
        },
    )
    override_cpus: int | None = field(
        default=None,
        metadata={
            "help": (
                "Override CPU count for ALL task sandboxes regardless of each "
                "task.toml's [environment].cpus value. When None, harbor uses "
                "the per-task value (default 1). Setting this overrides every "
                "task uniformly."
            ),
        },
    )
    override_memory_mb: int | None = field(
        default=None,
        metadata={
            "help": (
                "Override sandbox memory (MB) for ALL tasks, ignoring per-task "
                "task.toml values. None = use per-task value (default 2048)."
            ),
        },
    )
    override_storage_mb: int | None = field(
        default=None,
        metadata={
            "help": (
                "Override sandbox disk (MB) for ALL tasks, ignoring per-task "
                "task.toml values. None = use per-task value (default 10240). "
                "Affects only NEW sandboxes; running ones keep their original "
                "allocation."
            ),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# EnvConfig — reward function and timeouts
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnvConfig:
    """Environment evaluation settings forwarded to TerminalEnvironment."""

    reward_fn: str = field(
        default="pass_ratio",
        metadata={
            "help": "Reward function name.",
            "choices": ["pass_ratio"],
        },
    )
    task_timeouts: dict = field(
        default_factory=lambda: {
            "_reset_env":                  300.0,
            "_reset_agent":                120.0,
            "agent_astep":                 300.0,
            "_evaluate_completion_sync":   600.0,
            "_cleanup":                    None,
        },
        metadata={
            "help": (
                "Per-stage timeout overrides (seconds). "
                "Keys: _reset_env, _reset_agent, agent_astep, "
                "_evaluate_completion_sync, _cleanup."
            ),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# TerminalEnvConfig — composed config for TerminalEnvironment / GRPORollout
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TerminalEnvConfig:
    """Everything needed to construct and run GRPORollout → TerminalEnvironment.

    When model is provided (default), TerminalEnvironment creates the model
    internally via ModelFactory.  When model is set to null in YAML (AReaL
    workflow), the caller must set model_config={"model": <instance>} at
    runtime before passing to GRPORollout.
    """

    agent: AgentConfig = field(
        default_factory=AgentConfig,
        metadata={"help": "Agent behaviour and tool configuration."},
    )
    model: Optional[ModelConfig] = field(
        default_factory=ModelConfig,
        metadata={
            "help": (
                "LLM inference backend. "
                "Set to null when the model is passed as an instance at runtime (e.g. AReaL)."
            ),
        },
    )
    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig,
        metadata={"help": "Environment runtime and trial output configuration."},
    )
    env: EnvConfig = field(
        default_factory=EnvConfig,
        metadata={"help": "Reward function and timeout settings."},
    )


# ══════════════════════════════════════════════════════════════════════════════
# EvalConfig — top-level evaluation configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """Top-level evaluation configuration.

    Composes TerminalEnvConfig with eval-loop concerns:
    parallelism, dataset, and output paths.
    """

    terminal_env: TerminalEnvConfig = field(
        default_factory=TerminalEnvConfig,
        metadata={"help": "TerminalEnvironment / GRPORollout configuration."},
    )
    n_trajs: int = field(
        default=1,
        metadata={"help": "Number of parallel trajectories per task (use >1 for pass@k eval)."},
    )
    workers: int = field(
        default=4,
        metadata={"help": "Maximum number of tasks running concurrently (asyncio semaphore)."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."},
    )
    dataset: str = field(
        default="seta-env-v2",
        metadata={
            "help": (
                "Dataset label from seta_env/dataset/datasets.yaml (e.g. 'seta-env-v2'), "
                "or a local path to a harbor-format task folder. "
                "Labels are auto-downloaded to dataset/<label>/ if not already present."
            ),
        },
    )
    output_dir: str = field(
        default="outputs/eval",
        metadata={"help": "Root directory for all trial outputs (trials/, summary.json, results.csv)."},
    )
    experiment_name: str = field(
        default="eval",
        metadata={"help": "Sub-directory grouping related trials under output_dir."},
    )
    trial_name: str = field(
        default="",
        metadata={
            "help": (
                "Unique identifier for this run. "
                "Auto-generated as '{model_slug}_{YYYYMMDD_HHMMSS}' when left empty."
            ),
        },
    )
    rank: int = field(
        default=0,
        metadata={"help": "Rank for multi-node task sharding (0-indexed)."},
    )
    world_size: int = field(
        default=1,
        metadata={"help": "Total number of evaluation nodes for task sharding."},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Config loader
# ══════════════════════════════════════════════════════════════════════════════

def _parse_cli_args(argv: list[str]):
    """Parse --config path plus hydra-style key=value overrides."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True,
                        help="Path to a YAML config file.")
    args, overrides = parser.parse_known_args(argv)
    return Path(args.config).absolute(), overrides


def load_eval_config(
    argv: list[str],
    config_cls: type[ConfigT] = EvalConfig,  # type: ignore[assignment]
) -> tuple[ConfigT, str]:
    """Load a YAML config file and apply Hydra-style CLI overrides.

    Returns (config_instance, config_file_path_str).
    """
    from hydra import compose as hydra_compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir
    from omegaconf import OmegaConf

    config_file, overrides = _parse_cli_args(argv)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config_name = config_file.stem

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize_config_dir(config_dir=str(config_file.parent), job_name="seta_eval", version_base=None)
    raw_cfg = hydra_compose(config_name=config_name, overrides=overrides)

    structured = OmegaConf.structured(config_cls)
    merged = OmegaConf.merge(structured, raw_cfg)
    cfg = OmegaConf.to_object(merged)

    if not isinstance(cfg, config_cls):
        raise TypeError(
            f"Config resolved to {type(cfg).__name__}, expected {config_cls.__name__}"
        )

    return cfg, str(config_file)


def save_config(cfg: Any, output_dir: str, filename: str = "eval_config.yaml") -> str:
    """Save a config object to YAML in output_dir."""
    from dataclasses import asdict
    import yaml

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    try:
        from omegaconf import OmegaConf, DictConfig
        if isinstance(cfg, DictConfig):
            data = OmegaConf.to_container(cfg, resolve=True)
        else:
            data = asdict(cfg)
    except Exception:
        from dataclasses import asdict
        data = asdict(cfg)

    with open(out_path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Config → runtime dict builders (backward compat for TerminalEnvironment)
# ══════════════════════════════════════════════════════════════════════════════


def build_agent_config(cfg: AgentConfig) -> dict:
    """Convert AgentConfig to the dict expected by TerminalEnvironment._reset_agent()."""
    return {
        "agent":                 cfg.agent,
        "prompt":                cfg.prompt,
        "max_total_tokens":      cfg.max_total_tokens,
        "max_completion_tokens": cfg.max_completion_tokens,
        "max_iteration":         cfg.max_iteration,
        "tool_names":            cfg.tool_names,
        "thinking":              cfg.thinking,
        "max_parallel_tool_calls": cfg.max_parallel_tool_calls,
        "parallel_internal_tools": cfg.parallel_internal_tools,
    }


def build_env_config(runtime: RuntimeConfig, env: EnvConfig) -> dict:
    """Convert RuntimeConfig + EnvConfig to the dict expected by GRPORollout."""
    return {
        "environment_type": runtime.env_type,
        "scheduler_url":    runtime.scheduler_url or None,
        "node_api_key":     runtime.node_api_key or None,
        "reward_fn":        env.reward_fn,
        "task_timeouts":    env.task_timeouts,
    }


def build_model_config(cfg: Optional[ModelConfig]) -> dict:
    """Convert ModelConfig to the dict expected by TerminalEnvironment.

    Returns an empty dict if cfg is None (model will be passed as instance).
    """
    if cfg is None:
        return {}
    from dataclasses import asdict
    return asdict(cfg)


def build_configs_from_terminal_env(te_cfg: TerminalEnvConfig) -> tuple[dict, dict, dict]:
    """Convenience: extract all three dicts from a TerminalEnvConfig.

    Returns (agent_config, model_config, env_config) as dicts.
    """
    agent_config = build_agent_config(te_cfg.agent)
    model_config = build_model_config(te_cfg.model)
    env_config = build_env_config(te_cfg.runtime, te_cfg.env)
    return agent_config, model_config, env_config


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loader
# ══════════════════════════════════════════════════════════════════════════════

def resolve_dataset_path(dataset: str, repo_root: Optional[str] = None) -> Path:
    """Resolve a dataset label or path to an absolute directory path.

    If ``dataset`` is a registered label in datasets.yaml, auto-downloads
    it to ``dataset/<label>/`` under repo root if not already present.
    Otherwise treats it as a local path (absolute or repo-relative).
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]

    # Check if it's a registered label
    try:
        from seta_env.dataset.registry import load_registry
        registry = load_registry()
        if dataset in registry:
            local_path = root / "dataset" / dataset
            if not local_path.exists() or not any(local_path.iterdir()):
                from seta_env.dataset.download import download_dataset
                download_dataset(dataset, dest=local_path)
            return local_path
    except Exception:
        pass

    # Treat as a path
    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    return dataset_path


def load_tasks(cfg: EvalConfig, repo_root: Optional[str] = None) -> list:
    """Load harbor-format dataset and apply rank/world_size sharding."""
    from seta_env.dataset import load_harbor_dataset

    dataset_path = resolve_dataset_path(cfg.dataset, repo_root)
    ds = load_harbor_dataset(dataset_path)

    if cfg.world_size > 1:
        indices = [i for i in range(len(ds)) if i % cfg.world_size == cfg.rank]
        ds = ds.select(indices)

    return [dict(row) for row in ds]
