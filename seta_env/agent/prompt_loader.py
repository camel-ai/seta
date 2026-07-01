"""Prompt and agent loader for seta_env.

Prompts are flat Markdown files under seta_env/agent/prompts/<name>.md.
They are fully static — system, machine, and is_workforce are already baked in.

Usage in agent_config:
    {
        "agent":  "train_agent",        # optional, defaults to "train_agent"
        "prompt": "sys_prompt_default", # optional, takes precedence over "system_message"
        "system_message": "...",        # used only when "prompt" is absent
        ...
    }
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_system_message(prompt_name: str) -> str:
    """Return the contents of seta_env/agent/prompts/<prompt_name>.md."""
    path = _PROMPTS_DIR / f"{prompt_name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt '{prompt_name}' not found. "
            f"Expected file: {path}"
        )
    return path.read_text()


# Registry of supported agent classes.
# Add new entries here as additional agent implementations are created.
_AGENT_REGISTRY: dict[str, str] = {
    # "name": "module.path.ClassName"
    "train_agent": "seta_env.agent.train_agent.AgentTrain",
    "tito_train_agent": "seta_env.agent.tito_train_agent.AgentTrainTITO",
    # Model-free agent for stress/placement testing of the env + Daytona path
    # (see seta_env/agent/noop_agent.py). Exercises sandbox create/exec/teardown
    # without any LLM call.
    "noop": "seta_env.agent.noop_agent.AgentNoop",
}


def get_agent_class(agent_name: str):
    """Return the agent class for the given name.

    Args:
        agent_name: key in _AGENT_REGISTRY (e.g. "train_agent").

    Returns:
        The agent class (not an instance).

    Raises:
        ValueError: if agent_name is not in the registry.
    """
    if agent_name not in _AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Available: {list(_AGENT_REGISTRY)}"
        )
    module_path, _, class_name = _AGENT_REGISTRY[agent_name].rpartition(".")
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
