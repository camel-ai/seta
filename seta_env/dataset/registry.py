"""Load and query the unified dataset registry (datasets.yaml)."""

from pathlib import Path

import yaml

_REGISTRY_PATH = Path(__file__).resolve().parent / "datasets.yaml"


def load_registry() -> dict:
    """Return the full datasets dict from datasets.yaml."""
    return yaml.safe_load(_REGISTRY_PATH.read_text())["datasets"]


def resolve_dataset(name: str) -> dict:
    """Look up a dataset by name and return its config.

    Args:
        name: Key in datasets.yaml (e.g. "terminal-bench-core_migrated").

    Returns:
        Dict with keys: repo (str), subfolder (str | None).

    Raises:
        KeyError: Unknown dataset name.
    """
    registry = load_registry()
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(
            f"Unknown dataset: {name!r}. Available: {available}"
        )

    entry = registry[name]
    return {
        "repo": entry["repo"],
        "subfolder": entry.get("subfolder"),
    }
