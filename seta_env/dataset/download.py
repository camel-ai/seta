"""Download a dataset from the unified registry to a local directory.

Can be used as a CLI script or imported as a library function.

CLI usage:
    python -m seta_env.dataset.download <name> [--output dataset/]
    python -m seta_env.dataset.download --list

Library usage:
    from seta_env.dataset.download import download_dataset
    download_dataset("terminal-bench-core_migrated", dest=Path("dataset/terminal-bench-core_migrated"))
"""

import argparse
import os
import subprocess
from pathlib import Path

from seta_env.dataset.registry import load_registry, resolve_dataset


# ── Default download root ────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]  # terminal_agent_code/
DEFAULT_DATA_DIR = _REPO_ROOT / "dataset"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _git_env() -> dict:
    """Git env that bypasses VSCode credential helper."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.pop("GIT_ASKPASS", None)
    env.pop("GIT_CREDENTIAL_HELPER", None)
    return env


def _inject_token(repo_url: str) -> str:
    token = os.environ.get("HF_TOKEN", "")
    if token and "huggingface.co" in repo_url:
        return repo_url.replace("https://", f"https://user:{token}@")
    return repo_url


def _has_git_lfs() -> bool:
    try:
        subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _repo_id_from_url(repo_url: str) -> str:
    """Extract 'org/name' from 'https://huggingface.co/datasets/org/name'."""
    return "/".join(repo_url.rstrip("/").split("/")[-2:])


# ── Download backends ────────────────────────────────────────────────────────

def _download_via_git(name: str, repo_url: str, dest: Path, subfolder: str | None = None) -> None:
    repo_url = _inject_token(repo_url)
    env = _git_env()
    display_url = repo_url.split("@")[-1] if "@" in repo_url else repo_url

    if subfolder:
        print(f"Sparse cloning {display_url} (subfolder: {subfolder})...")
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], check=True, cwd=str(dest), env=env)
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True, cwd=str(dest), env=env)
        subprocess.run(["git", "sparse-checkout", "set", subfolder], check=True, cwd=str(dest), env=env)
        subprocess.run(["git", "pull", "origin", "main"], check=True, cwd=str(dest), env=env)
        subprocess.run(["git", "lfs", "pull"], cwd=str(dest), env=env)
    else:
        print(f"Cloning {name} from {display_url}...")
        subprocess.run(["git", "clone", repo_url, str(dest)], check=True, env=env)
        subprocess.run(["git", "lfs", "pull"], cwd=str(dest), env=env)


def _download_via_hf_api(name: str, repo_url: str, dest: Path, subfolder: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    repo_id = _repo_id_from_url(repo_url)
    print(f"Downloading {repo_id} via HuggingFace API (git-lfs not available)...")
    kwargs = dict(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
    )
    if subfolder:
        kwargs["allow_patterns"] = f"{subfolder}/**"
    snapshot_download(**kwargs)


# ── Public API ───────────────────────────────────────────────────────────────

def download_dataset(name: str, dest: Path | None = None) -> Path:
    """Download a dataset by registry name.

    Args:
        name: Key in datasets.yaml.
        dest:  Target directory. Defaults to ``dataset/<name>`` under repo root.

    Returns:
        Path to the downloaded dataset directory.
    """
    info = resolve_dataset(name)
    if dest is None:
        dest = DEFAULT_DATA_DIR / name

    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} already exists, skipping.")
        return dest

    repo_url = info["repo"]
    subfolder = info["subfolder"]

    if _has_git_lfs():
        _download_via_git(name, repo_url, dest, subfolder)
    elif "huggingface.co" in repo_url:
        _download_via_hf_api(name, repo_url, dest, subfolder)
    else:
        raise RuntimeError(
            f"git-lfs is not installed and {repo_url} is not a HuggingFace URL "
            f"(cannot fall back to HF API). Install git-lfs first."
        )

    print(f"Downloaded to {dest}")
    return dest


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    registry = load_registry()

    parser = argparse.ArgumentParser(
        description="Download a dataset from the seta_env registry.",
    )
    parser.add_argument(
        "name",
        nargs="?",
        choices=list(registry.keys()),
        help="Dataset to download.",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_datasets",
        help="List available datasets and exit.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Download destination (default: {DEFAULT_DATA_DIR}/<name>).",
    )
    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:")
        for name, cfg in registry.items():
            subfolder = cfg.get("subfolder", "")
            suffix = f"  (subfolder: {subfolder})" if subfolder else ""
            print(f"  {name}{suffix}")
        return

    if args.name is None:
        parser.error("dataset name is required (use --list to see available datasets)")

    download_dataset(args.name, dest=args.output)


if __name__ == "__main__":
    main()
