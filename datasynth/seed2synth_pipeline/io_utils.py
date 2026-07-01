import os
import json
import pathlib
import re
from typing import Dict, List, Optional, Any
from pipeline_base import TaskContext

def load_text_file(path: pathlib.Path) -> str:
    """Safely load text file content."""
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}")
        return ""

def load_seed_files(task_path: pathlib.Path) -> Dict[str, str]:
    """Load all text files from the seed task directory and validate existence."""
    files = {}
    if not task_path.exists():
        # print(f"WARNING: Task seed path does not exist: {task_path}")
        raise FileNotFoundError(f"Task seed path does not exist: {task_path}")
        
    # --- Validation: Ensure files shown in terminal exist ---
    # essential files: Dockerfile, docker-compose.yaml, draft_spec.md, run-tests.sh, solution.sh, task.yaml, weights.json
    essential_files = [
        "Dockerfile", "docker-compose.yaml", 
        # "draft_spec.md", 
        "run-tests.sh", "solution.sh", "task.yaml", 
        # "weights.json"
    ]
    for ef in essential_files:
        if not (task_path / ef).exists():
            # print(f"WARNING: Essential file {ef} is missing in {task_path}")
            raise FileNotFoundError(f"Essential file {ef} is missing in {task_path}")

    # tests directory check
    if not (task_path / "tests").is_dir():
        print(f"WARNING: Required 'tests' directory missing in {task_path}")
        return files

    # ask_ubuntu.json can optionally be replaced with seed_data.json or none
    # No warning if neither exists since "none" is allowed, but we know it's a seed metadata file.
    # -------------------------------------------------------

    for item in task_path.rglob('*'):
        if item.is_file():
            # Store relative path as key
            rel_path = item.relative_to(task_path).as_posix()
            # Basic textual check - might want to exclude binaries explicitly if needed
            # For now, we try to read everything as text
            content = load_text_file(item)
            files[rel_path] = content
            
    return files

def collect_rollout_data(rollout_path: pathlib.Path) -> Optional[Dict[str, str]]:
    """
    Find trajectory and test results in a specific rollout folder.
    Returns: Dict with 'trajectory' and 'test_results' paths if valid, else None.
    """
    log_dir = rollout_path / "CAMEL_LOG_DIR"
    test_results_file = rollout_path / "test_results.json"
    
    if not log_dir.exists():
        return None
        
    json_files = list(log_dir.glob("*.json"))
    if len(json_files) < 2:
        print(f"WARNING: Not enough JSON files in {log_dir}")
        return None
        
    # Sort by size, descending to find the main trajectory
    json_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    trajectory_path = str(json_files[0].absolute())
    
    # test_results required — skip runs that didn't complete evaluation
    if not test_results_file.exists():
        print(f"INFO: Skipping {rollout_path.name} — no test_results.json found (incomplete run)")
        return None
    test_results_path = str(test_results_file.absolute())
    with open(test_results_path, 'r') as f:
        try:
            test_results = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in test results file {test_results_path}: {e}")


    # find terminal log files as well, a shorter version to use when context length is a concern
    #   find rollout_path / sessions / terminal_toolkit_session_logs / *.log
    terminal_logs_dir = rollout_path / "sessions" / "terminal_toolkit_session_logs"
    if terminal_logs_dir.exists():
        log_files = list(terminal_logs_dir.glob("*.log"))
    
    return {
        "trajectory": trajectory_path,
        "test_results": test_results,
        "terminal_logs": [str(p.absolute()) for p in log_files] if terminal_logs_dir.exists() else []
    }

def load_task_context(task_id: str, seed_path: str, rollout_path: str) -> TaskContext:
    """
    Load task context including seed files and valid trajectories.
    
    Args:
        task_id: The task identifier (e.g., "0").
        seed_path: Path to the specific task's seed data directory.
        rollout_path: Path to the specific task's rollout root directory.
        
    Returns:
        TaskContext loaded with files and valid trajectory paths.
    """
    seed_p = pathlib.Path(seed_path)
    rollout_p = pathlib.Path(rollout_path)
    
    # 1. Load seed files
    seed_files = load_seed_files(seed_p)
    
    # 2. Find valid trajectories
    rollouts = []
    
    if rollout_p.exists():
        # Iterate over potential rollout subdirectories (e.g., 0.1c10edec.areal-run)
        for item in rollout_p.iterdir():
            if item.is_dir():
                rollout_data = collect_rollout_data(item)
                if rollout_data:
                    rollouts.append(rollout_data)
    
    if not rollouts:
        print(f"WARNING: No valid trajectories found for task {task_id}")

    if not seed_files:
        print(f"WARNING: No seed files found for task {task_id}")

    return TaskContext(
        task_id=task_id,
        seed_files=seed_files,
        rollouts=rollouts,
        metadata={
            "seed_path": str(seed_p),
            "rollout_path": str(rollout_p)
        }
    )

def get_next_version(task_id: str, direction: str) -> str:
    """
    Derive the next version path (e.g., file-ops -> file-ops__d1 if direction is 'depth').
    """
    direction_map = {
        "depth": "d",
        "breadth": "b"
    }
    if direction not in direction_map:
        raise ValueError(f"Invalid evolution direction: {direction}")
    suffix = direction_map[direction]
    next_version = f"{task_id}__{suffix}1"
    return next_version

def parse_task_id(task_id: str, evol_data_base: str=None, rollout_base: str=None) -> Dict[str, Any]:
    """
    Parse the task ID to extract components like seed name, evolution step, and lineage.
    
    Args:
        task_id: The task ID string, e.g. "file-ops__d1__b1".
    
    Returns:
        A dictionary with parsed components:
            - seed_name: The original seed task name (e.g. "file-ops")
            - evol_steps: List of evolution steps in order (e.g. ["d1", "b1"])
            - lineage: List of ancestor task IDs in order (e.g. ["file-ops", "file-ops__d1"])
    """
    parts = task_id.split("__")
    seed_name = parts[0]
    evol_steps = parts[1:] if len(parts) > 1 else []
    
    lineage = []
    current_id = seed_name
    for step in evol_steps:
        lineage.append(current_id)
        current_id += f"__{step}"
    
    return {
        "seed_name": seed_name,
        "task_path": f"{evol_data_base}/{seed_name}/{task_id}" if evol_data_base else f"{seed_name}/{task_id}",
        "rollout_path": f"{rollout_base}/{task_id}" if rollout_base else task_id,
        "evol_steps": evol_steps,
        "lineage": lineage
    }

def load_cleaned_trajectory(path: str, include_system: bool = False) -> str:
    """
    Load a trajectory JSON and return a thinned text representation.
    Strips redundant formatting and labels iterations to save tokens.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return "Trajectory file not found."
    
    try:
        data = json.loads(p.read_text(encoding='utf-8', errors='replace'))
        messages = data.get("request", {}).get("messages", [])
        
        cleaned_lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content") or ""

            # Label prefix with iteration number
            prefix = ""
            
            if role == "system":
                if not include_system:
                    continue
                cleaned_lines.append(f"{prefix} [System]: {content.strip()}")
            
            elif role == "user":
                cleaned_lines.append(f"{prefix} [User]: {content.strip()}")
            
            elif role == "assistant":
                if content.strip():
                    cleaned_lines.append(f"{prefix} [Assistant]: {content.strip()}")
                
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn_name = tc.get("function", {}).get("name")
                        fn_args = tc.get("function", {}).get("arguments")
                        cleaned_lines.append(f"{prefix} [Tool Call]: {fn_name}({fn_args})")
            
            elif role == "tool":
                cleaned_lines.append(f"{prefix} [Tool Result]: {content.strip()}")
                
        # Also include the final response if available
        choices = data.get("response", {}).get("choices")
        if choices and len(choices) > 0:
            final_response = choices[0].get("message", {}).get("content", "")
            if final_response:
                cleaned_lines.append(f"[Final Response]: {final_response.strip()}")
            
        return "\n".join(cleaned_lines)
        
    except Exception as e:
        return f"Error cleaning trajectory: {e}"

def load_seed_data(seed_data_path: pathlib.Path) -> Dict[str, Any]:
    """Load a single seed data JSON file (NL2Bash, StackOverflow, NVD format).

    Validates that the 'source' field exists and returns the parsed dict.
    """
    if not seed_data_path.exists():
        raise FileNotFoundError(f"Seed data file not found: {seed_data_path}")
    with open(seed_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if "source" not in data:
        raise ValueError(f"Seed data file missing 'source' field: {seed_data_path}")
    return data


_KNOWN_SOURCES = {"nl2bash", "stackoverflow", "stack_overflow", "unix_linux_se", "kaggle_notebook", "nvd"}


def validate_seed_folder(seed_data_folder: str) -> None:
    """Validate that a seed data folder is well-formed.

    Raises FileNotFoundError or NotADirectoryError if the folder or required metadata
    is missing. For kaggle_notebook source, looks for kernel-metadata.json.
    For other sources, looks for main.json. Source type is NOT required in the JSON
    itself — it can be inferred from the folder's parent directory name.
    """
    folder = pathlib.Path(seed_data_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Seed data folder not found: {seed_data_folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Seed data path is not a directory: {seed_data_folder}")

    # Determine source from parent directory name
    parent_name = folder.parent.name
    is_kaggle = parent_name == "kaggle_notebook"

    # Check for appropriate metadata file
    if is_kaggle:
        metadata_json = folder / "kernel-metadata.json"
        if not metadata_json.exists():
            raise FileNotFoundError(f"kernel-metadata.json missing in Kaggle seed folder: {seed_data_folder}")
    else:
        metadata_json = folder / "main.json"
        if not metadata_json.exists():
            raise FileNotFoundError(f"main.json missing in seed folder: {seed_data_folder}")

    # Validate JSON is parseable
    with open(metadata_json, 'r', encoding='utf-8') as f:
        json.load(f)


def load_seed_folder(seed_data_folder: str) -> tuple:
    """Load a seed data folder and return (source_type, metadata).

    For kaggle_notebook, loads kernel-metadata.json. For other sources, loads main.json.

    Source type resolution order:
      1. metadata['source'] if present
      2. Parent directory name if it matches a known source (e.g. unix_linux_se/13/)
      3. ValueError if neither works
    """
    validate_seed_folder(seed_data_folder)

    # Determine source from parent directory name
    folder_path = pathlib.Path(seed_data_folder)
    parent_name = folder_path.parent.name
    is_kaggle = parent_name == "kaggle_notebook"

    # Load appropriate metadata file
    if is_kaggle:
        metadata_json = folder_path / "kernel-metadata.json"
    else:
        metadata_json = folder_path / "main.json"

    with open(metadata_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "source" in data:
        return data["source"], data

    # Infer from parent directory name (e.g. seed_data/unix_linux_se/13/)
    if parent_name in _KNOWN_SOURCES:
        return parent_name, data

    raise ValueError(
        f"Cannot determine source type for {seed_data_folder!r}. "
        f"Add a 'source' field to the metadata JSON or place the folder under a "
        f"directory named after the source (e.g. unix_linux_se/13/)."
    )


def validate_output_path(output_path: str, seed_data_folder: str) -> None:
    """Ensure the output path does not overlap with the seed data folder.

    Raises ValueError if output_path is inside or equal to seed_data_folder.
    """
    abs_output = os.path.abspath(output_path)
    abs_seed = os.path.abspath(seed_data_folder)
    if abs_output == abs_seed or abs_output.startswith(abs_seed + os.sep):
        raise ValueError(
            f"output_path ({output_path!r}) must not be inside "
            f"seed_data_folder ({seed_data_folder!r})"
        )


def is_harbor_task_complete(task_path: str) -> bool:
    """Return True if a Harbor task folder has all required files."""
    required = [
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "solution/solve.sh",
    ]
    base = pathlib.Path(task_path)
    return all((base / p).exists() for p in required)


def setup_seed2task_dir(
    seed_data_path: str,
    output_base: str,
    source_type: str,
    task_name: str,
) -> str:
    """Create task output directory and copy seed_data.json into it.

    Directory layout: {output_base}/{source_type}/{task_name}/seed_data.json

    Returns the task output directory path.
    """
    import shutil

    task_dir = os.path.join(output_base, source_type, task_name)
    if os.path.exists(task_dir):
        print(f"Task directory already exists: {task_dir}. Removing.")
        shutil.rmtree(task_dir)
    os.makedirs(task_dir, exist_ok=True)

    # Copy seed data into task directory
    dest = os.path.join(task_dir, "seed_data.json")
    shutil.copy2(seed_data_path, dest)

    return task_dir


if __name__ == "__main__":
    traj_path = "path/to/conversation.json"

    print(load_cleaned_trajectory(traj_path))