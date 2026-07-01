# Plan 02 — Utils (Used Subset Only)

## Source
`seta_env/utils/utils.py`

Only two functions from this file are imported anywhere in the package:
- `async_timer` — imported in `seta_env/environments/terminal_env.py:27`
- `load_main_trajectory` — imported in `seta_env/environments/terminal_env.py:27`

## Test File
`test/test_utils.py`

## Dependencies
- `aiofiles` (already a package dependency)
- `tmp_path` pytest fixture for temp directories

## Functions Under Test

```python
# seta_env/utils/utils.py line 163

@asynccontextmanager
async def async_timer(stage_name: str, timings_dict: Dict[str, Dict[str, float]]):
    """
    Context manager. On exit writes to timings_dict[stage_name]:
        {"start": float, "end": float, "elapsed": float}
    Uses time.time(). Always writes even if body raises.
    """

# seta_env/utils/utils.py line 282

async def load_main_trajectory(log_dir: str) -> str:
    """
    Calls find_trajectory_files(log_dir), picks the largest JSON file,
    calls load_cleaned_trajectory() on it.
    Raises FileNotFoundError if no JSON files found (or fewer than 2).
    Returns cleaned trajectory string.
    """

# (internal, called by load_main_trajectory)
async def find_trajectory_files(log_dir: str) -> List[Path]:
    """
    Globs *.json in log_dir. If fewer than 2, checks one level of subdirs.
    Sorts by file size descending.
    Returns [] if still fewer than 2.
    """

# (internal, called by load_main_trajectory)
async def load_cleaned_trajectory(path: str, include_system: bool = False) -> str:
    """
    Reads JSON file. Parses messages list from data["request"]["messages"].
    Returns formatted string of [User], [Assistant], [Tool Call], [Tool Result] lines.
    Returns "Trajectory file not found." if path doesn't exist.
    Returns "Error cleaning trajectory: ..." on parse failure.
    """
```

## Test Cases

### `async_timer`

| Scenario | What to check |
|---|---|
| Normal usage | `timings["my_stage"]` has keys `start`, `end`, `elapsed`; all floats |
| Elapsed ≥ 0 | `elapsed >= 0` |
| Elapsed accuracy | Sleep 0.1s inside; `elapsed` between 0.05 and 0.5 (generous bounds) |
| Exception inside body | Body raises; `timings` still populated; exception propagates |
| Multiple stages | Two sequential `async_timer` blocks write two keys to same dict |

### `load_main_trajectory`

| Scenario | Setup | Expected |
|---|---|---|
| Valid dir with 2+ JSON files | Write 2 valid trajectory JSON files | Returns string containing `[User]` or `[Assistant]` |
| Picks largest file | Write small.json (1KB) and large.json (5KB) | Returns content from large.json |
| Empty dir | Empty tmp dir | raises `FileNotFoundError` |
| Dir with 1 JSON and no subdirs | 1 JSON file only | raises `FileNotFoundError` (find_trajectory_files returns []) |
| Dir doesn't exist | Non-existent path | raises `FileNotFoundError` |
| Valid trajectory JSON format | See structure below | Cleaned lines in output |

### Trajectory JSON Structure (for fixtures)

```json
{
  "request": {
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi", "tool_calls": [
        {"function": {"name": "shell_exec", "arguments": "{\"id\": \"s1\", \"command\": \"ls\"}"}}
      ]},
      {"role": "tool", "content": "file1.txt\nfile2.txt"}
    ]
  },
  "response": {
    "choices": [{"message": {"content": "Done"}}]
  }
}
```

Cleaned output for this fixture should contain:
- `[User]: Hello`
- `[Assistant]: Hi`
- `[Tool Call]: shell_exec(...)`
- `[Tool Result]: file1.txt`
- `[Final Response]: Done`

## Setup

```python
import pytest
import json
import asyncio
from pathlib import Path
from seta_env.utils.utils import async_timer, load_main_trajectory

# Use tmp_path fixture from pytest for temp directories
```

All tests `@pytest.mark.asyncio`.
