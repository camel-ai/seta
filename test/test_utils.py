import asyncio
import json
import pytest
from pathlib import Path

from seta_env.utils.utils import async_timer, load_main_trajectory


# ---------------------------------------------------------------------------
# async_timer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_timer_keys_written():
    timings = {}
    async with async_timer("stage1", timings):
        pass
    assert "stage1" in timings
    assert {"start", "end", "elapsed"} == set(timings["stage1"].keys())

@pytest.mark.asyncio
async def test_async_timer_all_floats():
    timings = {}
    async with async_timer("stage1", timings):
        pass
    for v in timings["stage1"].values():
        assert isinstance(v, float)

@pytest.mark.asyncio
async def test_async_timer_elapsed_accuracy():
    timings = {}
    async with async_timer("sleep_stage", timings):
        await asyncio.sleep(0.1)
    assert 0.05 < timings["sleep_stage"]["elapsed"] < 0.5

@pytest.mark.asyncio
async def test_async_timer_elapsed_nonnegative():
    timings = {}
    async with async_timer("fast", timings):
        pass
    assert timings["fast"]["elapsed"] >= 0

@pytest.mark.asyncio
async def test_async_timer_exception_still_records():
    timings = {}
    with pytest.raises(ValueError):
        async with async_timer("failing_stage", timings):
            raise ValueError("boom")
    assert "failing_stage" in timings
    assert timings["failing_stage"]["elapsed"] >= 0

@pytest.mark.asyncio
async def test_async_timer_multiple_stages():
    timings = {}
    async with async_timer("a", timings):
        pass
    async with async_timer("b", timings):
        pass
    assert "a" in timings and "b" in timings

@pytest.mark.asyncio
async def test_async_timer_start_lt_end():
    timings = {}
    async with async_timer("s", timings):
        await asyncio.sleep(0.01)
    assert timings["s"]["start"] < timings["s"]["end"]


# ---------------------------------------------------------------------------
# load_main_trajectory — helpers
# ---------------------------------------------------------------------------

def _write_trajectory(path: Path, size_bytes: int = 500):
    """Write a minimal valid trajectory JSON of approximately size_bytes."""
    messages = [
        {"role": "user", "content": "A" * max(10, size_bytes - 200)},
        {"role": "assistant", "content": "Hi", "tool_calls": [
            {"function": {"name": "shell_exec", "arguments": '{"id": "s1", "command": "ls"}'}}
        ]},
        {"role": "tool", "content": "file1.txt\nfile2.txt"},
    ]
    data = {
        "request": {"messages": messages},
        "response": {"choices": [{"message": {"content": "Done"}}]},
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# load_main_trajectory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_trajectory_returns_string(tmp_path):
    _write_trajectory(tmp_path / "a.json", size_bytes=600)
    _write_trajectory(tmp_path / "b.json", size_bytes=200)
    result = await load_main_trajectory(str(tmp_path))
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_load_trajectory_contains_cleaned_lines(tmp_path):
    _write_trajectory(tmp_path / "a.json", size_bytes=600)
    _write_trajectory(tmp_path / "b.json", size_bytes=200)
    result = await load_main_trajectory(str(tmp_path))
    assert "[User]:" in result
    assert "[Tool Call]: shell_exec" in result
    assert "[Tool Result]:" in result
    assert "[Final Response]: Done" in result

@pytest.mark.asyncio
async def test_load_trajectory_picks_largest(tmp_path):
    # small.json has short content, large.json has long unique content
    small = tmp_path / "small.json"
    large = tmp_path / "large.json"
    _write_trajectory(small, size_bytes=200)

    large_messages = [{"role": "user", "content": "UNIQUE_MARKER_XYZ" + "B" * 2000}]
    large.write_text(json.dumps({
        "request": {"messages": large_messages},
        "response": {"choices": [{"message": {"content": "big done"}}]},
    }))

    result = await load_main_trajectory(str(tmp_path))
    assert "UNIQUE_MARKER_XYZ" in result

@pytest.mark.asyncio
async def test_load_trajectory_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await load_main_trajectory(str(tmp_path))

@pytest.mark.asyncio
async def test_load_trajectory_single_file_raises(tmp_path):
    # Only 1 JSON file — find_trajectory_files returns [] (needs >= 2)
    _write_trajectory(tmp_path / "only.json")
    with pytest.raises(FileNotFoundError):
        await load_main_trajectory(str(tmp_path))

@pytest.mark.asyncio
async def test_load_trajectory_nonexistent_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await load_main_trajectory(str(tmp_path / "does_not_exist"))

@pytest.mark.asyncio
async def test_load_trajectory_finds_in_subdir(tmp_path):
    # If top level has < 2 files, checks one level of subdirs
    subdir = tmp_path / "sub"
    subdir.mkdir()
    _write_trajectory(subdir / "a.json", size_bytes=600)
    _write_trajectory(subdir / "b.json", size_bytes=200)
    result = await load_main_trajectory(str(tmp_path))
    assert "[User]:" in result
