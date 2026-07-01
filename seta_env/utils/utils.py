import asyncio
import json
import pathlib
import time
import traceback
from concurrent.futures import Executor
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from contextlib import asynccontextmanager

import aiofiles
import aiofiles.os
import os
import pandas as pd


@dataclass
class ErrorInfo:
    """Comprehensive error information for debugging failed trajectories."""
    error_type: str  # Exception class name (e.g., "TimeoutError", "ConnectionError")
    error_message: str  # The exception message
    failed_stage: str  # Which stage failed (e.g., "1_reset_env", "3_run_agent")
    traceback: str  # Full traceback string
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context (task_name, uid, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_log_string(self) -> str:
        """Format as a human-readable log string."""
        lines = [
            "=" * 60,
            f"ERROR REPORT - {self.timestamp}",
            "=" * 60,
            f"Failed Stage: {self.failed_stage}",
            f"Error Type: {self.error_type}",
            f"Error Message: {self.error_message}",
            "",
            "Context:",
        ]
        for key, value in self.context.items():
            lines.append(f"  {key}: {value}")
        lines.extend([
            "",
            "Traceback:",
            self.traceback,
            "=" * 60,
        ])
        return "\n".join(lines)


def capture_error(
    exception: Exception,
    failed_stage: str,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorInfo:
    """
    Capture comprehensive error information from an exception.

    Args:
        exception: The caught exception
        failed_stage: Which stage failed (e.g., "1_reset_env", "3_run_agent")
        context: Additional context like task_name, uid, remote_url, etc.

    Returns:
        ErrorInfo dataclass with all error details
    """
    return ErrorInfo(
        error_type=type(exception).__name__,
        error_message=str(exception),
        failed_stage=failed_stage,
        traceback=traceback.format_exc(),
        context=context or {},
    )


async def save_error_log(
    output_path: pathlib.Path,
    error_info: ErrorInfo,
) -> pathlib.Path:
    """
    Save error details to a dedicated error.log file in the trajectory folder.

    Args:
        output_path: Task-specific output directory path
        error_info: ErrorInfo object with error details

    Returns:
        Path to the saved error log file
    """
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save human-readable error.log
    error_log_path = output_path / "error.log"
    async with aiofiles.open(error_log_path, mode='w', encoding='utf-8') as f:
        await f.write(error_info.to_log_string())

    # Also save structured error.json for programmatic access
    error_json_path = output_path / "error.json"
    async with aiofiles.open(error_json_path, mode='w', encoding='utf-8') as f:
        await f.write(json.dumps(error_info.to_dict(), indent=2, default=str))

    print(f"  Error details saved to: {error_log_path}")
    return error_log_path


@dataclass
class ExecutorResult:
    """Result of running a function in an executor with timeout."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    error_type: Optional[str] = None


async def run_in_executor_with_timeout(
    executor: Executor,
    func: Callable,
    *args,
    timeout: float,
    **kwargs,
) -> ExecutorResult:
    """Run a function in an executor with timeout control and error capture.

    Args:
        executor: The executor to run the function in.
        func: The function to run.
        *args: Positional arguments to pass to the function.
        timeout: Timeout in seconds.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        ExecutorResult with success status, result or error information.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(func, *args, **kwargs)
            ),
            timeout=timeout
        )
        return ExecutorResult(success=True, result=result)
    except asyncio.TimeoutError as e:
        return ExecutorResult(
            success=False,
            error=e,
            error_type="TimeoutError"
        )
    except Exception as e:
        return ExecutorResult(
            success=False,
            error=e,
            error_type=type(e).__name__
        )

@asynccontextmanager
async def async_timer(
    stage_name: str,
    timings_dict: Dict[str, Dict[str, float]],
    timeout: float | None = None,
):
    """
    Async context manager for timing stages with start/end timestamps.

    Records start time, end time, and elapsed duration for Perfetto trace visualization.

    Usage:
        timings = {}
        async with async_timer("stage1", timings):
            await some_operation()
        # timings["stage1"] will contain: {"start": ts, "end": ts, "elapsed": duration}
    """
    start_time = time.time()
    raised_exc = None
    print(f"  ⏱️  ENTER {stage_name}: timeout={timeout}s", flush=True)
    try:
        if timeout is None:
            yield
        else:
            async with asyncio.timeout(timeout):
                yield
    except BaseException as e:
        raised_exc = e
        elapsed = time.time() - start_time
        print(
            f"  ⏱️  EXC in {stage_name} at elapsed={elapsed:.2f}s "
            f"(configured timeout={timeout}s): "
            f"{type(e).__name__}: {e!r}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        timings_dict[stage_name] = {
            "start": start_time,
            "end": end_time,
            "elapsed": elapsed
        }
        outcome = "OK" if raised_exc is None else f"FAIL({type(raised_exc).__name__})"
        print(f"  ⏱️  {stage_name}: {elapsed:.2f}s [{outcome}]", flush=True)


async def load_cleaned_trajectory(path: str, include_system: bool = False) -> str:
    """
    Load a trajectory JSON and return a thinned text representation.
    Strips redundant formatting and labels iterations to save tokens.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return "Trajectory file not found."
    
    try:
        async with aiofiles.open(p, mode='r', encoding='utf-8', errors='replace') as f:
            content = await f.read()
        data = json.loads(content)
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


async def find_trajectory_files(log_dir: str) -> List[pathlib.Path]:
    """
    Find trajectory files in a specific rollout folder.
    Returns: List of trajectory file paths sorted by size (descending).
    """
    log_dir_path = pathlib.Path(log_dir)
    if not log_dir_path.exists():
        return []

    json_files = list(log_dir_path.glob("*.json"))

    # If not enough JSON files, check one level deep in subdirectories
    if len(json_files) < 2:
        subdirs = [item for item in log_dir_path.iterdir() if item.is_dir()]
        for subdir in subdirs:
            subdir_json_files = list(subdir.glob("*.json"))
            if len(subdir_json_files) >= 2:
                json_files = subdir_json_files
                break

        if len(json_files) < 2:
            print(f"WARNING: Not enough JSON files in {log_dir_path}")
            return []

    # Sort by size, descending to find the main trajectory
    # Use async stat for file sizes - run in parallel for performance
    import asyncio
    stat_results = await asyncio.gather(*[aiofiles.os.stat(p) for p in json_files])
    file_sizes = [(p, stat_result.st_size) for p, stat_result in zip(json_files, stat_results)]

    file_sizes.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in file_sizes]


async def load_main_trajectory(log_dir: str) -> str:
    """
    Load the main trajectory file from a specific rollout folder.
    Returns: Cleaned trajectory content.
    """
    json_files = await find_trajectory_files(log_dir)
    if len(json_files) == 0:
        raise FileNotFoundError(f"No trajectory files found in {log_dir}")

    return await load_cleaned_trajectory(str(json_files[0]))

def load_data_from_taskdir(task_path: str):
    # load task_name, task_path, instruction from task_path
    task_name = os.path.basename(task_path)
    task_path = task_path

    files_to_check = {
        "task.yaml",
        "Dockerfile",
        "docker-compose.yaml",
        "run-tests.sh",
        "solution.sh"
    }
    for f in files_to_check:
        # if file not exist throw filenotfound error
        if not os.path.exists(os.path.join(task_path, f)):
            raise FileNotFoundError(f"Required file {f} not found in {task_path}")

    # load instruction from task.yaml
    import yaml
    with open(os.path.join(task_path, "task.yaml"), "r") as f:
        task_info = yaml.safe_load(f)
    instruction = task_info.get("instruction")
    return {"task_name": task_name, "task_path": task_path, "instruction": instruction}

def load_dataset_from_folder(data_folder: str):
    # iterate through
    dataset = []
    for d in os.listdir(data_folder):
        full_path = os.path.join(data_folder, d)
        if not os.path.isdir(full_path):
            continue
        try:
            data = load_data_from_taskdir(f"{data_folder}/{d}")
            dataset.append(data)
        except Exception as e:
            print(e)
            continue
    return dataset

def build_docker_image(task_path: str, timeout=1200.0):
    # --- Terminal Bench import
    from terminal_bench.handlers.trial_handler import TrialHandler
    from terminal_bench.terminal.docker_compose_manager import DockerComposeManager

    task_path = pathlib.Path(task_path)
    trial_handler = TrialHandler(
        trial_name=f"build_run",
        input_path=task_path,
        output_path=pathlib.Path("build_outputs"),
    )
    print(f"Task path: {task_path}")

    compose_manager = DockerComposeManager(
            client_container_name=trial_handler.client_container_name,
            client_image_name=trial_handler.client_image_name,
            docker_image_name_prefix=trial_handler.docker_image_name_prefix,
            docker_compose_path=trial_handler.task_paths.docker_compose_path,
            no_rebuild=True,
            cleanup=True,
            sessions_logs_path=trial_handler.trial_paths.sessions_path,
            agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
        )
    compose_manager.build(timeout=timeout)

def build_docker_images_parallel(task_paths: List[str], timeout=1200.0):
    # build in parallel with multiprocessing
    from multiprocessing import Pool
    with Pool(processes=4) as pool:
        pool.map(build_docker_image, [(task_path, timeout) for task_path in task_paths])


def extract_agent_summary(response, agent, elapsed_time: float, agent_type: str, status: str = "completed") -> dict:
    """
    Extract summary information from agent response.

    Args:
        response: Agent response object (can be None for timeout/error cases)
        agent: Agent instance (can be None for empty/oracle agents)
        elapsed_time: Time taken for agent execution
        agent_type: Type of agent (e.g., 'camel', 'empty', 'oracle')
        status: Status of the run ('completed', 'timeout', 'error')

    Returns:
        dict: Summary information including tokens, tool calls, termination reason, etc.
    """
    summary = {
        "agent_type": agent_type,
        "status": status,
        "elapsed_time": elapsed_time,
    }

    # Extract info from response if available
    if response and hasattr(response, 'info'):
        info = response.info

        # Extract tool call info
        tool_calls = info.get('tool_calls', [])
        summary["num_tool_calls"] = len(tool_calls)

        # Extract token usage
        if 'usage' in info:
            summary["prompt_tokens"] = info['usage'].get('prompt_tokens', 0)
            summary["completion_tokens"] = info['usage'].get('completion_tokens', 0)
            summary["total_tokens"] = info['usage'].get('total_tokens', 0)

    # Extract agent state info if agent is available
    if agent:
        # On timeout/error, try to get accumulated statistics from agent instance variables
        if response is None and status in ["timeout", "error"]:
            # Get token usage from accumulated values if available
            if hasattr(agent, 'accumulated_prompt_tokens'):
                summary["prompt_tokens"] = agent.accumulated_prompt_tokens
            if hasattr(agent, 'accumulated_completion_tokens'):
                summary["completion_tokens"] = agent.accumulated_completion_tokens
            if hasattr(agent, 'accumulated_total_tokens'):
                summary["total_tokens"] = agent.accumulated_total_tokens

            # Get tool call count from accumulated value if available
            if hasattr(agent, 'total_tool_calls'):
                summary["num_tool_calls"] = agent.total_tool_calls

        # Extract termination info if agent has it
        if hasattr(agent, 'termination_reason'):
            summary["termination_reason"] = agent.termination_reason.value

        # Extract parse error count if agent has it
        if hasattr(agent, 'parse_error_count'):
            summary["parse_error_count"] = agent.parse_error_count

        # Count iterations (number of model calls)
        if hasattr(agent, 'iteration_count'):
            summary["iteration_count"] = agent.iteration_count

    return summary


def display_agent_summary(summary: dict):
    """
    Display agent summary information in a formatted way.

    Args:
        summary: Summary dict from extract_agent_summary
    """
    if not summary:
        return

    # Display tool calls
    if "num_tool_calls" in summary:
        print(f"  Tool calls: {summary['num_tool_calls']}")

    # Display token usage
    if "total_tokens" in summary:
        print(f"  Tokens: {summary['total_tokens']} "
              f"(prompt: {summary.get('prompt_tokens', 0)}, "
              f"completion: {summary.get('completion_tokens', 0)})")

    # Display iterations
    if "iteration_count" in summary:
        print(f"  Iterations: {summary['iteration_count']}")

    # Display termination reason
    if "termination_reason" in summary:
        print(f"  Termination: {summary['termination_reason']}")

    # Display parse errors if any
    if summary.get("parse_error_count", 0) > 0:
        print(f"  Parse errors: {summary['parse_error_count']}")


async def save_task_results(
    output_path: pathlib.Path,
    task_name: str,
    uid: str,
    results: Dict[str, Any],
    reward: Optional[float] = None,
    agent_summary: Optional[Dict[str, Any]] = None,
    traj_i: int = 0,
    stage_timings: Optional[Dict[str, float]] = None,
    error_info: Optional[ErrorInfo] = None,
) -> pathlib.Path:
    """
    Save task execution results to a JSON file in the task-specific output directory.

    Args:
        output_path: Task-specific output directory path
        task_name: Name of the task
        uid: Unique identifier for this run
        results: Evaluation results dict (status, pass_ratio, test_results, etc.)
        reward: Calculated reward (optional)
        agent_summary: Agent summary dict from extract_agent_summary (optional)
        traj_i: Trajectory index (default: 0)
        stage_timings: Dict of stage names to elapsed times in seconds (optional)
        error_info: ErrorInfo object with error details if trajectory failed (optional)

    Returns:
        Path to the saved results file
    """
    # Ensure output_path is a Path object
    output_path = pathlib.Path(output_path)

    # Create the results dict
    task_results = {
        "task_name": task_name,
        "uid": uid,
        "traj_i": traj_i,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "reward": reward,
        "agent_summary": agent_summary or {},
        "stage_timings": stage_timings or {},
        "error_info": error_info.to_dict() if error_info else None,
    }

    # Save to file
    results_file = output_path / "task_results.json"
    async with aiofiles.open(results_file, mode='w', encoding='utf-8') as f:
        await f.write(json.dumps(task_results, indent=2, default=str))

    print(f"Results saved to: {results_file}")
    return results_file


async def collect_all_results(
    output_root_dir: pathlib.Path,
    pattern: str = "task_results.json"
) -> Dict[str, Any]:
    """
    Collect all task results from an output root directory.

    Args:
        output_root_dir: Root directory containing task-specific output subdirectories
        pattern: Filename pattern to search for (default: "task_results.json")

    Returns:
        Dict containing:
            - 'results': List of all task results
            - 'summary': Aggregated statistics
            - 'by_task': Results grouped by task name
    """
    output_root_dir = pathlib.Path(output_root_dir)

    if not output_root_dir.exists():
        raise FileNotFoundError(f"Output root directory not found: {output_root_dir}")

    # Find all result files
    result_files = list(output_root_dir.glob(f"**/{pattern}"))

    print(f"Found {len(result_files)} result files in {output_root_dir}")

    # Load all results
    all_results = []
    for result_file in result_files:
        try:
            async with aiofiles.open(result_file, mode='r', encoding='utf-8') as f:
                content = await f.read()
            result_data = json.loads(content)
            result_data['_file_path'] = str(result_file)
            all_results.append(result_data)
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
            continue

    # Group by task name
    by_task = {}
    for result in all_results:
        task_name = result.get('task_name', 'unknown')
        if task_name not in by_task:
            by_task[task_name] = []
        by_task[task_name].append(result)

    # Compute summary statistics
    summary = _compute_summary_statistics(all_results)

    return {
        'results': all_results,
        'by_task': by_task,
        'summary': summary,
        'total_count': len(all_results),
    }


def _compute_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics from collected results.

    Args:
        results: List of task result dicts

    Returns:
        Dict containing summary statistics
    """
    if not results:
        return {}

    # Extract metrics
    pass_ratios = [r.get('results', {}).get('pass_ratio', 0.0) for r in results]
    rewards = [r.get('reward') for r in results if r.get('reward') is not None]
    statuses = [r.get('results', {}).get('status', 'unknown') for r in results]

    # Agent-specific metrics
    elapsed_times = []
    total_tokens = []
    num_tool_calls = []

    # Stage timing metrics
    stage_timing_lists = {
        '1_reset_env': [],
        '2_reset_agent': [],
        '3_run_agent': [],
        '4_evaluation': [],
        '5_calculate_reward': [],
        '6_cleanup': [],
        'total': [],
    }

    for r in results:
        agent_summary = r.get('agent_summary', {})
        if 'elapsed_time' in agent_summary:
            elapsed_times.append(agent_summary['elapsed_time'])
        if 'total_tokens' in agent_summary:
            total_tokens.append(agent_summary['total_tokens'])
        if 'num_tool_calls' in agent_summary:
            num_tool_calls.append(agent_summary['num_tool_calls'])

        # Extract stage timings
        stage_timings = r.get('stage_timings', {})
        if stage_timings:
            for stage_key in ['1_reset_env', '2_reset_agent', '3_run_agent', '4_evaluation', '5_calculate_reward', '6_cleanup']:
                if stage_key in stage_timings:
                    # Extract elapsed time from timing dict
                    timing_data = stage_timings[stage_key]
                    if isinstance(timing_data, dict):
                        elapsed = timing_data.get('elapsed', 0)
                    else:
                        # Backward compatibility: if it's just a number
                        elapsed = timing_data
                    stage_timing_lists[stage_key].append(elapsed)
            # Calculate total
            total = sum(timing_data.get('elapsed', 0) if isinstance(timing_data, dict) else timing_data
                       for timing_data in stage_timings.values())
            stage_timing_lists['total'].append(total)

    # Count failures by stage and error type
    failed_stages = [r.get('error_info', {}).get('failed_stage') for r in results if r.get('error_info')]
    error_types = [r.get('error_info', {}).get('error_type') for r in results if r.get('error_info')]
    failed_count = sum(1 for r in results if r.get('error_info'))

    summary = {
        'total_tasks': len(results),
        'pass_ratio': {
            'mean': sum(pass_ratios) / len(pass_ratios) if pass_ratios else 0.0,
            'min': min(pass_ratios) if pass_ratios else 0.0,
            'max': max(pass_ratios) if pass_ratios else 0.0,
        },
        'status_counts': {status: statuses.count(status) for status in set(statuses)},
        'failed_count': failed_count,
        'failure_rate': failed_count / len(results) if results else 0.0,
        'failures_by_stage': {stage: failed_stages.count(stage) for stage in set(failed_stages) if stage},
        'failures_by_error_type': {etype: error_types.count(etype) for etype in set(error_types) if etype},
    }

    # Add reward stats if available
    if rewards:
        summary['reward'] = {
            'mean': sum(rewards) / len(rewards),
            'min': min(rewards),
            'max': max(rewards),
            'count': len(rewards),
        }

    # Add agent stats if available
    if elapsed_times:
        summary['elapsed_time'] = {
            'mean': sum(elapsed_times) / len(elapsed_times),
            'min': min(elapsed_times),
            'max': max(elapsed_times),
        }

    if total_tokens:
        summary['total_tokens'] = {
            'mean': sum(total_tokens) / len(total_tokens),
            'min': min(total_tokens),
            'max': max(total_tokens),
            'sum': sum(total_tokens),
        }

    if num_tool_calls:
        summary['num_tool_calls'] = {
            'mean': sum(num_tool_calls) / len(num_tool_calls),
            'min': min(num_tool_calls),
            'max': max(num_tool_calls),
        }

    # Add stage timing stats
    summary['stage_timings'] = {}
    for stage_key, times in stage_timing_lists.items():
        if times:
            summary['stage_timings'][stage_key] = {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
            }

    return summary


def print_results_summary(collected_results: Dict[str, Any]):
    """
    Print a formatted summary of collected results.

    Args:
        collected_results: Output from collect_all_results()
    """
    summary = collected_results.get('summary', {})
    total_count = collected_results.get('total_count', 0)
    by_task = collected_results.get('by_task', {})

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\nTotal tasks: {total_count}")

    # Pass ratio stats
    if 'pass_ratio' in summary:
        pr = summary['pass_ratio']
        print(f"\nPass Ratio:")
        print(f"  Mean: {pr['mean']:.2%}")
        print(f"  Min:  {pr['min']:.2%}")
        print(f"  Max:  {pr['max']:.2%}")

    # Status counts
    if 'status_counts' in summary:
        print(f"\nStatus Counts:")
        for status, count in summary['status_counts'].items():
            print(f"  {status}: {count}")

    # Failure statistics
    if summary.get('failed_count', 0) > 0:
        print(f"\nFailure Statistics:")
        print(f"  Failed: {summary['failed_count']} ({summary.get('failure_rate', 0):.1%})")

        if summary.get('failures_by_stage'):
            print(f"  By Stage:")
            for stage, count in sorted(summary['failures_by_stage'].items()):
                print(f"    {stage}: {count}")

        if summary.get('failures_by_error_type'):
            print(f"  By Error Type:")
            for etype, count in sorted(summary['failures_by_error_type'].items()):
                print(f"    {etype}: {count}")

    # Reward stats
    if 'reward' in summary:
        rw = summary['reward']
        print(f"\nReward:")
        print(f"  Mean: {rw['mean']:.4f}")
        print(f"  Min:  {rw['min']:.4f}")
        print(f"  Max:  {rw['max']:.4f}")

    # Agent stats
    if 'elapsed_time' in summary:
        et = summary['elapsed_time']
        print(f"\nElapsed Time:")
        print(f"  Mean: {et['mean']:.1f}s")
        print(f"  Min:  {et['min']:.1f}s")
        print(f"  Max:  {et['max']:.1f}s")

    if 'total_tokens' in summary:
        tt = summary['total_tokens']
        print(f"\nToken Usage:")
        print(f"  Mean: {tt['mean']:.0f}")
        print(f"  Total: {tt['sum']:.0f}")

    # Stage timings breakdown
    if 'stage_timings' in summary and summary['stage_timings']:
        print(f"\nStage Timings (seconds):")
        stage_names = {
            '1_reset_env': 'Reset Env',
            '2_reset_agent': 'Reset Agent',
            '3_run_agent': 'Run Agent',
            '4_evaluation': 'Evaluation',
            '5_calculate_reward': 'Calculate Reward',
            '6_cleanup': 'Cleanup',
            'total': 'Total',
        }
        for stage_key, times in summary['stage_timings'].items():
            stage_label = stage_names.get(stage_key, stage_key)
            print(f"  {stage_label:20} - Mean: {times['mean']:6.2f}s  Min: {times['min']:6.2f}s  Max: {times['max']:6.2f}s")

    # Per-task breakdown
    print(f"\nResults by Task ({len(by_task)} unique tasks):")
    for task_name, task_results in sorted(by_task.items()):
        task_pass_ratios = [r.get('results', {}).get('pass_ratio', 0.0) for r in task_results]
        mean_pass_ratio = sum(task_pass_ratios) / len(task_pass_ratios) if task_pass_ratios else 0.0
        print(f"  {task_name}: {len(task_results)} runs, avg pass ratio: {mean_pass_ratio:.2%}")

    print("="*60)


def results_to_dataframe(collected_results: Dict[str, Any]):
    """
    Convert collected results to a pandas DataFrame.

    Args:
        collected_results: Output from collect_all_results()

    Returns:
        pandas.DataFrame with flattened results

    Raises:
        ImportError: If pandas is not installed
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to convert results to DataFrame. "
            "Install it with: pip install pandas"
        )

    rows = []
    for result in collected_results['results']:
        # Extract agent summary fields
        agent_summary = result.get('agent_summary', {})

        # Extract test results summary
        test_results = result.get('results', {}).get('test_results', {})
        num_tests_passed = sum(1 for v in test_results.values() if v) if test_results else 0
        num_tests_total = len(test_results) if test_results else 0

        # Extract stage timings
        stage_timings = result.get('stage_timings', {})

        row = {
            # Task metadata
            'task_name': result.get('task_name'),
            'uid': result.get('uid'),
            'traj_i': result.get('traj_i'),
            'timestamp': result.get('timestamp'),

            # Results
            'status': result.get('results', {}).get('status'),
            'pass_ratio': result.get('results', {}).get('pass_ratio'),
            'all_passed': result.get('results', {}).get('all_passed'),
            'num_tests_passed': num_tests_passed,
            'num_tests_total': num_tests_total,
            'reward': result.get('reward'),

            # Agent summary - basic info
            'agent_type': agent_summary.get('agent_type'),
            'agent_status': agent_summary.get('status'),
            'elapsed_time': agent_summary.get('elapsed_time'),

            # Agent summary - token usage
            'total_tokens': agent_summary.get('total_tokens'),
            'prompt_tokens': agent_summary.get('prompt_tokens'),
            'completion_tokens': agent_summary.get('completion_tokens'),

            # Agent summary - tool calls and iterations
            'num_tool_calls': agent_summary.get('num_tool_calls'),
            'iteration_count': agent_summary.get('iteration_count'),
            'parse_error_count': agent_summary.get('parse_error_count'),

            # Agent summary - termination
            'termination_reason': agent_summary.get('termination_reason'),
            'timeout_sec': agent_summary.get('timeout_sec'),

            # Stage timings (extract elapsed time from timing dicts)
            'time_reset_env': stage_timings.get('1_reset_env', {}).get('elapsed') if isinstance(stage_timings.get('1_reset_env'), dict) else stage_timings.get('1_reset_env'),
            'time_reset_agent': stage_timings.get('2_reset_agent', {}).get('elapsed') if isinstance(stage_timings.get('2_reset_agent'), dict) else stage_timings.get('2_reset_agent'),
            'time_run_agent': stage_timings.get('3_run_agent', {}).get('elapsed') if isinstance(stage_timings.get('3_run_agent'), dict) else stage_timings.get('3_run_agent'),
            'time_evaluation': stage_timings.get('4_evaluation', {}).get('elapsed') if isinstance(stage_timings.get('4_evaluation'), dict) else stage_timings.get('4_evaluation'),
            'time_calculate_reward': stage_timings.get('5_calculate_reward', {}).get('elapsed') if isinstance(stage_timings.get('5_calculate_reward'), dict) else stage_timings.get('5_calculate_reward'),
            'time_cleanup': stage_timings.get('6_cleanup', {}).get('elapsed') if isinstance(stage_timings.get('6_cleanup'), dict) else stage_timings.get('6_cleanup'),
            'time_total': sum(t.get('elapsed', 0) if isinstance(t, dict) else t for t in stage_timings.values()) if stage_timings else None,

            # Stage timing timestamps (for Perfetto visualization)
            'time_reset_env_start': stage_timings.get('1_reset_env', {}).get('start') if isinstance(stage_timings.get('1_reset_env'), dict) else None,
            'time_reset_env_end': stage_timings.get('1_reset_env', {}).get('end') if isinstance(stage_timings.get('1_reset_env'), dict) else None,
            'time_reset_agent_start': stage_timings.get('2_reset_agent', {}).get('start') if isinstance(stage_timings.get('2_reset_agent'), dict) else None,
            'time_reset_agent_end': stage_timings.get('2_reset_agent', {}).get('end') if isinstance(stage_timings.get('2_reset_agent'), dict) else None,
            'time_run_agent_start': stage_timings.get('3_run_agent', {}).get('start') if isinstance(stage_timings.get('3_run_agent'), dict) else None,
            'time_run_agent_end': stage_timings.get('3_run_agent', {}).get('end') if isinstance(stage_timings.get('3_run_agent'), dict) else None,
            'time_evaluation_start': stage_timings.get('4_evaluation', {}).get('start') if isinstance(stage_timings.get('4_evaluation'), dict) else None,
            'time_evaluation_end': stage_timings.get('4_evaluation', {}).get('end') if isinstance(stage_timings.get('4_evaluation'), dict) else None,
            'time_calculate_reward_start': stage_timings.get('5_calculate_reward', {}).get('start') if isinstance(stage_timings.get('5_calculate_reward'), dict) else None,
            'time_calculate_reward_end': stage_timings.get('5_calculate_reward', {}).get('end') if isinstance(stage_timings.get('5_calculate_reward'), dict) else None,
            'time_cleanup_start': stage_timings.get('6_cleanup', {}).get('start') if isinstance(stage_timings.get('6_cleanup'), dict) else None,
            'time_cleanup_end': stage_timings.get('6_cleanup', {}).get('end') if isinstance(stage_timings.get('6_cleanup'), dict) else None,

            # Error info - enhanced with stage and type
            'error': result.get('results', {}).get('error') or agent_summary.get('error'),
            'error_type': result.get('results', {}).get('error_type') or (result.get('error_info', {}) or {}).get('error_type'),
            'failed_stage': result.get('results', {}).get('failed_stage') or (result.get('error_info', {}) or {}).get('failed_stage'),
            'has_error': result.get('error_info') is not None,

            # File path for reference
            'file_path': result.get('_file_path'),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by task_name and timestamp for better readability
    if not df.empty:
        df = df.sort_values(['task_name', 'timestamp']).reset_index(drop=True)

    return df


async def collect_failed_trajectories(
    output_root_dir: pathlib.Path,
) -> Dict[str, Any]:
    """
    Collect all failed trajectories from an output root directory.

    This is useful for quickly identifying which trajectories failed and why,
    without having to parse through all results.

    Args:
        output_root_dir: Root directory containing task-specific output subdirectories

    Returns:
        Dict containing:
            - 'failed_count': Number of failed trajectories
            - 'total_count': Total number of trajectories
            - 'failures': List of failure details with task_name, uid, stage, error
            - 'by_stage': Failures grouped by which stage failed
            - 'by_error_type': Failures grouped by error type
    """
    output_root_dir = pathlib.Path(output_root_dir)

    if not output_root_dir.exists():
        raise FileNotFoundError(f"Output root directory not found: {output_root_dir}")

    # Find all error.json files
    error_files = list(output_root_dir.glob("**/error.json"))

    failures = []
    by_stage = {}
    by_error_type = {}

    for error_file in error_files:
        try:
            async with aiofiles.open(error_file, mode='r', encoding='utf-8') as f:
                content = await f.read()
            error_data = json.loads(content)

            failure_info = {
                "task_name": error_data.get("context", {}).get("task_name", "unknown"),
                "uid": error_data.get("context", {}).get("uid", "unknown"),
                "traj_i": error_data.get("context", {}).get("traj_i", 0),
                "failed_stage": error_data.get("failed_stage", "unknown"),
                "error_type": error_data.get("error_type", "unknown"),
                "error_message": error_data.get("error_message", "unknown"),
                "timestamp": error_data.get("timestamp"),
                "error_file": str(error_file),
            }
            failures.append(failure_info)

            # Group by stage
            stage = failure_info["failed_stage"]
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(failure_info)

            # Group by error type
            error_type = failure_info["error_type"]
            if error_type not in by_error_type:
                by_error_type[error_type] = []
            by_error_type[error_type].append(failure_info)

        except Exception as e:
            print(f"Warning: Failed to load {error_file}: {e}")
            continue

    # Also count total trajectories
    result_files = list(output_root_dir.glob("**/task_results.json"))

    return {
        "failed_count": len(failures),
        "total_count": len(result_files),
        "failures": failures,
        "by_stage": by_stage,
        "by_error_type": by_error_type,
    }


def print_failed_trajectories_summary(failed_results: Dict[str, Any]):
    """
    Print a formatted summary of failed trajectories.

    Args:
        failed_results: Output from collect_failed_trajectories()
    """
    print("\n" + "=" * 60)
    print("FAILED TRAJECTORIES SUMMARY")
    print("=" * 60)

    print(f"\nFailed: {failed_results['failed_count']} / {failed_results['total_count']} trajectories")

    # By stage
    if failed_results.get("by_stage"):
        print("\nFailures by Stage:")
        for stage, failures in sorted(failed_results["by_stage"].items()):
            print(f"  {stage}: {len(failures)}")

    # By error type
    if failed_results.get("by_error_type"):
        print("\nFailures by Error Type:")
        for error_type, failures in sorted(failed_results["by_error_type"].items()):
            print(f"  {error_type}: {len(failures)}")

    # Detailed list
    if failed_results.get("failures"):
        print("\nDetailed Failures:")
        for f in failed_results["failures"]:
            print(f"\n  Task: {f['task_name']}")
            print(f"  UID: {f['uid']}, Traj: {f['traj_i']}")
            print(f"  Stage: {f['failed_stage']}")
            print(f"  Error: {f['error_type']}: {f['error_message'][:100]}...")
            print(f"  Log: {f['error_file']}")

    print("=" * 60)


def export_to_perfetto_trace(collected_results: Dict[str, Any], output_path: str = "trace.json"):
    """
    Export stage timings to Perfetto-compatible trace format.

    Args:
        collected_results: Output from collect_all_results()
        output_path: Path to write the trace JSON file

    Returns:
        Path to the generated trace file
    """
    import json

    events = []

    for result in collected_results['results']:
        stage_timings = result.get('stage_timings', {})
        task_name = result.get('task_name', 'unknown')
        uid = result.get('uid', 'unknown')
        traj_i = result.get('traj_i', 0)

        # Create a unique process/thread ID for this trajectory
        pid = f"{task_name}"
        tid = f"traj_{traj_i}_{uid}"

        # Stage display names
        stage_names = {
            '1_reset_env': 'Reset Env',
            '2_reset_agent': 'Reset Agent',
            '3_run_agent': 'Run Agent',
            '4_evaluation': 'Evaluation',
            '5_calculate_reward': 'Calculate Reward',
            '6_cleanup': 'Cleanup',
        }

        # Add trace events for each stage
        for stage_key, stage_name in stage_names.items():
            timing = stage_timings.get(stage_key)
            if not timing or not isinstance(timing, dict):
                continue

            start_ts = timing.get('start', 0) * 1_000_000  # Convert to microseconds
            duration_us = timing.get('elapsed', 0) * 1_000_000

            # Add duration event
            events.append({
                "name": stage_name,
                "cat": "stage",
                "ph": "X",  # Complete event (duration)
                "ts": start_ts,
                "dur": duration_us,
                "pid": pid,
                "tid": tid,
                "args": {
                    "task": task_name,
                    "uid": uid,
                    "traj_i": traj_i,
                }
            })

    # Perfetto trace format
    trace = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
    }

    with open(output_path, 'w') as f:
        json.dump(trace, f, indent=2)

    print(f"Perfetto trace exported to: {output_path}")
    print(f"View at: https://ui.perfetto.dev/ (open {output_path})")

    return output_path
