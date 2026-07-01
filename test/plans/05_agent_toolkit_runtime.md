# Plan 05 — AgentTrain + TerminalToolkit + Runtime

## Source
`seta_env/agent/train_agent.py`

## Test File
`test/test_agent.py`

## Dependencies
- Docker runtime (from Plan 03)
- `TerminalToolkit` (from Plan 04)
- Claude API key: `ANTHROPIC_API_KEY` env var
- `camel-ai` package installed with Claude model support

## Classes Under Test

```python
# seta_env/agent/train_agent.py

class TerminationReason(Enum):
    MAX_PARSE_ERRORS = "max_parse_errors"
    MAX_ITERATION_REACHED = "max_iteration_reached"
    STEP_TIMEOUT = "step_timeout"
    MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"
    TASK_FINISHED = "task_finished"
    COMPLETION_LENGTH_EXCEEDED = "completion_length_exceeded"
    UNKNOWN_ERROR = "unknown_error"
    NOT_SET = "not_set"

class AgentTrain(ChatAgent):
    def __init__(self, task_name: str, *args, **kwargs):
        # Important attributes after init:
        self.max_parse_errors        # int, default 10
        self.parse_error_count       # int = 0
        self.task_name               # str
        self.termination_reason      # TerminationReason.NOT_SET
        self.parse_error_checker     # parse_error_check instance
        self.meta_info_record = {
            "iteration_count": 0,
            "termination_reason": TerminationReason.NOT_SET,
            "max_parallel_tool_call": 0,
            "parse_error_count": 0,
            "total_tool_calls": 0,
        }

    def reset(self):
        """Calls super().reset(), resets parse_error_checker and termination_reason"""

    async def _astep_non_streaming_task(self, input_message, response_format=None) -> ChatAgentResponse:
        """Core agent loop. Runs until: tool call leads to response, no tool call,
        max_iteration reached, or terminator triggered."""
```

### How `AgentTrain` is created in production (`terminal_env.py` line 266):
```python
agent = CamelAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Developer Agent",
        content=system_message,
    ),
    model=model,              # Claude model backend
    tools=tools,              # FunctionTool list from toolkit + note_toolkit
    token_limit=token_limit,
    max_iteration=max_iteration,
    task_name=task_name,
)
agent.reset()
```

### Running the agent (`terminal_env.py` line 217):
```python
response = await self.agent.astep(self.task.get("instruction"))
# agent_summary = agent.meta_info_record
```

## Model Setup (Claude API)

```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.ANTHROPIC,
    model_type=ModelType.CLAUDE_SONNET_4_6,  # or CLAUDE_HAIKU for cheaper tests
    # ANTHROPIC_API_KEY must be set in environment
)
```

## Fixtures

```python
@pytest.fixture
async def runtime_with_toolkit(task_dir, tmp_path):
    """Start docker runtime, return (runtime, toolkit, tools)."""
    rt = DockerHarborRuntime(
        task_dir=str(task_dir),
        trial_root=str(tmp_path / "trials"),
        session_id=f"agent_test_{uuid.uuid4().hex[:8]}",
        environment_type="docker",
    )
    await rt.reset()
    tools = await rt.get_tools()
    yield rt, rt.terminal_toolkit, tools
    await rt.stop(delete=True)

@pytest.fixture
def claude_model():
    """Claude Haiku model for cheap tests."""
    return ModelFactory.create(
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type="claude-haiku-4-5-20251001",
    )

@pytest.fixture
def agent(claude_model, runtime_with_toolkit):
    _, _, tools = runtime_with_toolkit
    a = AgentTrain(
        task_name="test_task",
        system_message=BaseMessage.make_assistant_message(
            role_name="Developer Agent",
            content="You are a developer agent. Use terminal tools to complete tasks.",
        ),
        model=claude_model,
        tools=tools,
        token_limit=8000,
        max_iteration=5,
    )
    a.reset()
    return a
```

## Test Cases

### `reset()`

| Scenario | Expected |
|---|---|
| `parse_error_checker.parse_error_count` after reset | `0` |
| `termination_reason` after reset | `TerminationReason.NOT_SET` |
| `meta_info_record["iteration_count"]` after reset | `0` |

### Single-step: agent calls a tool

Instruction: `"Run the command: echo hello_world and tell me the output."`

| Check | Expected |
|---|---|
| `response` returned without exception | True |
| `response.output_messages` non-empty | True |
| `shell_exec` was called | `terminal.log` contains `hello_world` |

### Multi-step task: create a file

Instruction: `"Create a file at /workdir/test_output.txt with content 'hello from agent'. Then verify it exists by reading it."`

| Check | Expected |
|---|---|
| No exception | True |
| File created inside runtime | `runtime.exec("cat /workdir/test_output.txt")` → `"hello from agent"` |
| `meta_info_record["total_tool_calls"]` | `>= 1` |
| `meta_info_record["iteration_count"]` | `>= 1` |

### Termination: `MAX_ITERATION_REACHED`

Create agent with `max_iteration=2`. Give an instruction requiring more than 2 steps.

| Check | Expected |
|---|---|
| Agent stops after 2 iterations | `meta_info_record["iteration_count"] <= 2` |
| No infinite loop | Returns within reasonable time |

### Tool filtering via tool name list

Create agent with only `["shell_exec"]` in tool list (filter from full tools list).
Verify agent cannot call `shell_view` (it's not in tools list).

| Check | Expected |
|---|---|
| Agent has 1 tool | `len(agent.tools) == 1` |
| Agent does not hallucinate calls to unlisted tools | Response does not show `shell_view` calls |

### `meta_info_record` after a completed run

After any completed `astep()`:

| Key | Expected |
|---|---|
| `iteration_count` | int > 0 |
| `termination_reason` | one of `TerminationReason.*` values |
| `parse_error_count` | int >= 0 |
| `total_tool_calls` | int >= 0 |

## Notes

- Use Claude Haiku (`claude-haiku-4-5-20251001`) to minimize API cost in tests.
- `astep()` is the public entry point (from `ChatAgent`); it internally calls `_astep_non_streaming_task`.
- Tests with live Docker + Claude API can take 30–120 seconds. Set pytest timeout accordingly.
- `ANTHROPIC_API_KEY` must be set or tests will fail at model creation.
