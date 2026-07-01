# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strands hook for limiting tool iterations within a single agent invocation.

A "tool iteration" is one model response that requests tools, followed by tool execution.
This limiter stops the agent loop cleanly after N complete iterations.
"""

import logging

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import MessageAddedEvent

logger = logging.getLogger(__name__)


class MaxToolIterationsReachedError(Exception):
    """Raised when `max_iterations` limit is reached.

    This exception is raised after a complete iteration (model response + tool execution),
    ensuring the trajectory is clean without requiring truncation.
    """

    pass


class ToolIterationLimiter(HookProvider):
    """Hook to enforce `max_iterations` limit on agent tool loops.

    An "iteration" is one cycle of: model generates tool call → tool executes → result returned.
    Multiple parallel tool calls in one model response count as a single iteration.

    The limiter raises MaxIterationsReachedError AFTER the iteration completes (on tool result),
    ensuring a clean trajectory without requiring token truncation.

    Example:
        >>> limiter = ToolIterationLimiter(max_iterations=5)
        >>> agent = Agent(model=model, tools=[...], hooks=[limiter])
        >>> try:
        ...     result = agent.invoke("solve this problem")
        ... except MaxToolIterationsReachedError:
        ...     # Trajectory is clean - contains exactly 5 complete iterations
        ...     print(f"Stopped after {limiter.iteration_count} iterations")
    """

    def __init__(self, max_iterations: int | None = None):
        """Initialize the limiter.

        Args:
            max_iterations: Maximum number of iterations allowed.
                None means no limit.
        """
        self.max_iterations = max_iterations
        self.reset()

    def reset(self) -> None:
        """Reset counters for a new invocation."""
        self.iteration_count = 0

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register hooks with the strands agent."""
        registry.add_callback(MessageAddedEvent, self._on_message_added)

    def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Count iterations and raise when limit exceeded.

        - Counts on assistant messages with toolUse (model requesting tools)
        - Raises on user messages with toolResult (iteration complete)
        """
        if self.max_iterations is None:
            return

        message = event.message
        content = message.get("content", [])

        if not isinstance(content, list):
            return

        # Count when model requests tools
        if message.get("role") == "assistant":
            if any(c.get("toolUse") for c in content):
                self.iteration_count += 1
                logger.debug(f"Iteration {self.iteration_count} started (tool use detected)")

        # Check limit when tool result arrives (iteration complete)
        elif message.get("role") == "user":
            if any(c.get("toolResult") for c in content):
                if self.iteration_count >= self.max_iterations:
                    logger.debug(f"Max iterations ({self.max_iterations}) reached, stopping")
                    raise MaxToolIterationsReachedError(f"Max iterations ({self.max_iterations}) reached")
