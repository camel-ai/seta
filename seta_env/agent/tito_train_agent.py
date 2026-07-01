"""AgentTrain variant for the TITO model backend.

A deliberately small agent loop that:
  * keeps a plain OpenAI ``message_list`` (no camel memory / summarization),
  * honours the TITO append-only contract (one system + one user to start,
    then only assistant / tool messages are appended),
  * runs tools sequentially,
  * tracks status + termination reason in ``meta_info_record``.

The TITO model backend (``TITOChatModel``) owns the session-server interaction;
this agent just feeds it the cumulative ``message_list`` each turn and appends
the assistant + tool results back onto that list.
"""

import asyncio
import copy
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse
from camel.types.agents import ToolCallingRecord
from camel.utils.agent_context import set_current_agent_id

from seta_env.agent.train_agent import AgentTrain, TerminationReason


class AgentTrainTITO(AgentTrain):
    """AgentTrain wired to the TITO model backend with a plain message list."""

    def reset(self):
        """Reset the agent and TITO model session for a new episode."""
        super().reset()
        if hasattr(self, "model_backend") and hasattr(
            self.model_backend, "reset"
        ):
            self.model_backend.reset()

        self.message_list: List[Dict[str, Any]] = []

    def count_message_tokens(self) -> Optional[int]:
        """Server-reported prompt token count from the last turn (or None)."""
        return getattr(self.model_backend, "last_prompt_tokens", None)

    def _get_full_tool_schemas(self) -> List[Dict[str, Any]]:
        """Tool schemas with OpenAI strict mode disabled.

        DeepSeek-V4 + sglang tool-call parsing breaks under strict tool
        calls (the model stops mid-reasoning without emitting the call), so
        we drop the ``strict`` flag that camel's FunctionTool sets by default.
        """
        schemas = copy.deepcopy(super()._get_full_tool_schemas())
        for schema in schemas:
            schema.get("function", {}).pop("strict", None)
        return schemas

    # ------------------------------------------------------------------ #
    # message_list helpers (replace camel-memory recording)
    # ------------------------------------------------------------------ #
    def _record_tool_calling(
        self,
        func_name: str,
        args: Dict[str, Any],
        result: Any,
        tool_call_id: str,
        mask_output: bool = False,
        extra_content: Optional[Dict[str, Any]] = None,
    ) -> ToolCallingRecord:
        """Build a ToolCallingRecord (no camel memory, no message append).

        Overrides the parent so the reused ``_aexecute_tool`` does NOT write to
        camel memory. The tool-result message is appended to ``message_list`` by
        the loop in request order (important for parallel execution).
        """
        return ToolCallingRecord(
            tool_name=func_name,
            args=args,
            result=result,
            tool_call_id=tool_call_id,
        )

    @staticmethod
    def _tool_result_content(result: Any) -> str:
        """Stringify a tool result for an OpenAI ``tool`` message."""
        content = getattr(result, "text", None)
        if content is None:
            content = result if isinstance(result, str) else str(result)
        return content

    # ------------------------------------------------------------------ #
    # agent loop
    # ------------------------------------------------------------------ #
    async def _astep_non_streaming_task(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
    ) -> ChatAgentResponse:
        set_current_agent_id(self.agent_id)

        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        # Seed the list: append-only contract means exactly one (system?, user)
        # at the start of an episode; reset() must be called between episodes.
        assert len(self.message_list) == 0, (
            "message_list must be empty at step start (call reset() per episode)"
        )
        if self._original_system_message is not None:
            self.message_list.append(
                self._original_system_message.to_openai_system_message()
            )
        self.message_list.append(input_message.to_openai_user_message())

        # Trackers
        tool_call_records: List[ToolCallingRecord] = []
        step_token_usage = self._create_token_usage_tracker()
        iteration_count: int = 0
        prev_num_openai_messages: int = 0
        num_tokens: int = 0
        response = None

        self.meta_info_record = {
            "iteration_count": 0,
            "termination_reason": TerminationReason.NOT_SET,
            "max_parallel_tool_call": 0,
            "total_tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        while True:
            # 1. model response over the full cumulative message_list
            response = await self._aget_model_response(
                self.message_list,
                current_iteration=iteration_count,
                response_format=response_format,
                tool_schemas=self._get_full_tool_schemas(),
                prev_num_openai_messages=prev_num_openai_messages,
            )
            prev_num_openai_messages = len(self.message_list)
            iteration_count += 1
            self.meta_info_record["iteration_count"] = iteration_count

            # 2. append the assistant message (with tool_calls) — the append-only
            #    boundary the TITO model scans back to on the next turn, and the
            #    local record of this turn's output.
            assistant_msg = response.response.choices[0].message.model_dump(
                exclude_none=True
            )
            assistant_msg["role"] = "assistant"
            self.message_list.append(assistant_msg)

            # 3. usage tracking
            self._update_token_usage_tracker(step_token_usage, response.usage_dict)
            self.meta_info_record["prompt_tokens"] = step_token_usage["prompt_tokens"]
            self.meta_info_record["completion_tokens"] = step_token_usage["completion_tokens"]
            self.meta_info_record["total_tokens"] = step_token_usage["total_tokens"]

            # 4. token-limit termination (exact server count for context sent)
            num_tokens = (response.usage_dict or {}).get("prompt_tokens", 0) or 0
            if self._token_limit is not None and num_tokens > self._token_limit:
                self.termination_reason = TerminationReason.MAX_TOKENS_EXCEEDED
                self.meta_info_record["termination_reason"] = self.termination_reason
                return self._step_terminate(
                    num_tokens, tool_call_records, "max_tokens_exceeded"
                )

            tool_call_requests = response.tool_call_requests

            # 5. no tool calls -> model is done
            if not tool_call_requests:
                self.termination_reason = TerminationReason.TASK_FINISHED
                self.meta_info_record["termination_reason"] = self.termination_reason
                break

            # 6. status: largest tool batch seen this step
            self.meta_info_record["max_parallel_tool_call"] = max(
                self.meta_info_record["max_parallel_tool_call"],
                len(tool_call_requests),
            )

            # 7. parallel-tool cap. With max_parallel_tool_calls>0 we execute
            #    only the first `cap` calls (cap=1 => one tool call per turn) and
            #    reject the rest. The assistant message already recorded ALL
            #    tool_calls (step 2), so each rejected call still needs a tool
            #    result keyed by its tool_call_id to keep the chat-template
            #    pairing valid; the first [:cap] preserve original order.
            cap = getattr(self, "max_parallel_tool_calls", 0) or 0
            if cap > 0 and len(tool_call_requests) > cap:
                exec_requests = list(tool_call_requests[:cap])
                rejected_requests = list(tool_call_requests[cap:])
            else:
                exec_requests = list(tool_call_requests)
                rejected_requests = []

            # 8. execute the (capped) tool calls — parallel (asyncio.gather) if
            #    enabled, else sequential. Results are appended to message_list
            #    afterwards in REQUEST order (append-only contract; keeps tool
            #    results aligned with the assistant's tool_calls).
            if self.parallel_internal_tools and len(exec_requests) > 1:
                gathered = await asyncio.gather(
                    *(self._aexecute_tool(r) for r in exec_requests),
                    return_exceptions=True,
                )
                records = []
                for request, rec in zip(exec_requests, gathered):
                    if isinstance(rec, BaseException):
                        rec = ToolCallingRecord(
                            tool_name=request.tool_name, args=request.args,
                            result=f"Error: {type(rec).__name__}: {rec}",
                            tool_call_id=request.tool_call_id,
                        )
                    records.append(rec)
            else:
                records = [
                    await self._aexecute_tool(r) for r in exec_requests
                ]

            # stub results for rejected (over-cap) tool calls
            for request in rejected_requests:
                records.append(ToolCallingRecord(
                    tool_name=request.tool_name, args=request.args,
                    result=(
                        "Error: rejected — only one tool call per turn is "
                        "allowed. Issue tool calls one at a time and wait for "
                        "each result before the next."
                    ),
                    tool_call_id=request.tool_call_id,
                ))
                self.meta_info_record["rejected_parallel_tool_calls"] = (
                    self.meta_info_record.get(
                        "rejected_parallel_tool_calls", 0) + 1
                )

            for rec in records:
                self.message_list.append({
                    "role": "tool",
                    "tool_call_id": rec.tool_call_id,
                    "content": self._tool_result_content(rec.result),
                })
                tool_call_records.append(rec)
            self.meta_info_record["total_tool_calls"] = len(tool_call_records)

            # 8. max-iteration termination
            if (
                self.max_iteration is not None
                and iteration_count >= self.max_iteration
            ):
                self.termination_reason = TerminationReason.MAX_ITERATION_REACHED
                self.meta_info_record["termination_reason"] = self.termination_reason
                break

            # otherwise continue the loop

        return self._convert_to_chatagent_response(
            response,
            tool_call_records,
            num_tokens,
            None,  # no external-tool handling in this minimal loop
            step_token_usage["prompt_tokens"],
            step_token_usage["completion_tokens"],
            step_token_usage["total_tokens"],
        )
