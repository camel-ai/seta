"""GLM-4.7-Flash tool-call parser for the seta_env SGLang backend.

GLM-4.7-Flash emits tool calls in its native XML-pair format (see the model's
chat_template.jinja), NOT Hermes JSON and NOT DeepSeek-V4 DSML::

    <tool_call>shell_exec
    <arg_key>id</arg_key><arg_value>get_pkgs</arg_value>
    <arg_key>command</arg_key><arg_value>dpkg -l</arg_value>
    <arg_key>block</arg_key><arg_value>true</arg_value>
    </tool_call>

i.e. the function NAME is the text immediately after ``<tool_call>`` (up to the
first ``<arg_key>``), followed by ``<arg_key>K</arg_key><arg_value>V</arg_value>``
pairs. Ported from SGLang's ``srt/function_call/glm47_moe_detector.py`` (the same
format SGLang itself parses), reduced to the non-streaming ``parse()`` the
seta_env backend needs.

API mirrors ``HermesToolCallParser`` from sglang_miles_model so it drops straight
into ``SGLangModel(tool_call_parser=GLMToolCallParser())``.
"""

from __future__ import annotations

import json
import logging
import re

from seta_env.models.sglang_miles_model.tool_parser import (
    UNKNOWN_TOOL_NAME,
    ToolCallParseResult,
    ToolCallParser,
)

logger = logging.getLogger(__name__)


def _coerce_value(raw: str):
    """GLM arg values carry no type hints. Try JSON-decode (so 'true'->True,
    '42'->42, '[1,2]'->list); fall back to the raw string. Matches the
    DSML parser's string="false" behaviour and lets Strands validate downstream.
    """
    s = raw.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return raw  # keep original (unstripped) string on non-JSON values


class GLMToolCallParser(ToolCallParser):
    """Parser for GLM-4.7 ``<tool_call>name<arg_key>..</arg_key><arg_value>..`` format."""

    DEFAULT_BOT_TOKEN = "<tool_call>"
    DEFAULT_EOT_TOKEN = "</tool_call>"
    DEFAULT_THINK_START_TOKEN = "<think>"
    DEFAULT_THINK_END_TOKEN = "</think>"

    def __init__(
        self,
        bot_token: str = DEFAULT_BOT_TOKEN,
        eot_token: str = DEFAULT_EOT_TOKEN,
        think_start_token: str | None = DEFAULT_THINK_START_TOKEN,
        think_end_token: str | None = DEFAULT_THINK_END_TOKEN,
    ) -> None:
        self.bot_token = bot_token
        self.eot_token = eot_token
        self._block_pattern = re.compile(
            rf"{re.escape(bot_token)}\s*(.*?)\s*{re.escape(eot_token)}", re.DOTALL
        )
        self._arg_pattern = re.compile(
            r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.DOTALL
        )
        if think_start_token and think_end_token:
            self._think_pattern: re.Pattern[str] | None = re.compile(
                rf"{re.escape(think_start_token)}.*?{re.escape(think_end_token)}", re.DOTALL
            )
        else:
            self._think_pattern = None

    @property
    def message_separator(self) -> str:
        # GLM's chat template concatenates messages without an inter-message
        # newline (the `{%- ... -%}` whitespace-control strips it), so the
        # incremental-TITO glue is empty. Validate against apply_chat_template
        # before trusting token-level loss masks.
        return ""

    def parse(self, text: str) -> list[ToolCallParseResult]:
        # Drop reasoning blocks so we don't execute draft tool calls.
        if self._think_pattern:
            text = self._think_pattern.sub("", text)

        results: list[ToolCallParseResult] = []
        for i, block in enumerate(self._block_pattern.finditer(text)):
            content = block.group(1).strip()
            # function name = text before the first <arg_key> (or whole block if no args)
            first_arg = content.find("<arg_key>")
            name = (content if first_arg == -1 else content[:first_arg]).strip()
            if not name:
                results.append(
                    ToolCallParseResult(name=UNKNOWN_TOOL_NAME, input={}, raw=content, id=f"call_{i}")
                )
                continue
            args = {k.strip(): _coerce_value(v) for k, v in self._arg_pattern.findall(content)}
            results.append(ToolCallParseResult(name=name, input=args, raw=None, id=f"call_{i}"))
        return results
