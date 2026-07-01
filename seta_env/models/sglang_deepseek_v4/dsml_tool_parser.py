"""DSML tool-call parser for DeepSeek-V4 models.

DeepSeek-V4 emits tool calls in its native DSML format (FULLWIDTH VERTICAL LINE
sentinels), not Hermes XML. We parse it directly so SGLang doesn't need
``--tool-call-parser deepseek`` set (which would diverge from miles' default
SGLang launch flags in ``run_deepseek_v4.py``).

Format (verified against an actual /v1/chat/completions response, see
``/tmp/harbor/trials/stage2_smoke/.../conv_*.json``)::

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="shell_exec">
    <｜DSML｜parameter name="id" string="true">get_packages</｜DSML｜parameter>
    <｜DSML｜parameter name="command" string="true">dpkg-query -Wf '${Package}\\n'</｜DSML｜parameter>
    <｜DSML｜parameter name="block" string="false">true</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

The sentinel character ``｜`` is U+FF5C FULLWIDTH VERTICAL LINE — not ASCII pipe.

Parameter typing:
- ``string="true"``  → value is preserved as a raw string (no JSON-decode)
- ``string="false"`` → value is JSON-decoded (``true`` → ``True``, ``42`` → ``42``,
  ``[1,2]`` → list, etc.). On JSON-decode failure we keep the raw string.

API mirrors ``HermesToolCallParser`` from sglang_miles_model so it's a drop-in
in the same model-backend slot.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from .tool_parser import (
    UNKNOWN_TOOL_NAME,
    ToolCallParseResult,
    ToolCallParser,
)

logger = logging.getLogger(__name__)


# FULLWIDTH VERTICAL LINE sentinel used by DeepSeek-V4
_BAR = "｜"  # ｜

# We anchor on the literal opening sequence "<｜DSML｜". Closing tags use
# "</｜DSML｜...>". Parsing is regex-based and deliberately permissive about
# whitespace so we don't choke on stray newlines the model emits.
_TOOL_CALLS_BLOCK = re.compile(
    rf"<{_BAR}DSML{_BAR}tool_calls>(.*?)</{_BAR}DSML{_BAR}tool_calls>",
    re.DOTALL,
)
_INVOKE_BLOCK = re.compile(
    rf'<{_BAR}DSML{_BAR}invoke\s+name="(?P<name>[^"]+)">'
    rf"(?P<body>.*?)"
    rf"</{_BAR}DSML{_BAR}invoke>",
    re.DOTALL,
)
_PARAM = re.compile(
    rf'<{_BAR}DSML{_BAR}parameter\s+name="(?P<name>[^"]+)"'
    rf'(?:\s+string="(?P<is_string>true|false)")?\s*>'
    rf"(?P<value>.*?)"
    rf"</{_BAR}DSML{_BAR}parameter>",
    re.DOTALL,
)

# DeepSeek often emits a thinking block before the tool call. We strip
# anything inside <think>...</think> so reasoning-time fake tool calls aren't
# executed (mirrors the policy in HermesToolCallParser).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _coerce_param_value(raw: str, is_string: str | None) -> object:
    """Decode a DSML parameter value.

    - ``string="true"``: keep as-is (raw string, no JSON unescape).
    - ``string="false"`` or attribute absent: try ``json.loads``; fall back to raw.
    """
    if is_string == "true":
        return raw
    # is_string in {"false", None}: try JSON-decode (handles bool / number / list / dict)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


class DSMLToolCallParser(ToolCallParser):
    """Parser for DeepSeek-V4 DSML tool-call format.

    Drop-in replacement for ``HermesToolCallParser`` in
    ``sglang_miles_model/sglang_model.py``. Same ``parse()`` signature, same
    ``ToolCallParseResult`` dataclass, same error semantics (parse failures
    yield results with ``is_error=True`` carrying the raw block).

    Attributes:
        strip_think: If True, drop ``<think>...</think>`` blocks before parsing
            so the agent doesn't execute draft tool calls inside reasoning.
            Default True — DeepSeek-V4 with thinking enabled often plans
            inside think blocks.
    """

    def __init__(self, strip_think: bool = True) -> None:
        self.strip_think = strip_think

    @property
    def message_separator(self) -> str:
        """DeepSeek-V4 chat template uses no extra separator between turns
        beyond what the template emits itself."""
        return ""

    def parse(self, text: str) -> list[ToolCallParseResult]:
        if not text:
            return []

        clean = _THINK_BLOCK.sub("", text) if self.strip_think else text

        results: list[ToolCallParseResult] = []
        for tc_block in _TOOL_CALLS_BLOCK.finditer(clean):
            block_body = tc_block.group(1)
            for m in _INVOKE_BLOCK.finditer(block_body):
                name = m.group("name") or UNKNOWN_TOOL_NAME
                body = m.group("body")
                tc_id = f"call_{uuid.uuid4().hex[:24]}"

                args: dict[str, object] = {}
                parse_failure = False
                for pm in _PARAM.finditer(body):
                    pname = pm.group("name")
                    pvalue = pm.group("value")
                    is_string = pm.group("is_string")
                    args[pname] = _coerce_param_value(pvalue, is_string)

                if parse_failure or not args and body.strip():
                    # Body exists but no parameters parsed → likely format drift
                    results.append(
                        ToolCallParseResult(
                            id=tc_id,
                            name=name,
                            input={},
                            raw=body,
                        )
                    )
                else:
                    results.append(
                        ToolCallParseResult(
                            id=tc_id,
                            name=name,
                            input=args,
                            raw=None,
                        )
                    )

        return results
