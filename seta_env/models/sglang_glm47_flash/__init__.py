"""SGLang model backend for GLM-4.7-Flash (seta_env).

Reuses the generic SGLangModel + SGLangClient + TokenManager from
sglang_miles_model (GLM-4.7-Flash has a proper chat_template.jinja, so
tokenizer.apply_chat_template handles encoding) and only swaps in a
GLM-native tool-call parser.
"""

from .glm_tool_parser import GLMToolCallParser

__all__ = ["GLMToolCallParser"]
