"""SGLang model backend specialised for DeepSeek-V4-Flash / -Pro.

Talks to the same SGLang server miles spins up internally (no flag changes
required), parses DeepSeek-V4's native DSML tool-call format, and tracks
per-token IDs + logprobs for TITO RL training.
"""

from .dsml_tool_parser import DSMLToolCallParser
from .sglang_v4_model import DeepSeekV4SGLangModel

__all__ = ["DSMLToolCallParser", "DeepSeekV4SGLangModel"]
