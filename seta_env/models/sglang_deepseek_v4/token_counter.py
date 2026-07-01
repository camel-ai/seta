"""HuggingFace-backed token counter for the DeepSeek-V4 SGLang backend.

Extracted from ``sglang_miles_model.sglang_model.TokenCounter`` so this package
is self-contained and doesn't reach into the sibling backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from camel.messages import OpenAIMessage
from camel.utils import BaseTokenCounter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TokenCounter(BaseTokenCounter):
    """Token counter using a HuggingFace tokenizer."""

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", tokens_per_message: int = 4):
        self.tokenizer = tokenizer
        self.tokens_per_message = tokens_per_message

    def count_tokens_from_messages(self, messages: list[OpenAIMessage]) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                if not isinstance(value, list):
                    num_tokens += len(self.tokenizer.encode(str(value)))
                else:
                    for item in value:
                        if item["type"] == "text":
                            num_tokens += len(self.tokenizer.encode(str(item["text"])))
                        else:
                            raise ValueError(f"Unsupported item type: {item['type']}")
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
