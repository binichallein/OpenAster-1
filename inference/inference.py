from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


IMAGE_TOKEN = "<|image_pad|>"


def detect_model_kind(model_type: str) -> Literal["text", "vision"]:
    if model_type == "qwen3_moe":
        return "text"
    if model_type == "llava":
        return "vision"
    raise ValueError(
        f"Unsupported model type {model_type!r}; expected 'qwen3_moe' or 'llava'."
    )


def expand_image_tokens(text: str, image_seq_len: int) -> str:
    if image_seq_len < 1:
        raise ValueError("image_seq_len must be at least 1")
    return text.replace(IMAGE_TOKEN, IMAGE_TOKEN * image_seq_len)


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    seed: int = 42
    context_tokens: int = 32768

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.context_tokens <= self.max_new_tokens:
            raise ValueError("context_tokens must be greater than max_new_tokens")

    def generation_kwargs(self) -> dict[str, Any]:
        self.validate()
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.temperature > 0:
            kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        return kwargs
