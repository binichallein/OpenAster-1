from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


IMAGE_TOKEN = "<|image_pad|>"
VISION_PREFIX = f"<|vision_start|>{IMAGE_TOKEN}<|vision_end|>"


class ContextLengthError(ValueError):
    pass


@dataclass(frozen=True)
class PromptResult:
    prompt: str
    messages: list[dict[str, str]]
    prompt_tokens: int
    dropped_turns: int
    image_turn: int | None


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


def validate_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("conversation history must contain at least one user message")

    normalized: list[dict[str, str]] = []
    expected_role = "user"
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role != expected_role:
            raise ValueError(
                f"message {index} must have role {expected_role!r}, got {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} content must be a non-empty string")
        normalized.append({"role": role, "content": content.strip()})
        expected_role = "assistant" if role == "user" else "user"
    return normalized


def render_prompt(
    tokenizer,
    messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str = "",
    thinking: bool = False,
    image_turn: int | None = None,
    image_seq_len: int = 576,
) -> str:
    normalized = validate_messages(messages)
    rendered_messages: list[dict[str, str]] = [dict(item) for item in normalized]

    if image_turn is not None:
        if not 0 <= image_turn < len(rendered_messages):
            raise ValueError("image_turn is outside the conversation history")
        if rendered_messages[image_turn]["role"] != "user":
            raise ValueError("image_turn must point to a user message")
        rendered_messages[image_turn]["content"] = (
            f"{VISION_PREFIX}\n{rendered_messages[image_turn]['content']}"
        )

    if system_prompt.strip():
        rendered_messages.insert(
            0, {"role": "system", "content": system_prompt.strip()}
        )

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": thinking,
    }
    try:
        prompt = tokenizer.apply_chat_template(rendered_messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        prompt = tokenizer.apply_chat_template(rendered_messages, **kwargs)
    return expand_image_tokens(prompt, image_seq_len)


def _prompt_token_count(tokenizer, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False)
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        input_ids = encoded["input_ids"]
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def fit_messages_to_context(
    tokenizer,
    messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str = "",
    thinking: bool = False,
    image_turn: int | None = None,
    image_seq_len: int = 576,
    context_tokens: int,
    max_new_tokens: int,
) -> PromptResult:
    if context_tokens <= max_new_tokens:
        raise ValueError("context_tokens must be greater than max_new_tokens")

    working = validate_messages(messages)
    if working[-1]["role"] != "user":
        raise ValueError("the current conversation must end with a user message")
    if image_turn is not None:
        if not 0 <= image_turn < len(working):
            raise ValueError("image_turn is outside the conversation history")
        if working[image_turn]["role"] != "user":
            raise ValueError("image_turn must point to a user message")

    prompt_budget = context_tokens - max_new_tokens
    current_image_turn = image_turn
    dropped_turns = 0

    while True:
        prompt = render_prompt(
            tokenizer,
            working,
            system_prompt=system_prompt,
            thinking=thinking,
            image_turn=current_image_turn,
            image_seq_len=image_seq_len,
        )
        prompt_tokens = _prompt_token_count(tokenizer, prompt)
        if prompt_tokens <= prompt_budget:
            return PromptResult(
                prompt=prompt,
                messages=[dict(item) for item in working],
                prompt_tokens=prompt_tokens,
                dropped_turns=dropped_turns,
                image_turn=current_image_turn,
            )

        drop_index: int | None = None
        for index in range(0, len(working) - 1, 2):
            if index + 1 >= len(working):
                break
            if current_image_turn in {index, index + 1}:
                continue
            drop_index = index
            break

        if drop_index is None:
            raise ContextLengthError(
                "The current conversation cannot fit in the selected context-token budget. "
                "Increase --context-tokens or shorten the current prompt/system prompt."
            )

        del working[drop_index : drop_index + 2]
        if current_image_turn is not None and drop_index < current_image_turn:
            current_image_turn -= 2
        dropped_turns += 1
