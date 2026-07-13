from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence


IMAGE_TOKEN = "<|image_pad|>"
VISION_PREFIX = f"<|vision_start|>{IMAGE_TOKEN}<|vision_end|>"
DEFAULT_TEXT_MODEL = "binichallein/OpenAster1-math"


@dataclass(frozen=True)
class ModelLoaders:
    config: Callable[..., Any]
    tokenizer: Callable[..., Any]
    text_model: Callable[..., Any]
    vision_model: Callable[..., Any]
    image_processor: Callable[..., Any]


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


def _default_loaders() -> ModelLoaders:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        CLIPImageProcessor,
        LlavaForConditionalGeneration,
    )

    return ModelLoaders(
        config=AutoConfig.from_pretrained,
        tokenizer=AutoTokenizer.from_pretrained,
        text_model=AutoModelForCausalLM.from_pretrained,
        vision_model=LlavaForConditionalGeneration.from_pretrained,
        image_processor=CLIPImageProcessor.from_pretrained,
    )


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}") from exc


def _model_context_length(config: Any, default: int = 32768) -> int:
    direct = getattr(config, "max_position_embeddings", None)
    if isinstance(direct, int) and direct > 0:
        return direct
    text_config = getattr(config, "text_config", None)
    nested = getattr(text_config, "max_position_embeddings", None)
    if isinstance(nested, int) and nested > 0:
        return nested
    return default


def _load_rgb_image(image: Any):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as opened:
            return opened.convert("RGB")
    if isinstance(image, bytes):
        import io

        with Image.open(io.BytesIO(image)) as opened:
            return opened.convert("RGB")
    raise TypeError("image must be a PIL image, file path, or encoded image bytes")


def _run_seeded_generate(
    model: Any,
    generation_inputs: Mapping[str, Any],
    *,
    seed: int,
    torch_module=None,
    seed_fn: Callable[[int], None] | None = None,
) -> None:
    if torch_module is None:
        import torch as torch_module
    if seed_fn is None:
        from transformers import set_seed

        seed_fn = set_seed
    seed_fn(seed)
    with torch_module.inference_mode():
        model.generate(**generation_inputs)


class OpenAsterEngine:
    def __init__(
        self,
        *,
        model_name: str,
        kind: Literal["text", "vision"],
        config: Any,
        tokenizer: Any,
        model: Any,
        image_processor: Any | None,
        image_seq_len: int,
    ) -> None:
        self.model_name = model_name
        self.kind = kind
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.image_seq_len = image_seq_len
        self.max_context_tokens = _model_context_length(config)
        self.last_prompt_result: PromptResult | None = None
        self._generation_lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
        loaders: ModelLoaders | None = None,
    ) -> "OpenAsterEngine":
        is_default_loaders = loaders is None
        active_loaders = loaders or _default_loaders()
        common = {"trust_remote_code": trust_remote_code}
        config = active_loaders.config(model_name_or_path, **common)
        kind = detect_model_kind(config.model_type)
        tokenizer = active_loaders.tokenizer(
            model_name_or_path,
            fix_mistral_regex=True,
            **common,
        )

        resolved_dtype = _resolve_dtype(dtype) if is_default_loaders else dtype
        model_kwargs: dict[str, Any] = {
            "dtype": resolved_dtype,
            "low_cpu_mem_usage": True,
            **common,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": device}
        if attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation

        image_processor = None
        if kind == "vision":
            model = active_loaders.vision_model(model_name_or_path, **model_kwargs)
            image_processor = active_loaders.image_processor(model_name_or_path)
        else:
            model = active_loaders.text_model(model_name_or_path, **model_kwargs)

        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        image_seq_len = int(getattr(config, "image_seq_length", 576))
        return cls(
            model_name=model_name_or_path,
            kind=kind,
            config=config,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            image_seq_len=image_seq_len,
        )

    def _input_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            return next(self.model.parameters()).device

    def _model_dtype(self):
        try:
            return next(self.model.parameters()).dtype
        except (AttributeError, StopIteration):
            import torch

            return torch.float32

    def _prepare_generation(
        self,
        messages: Sequence[Mapping[str, Any]],
        sampling: SamplingConfig,
        *,
        image: Any | None,
        image_turn: int | None,
        system_prompt: str,
        thinking: bool,
    ) -> tuple[dict[str, Any], PromptResult]:
        sampling.validate()
        if self.kind == "text" and image is not None:
            raise ValueError("The loaded OpenAster checkpoint is text-only and cannot accept an image.")
        if image_turn is not None and image is None:
            raise ValueError("image_turn was provided without an image")
        if image is not None and image_turn is None:
            image_turn = len(messages) - 1

        context_tokens = min(sampling.context_tokens, self.max_context_tokens)
        prompt_result = fit_messages_to_context(
            self.tokenizer,
            messages,
            system_prompt=system_prompt,
            thinking=thinking,
            image_turn=image_turn,
            image_seq_len=self.image_seq_len,
            context_tokens=context_tokens,
            max_new_tokens=sampling.max_new_tokens,
        )
        self.last_prompt_result = prompt_result

        encoded = self.tokenizer(
            prompt_result.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_device = self._input_device()
        input_ids = encoded["input_ids"].to(input_device)
        attention_mask = encoded.get("attention_mask")
        generation_inputs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            generation_inputs["attention_mask"] = attention_mask.to(input_device)

        if image is not None:
            image_token_id = getattr(self.config, "image_token_index", None)
            if image_token_id is None:
                image_token_id = getattr(self.config, "image_token_id", None)
            if image_token_id is None:
                image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            image_token_count = int((input_ids == image_token_id).sum().item())
            if image_token_count != self.image_seq_len:
                raise ValueError(
                    f"Prompt contains {image_token_count} image tokens; "
                    f"OpenAster1-VL expects {self.image_seq_len}."
                )
            rgb_image = _load_rgb_image(image)
            pixel_values = self.image_processor(
                images=rgb_image,
                return_tensors="pt",
            ).pixel_values
            generation_inputs["pixel_values"] = pixel_values.to(
                input_device,
                dtype=self._model_dtype(),
            )

        generation_inputs.update(sampling.generation_kwargs())
        generation_inputs["pad_token_id"] = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        generation_inputs["eos_token_id"] = self.tokenizer.eos_token_id
        return generation_inputs, prompt_result

    def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        sampling: SamplingConfig,
        *,
        image: Any | None = None,
        image_turn: int | None = None,
        system_prompt: str = "",
        thinking: bool = False,
        stop_event: threading.Event | None = None,
    ) -> Iterator[str]:
        if self.kind == "text" and image is not None:
            raise ValueError("The loaded OpenAster checkpoint is text-only and cannot accept an image.")

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        generation_inputs, _prompt_result = self._prepare_generation(
            messages,
            sampling,
            image=image,
            image_turn=image_turn,
            system_prompt=system_prompt,
            thinking=thinking,
        )
        active_stop_event = stop_event or threading.Event()

        class EventStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return active_stop_event.is_set()

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=180.0,
        )
        generation_inputs["streamer"] = streamer
        generation_inputs["stopping_criteria"] = StoppingCriteriaList(
            [EventStoppingCriteria()]
        )
        errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)

        def generate() -> None:
            try:
                with self._generation_lock:
                    _run_seeded_generate(
                        self.model,
                        generation_inputs,
                        seed=sampling.seed,
                        torch_module=torch,
                    )
            except BaseException as exc:
                errors.put(exc)
                streamer.end()

        worker = threading.Thread(target=generate, name="openaster-generate", daemon=True)
        worker.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            active_stop_event.set()
            worker.join(timeout=10)
        if not errors.empty():
            raise errors.get()


def parse_repl_command(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    command, _, value = stripped[1:].partition(" ")
    return command.lower(), value.strip()


def update_sampling_config(config: SamplingConfig, expression: str) -> SamplingConfig:
    key, separator, raw_value = expression.strip().partition(" ")
    allowed = {
        "max_new_tokens": int,
        "temperature": float,
        "top_p": float,
        "top_k": int,
        "repetition_penalty": float,
        "seed": int,
        "context_tokens": int,
    }
    if not separator or key not in allowed:
        raise ValueError(
            "Unknown sampling parameter. Use max_new_tokens, temperature, top_p, "
            "top_k, repetition_penalty, seed, or context_tokens."
        )
    try:
        value = allowed[key](raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for {key}: {raw_value!r}") from exc
    updated = replace(config, **{key: value})
    updated.validate()
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run text or vision inference with OpenAster1 checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_TEXT_MODEL, help="Hub model ID or local path")
    parser.add_argument("--prompt", help="Run one prompt and exit; omit for interactive chat")
    parser.add_argument("--image", help="Image path for OpenAster1-VL")
    parser.add_argument("--system", default="", help="Optional system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON in one-shot mode")
    return parser


def _sampling_from_args(args: argparse.Namespace) -> SamplingConfig:
    config = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        context_tokens=args.context_tokens,
    )
    config.validate()
    return config


def _format_sampling(config: SamplingConfig) -> str:
    return (
        f"max_new_tokens={config.max_new_tokens}, temperature={config.temperature:g}, "
        f"top_p={config.top_p:g}, top_k={config.top_k}, "
        f"repetition_penalty={config.repetition_penalty:g}, seed={config.seed}, "
        f"context_tokens={config.context_tokens}"
    )


def _print_banner(engine: OpenAsterEngine, sampling: SamplingConfig) -> None:
    print("\nOpenAster1 terminal chat")
    print(f"model  {engine.model_name}")
    print(f"mode   {engine.kind} · context up to {engine.max_context_tokens:,} tokens")
    print(f"sample {_format_sampling(sampling)}")
    print("commands: /help /clear /image PATH /system TEXT /params /set KEY VALUE /exit\n")


def _stream_answer(
    engine: OpenAsterEngine,
    messages: list[dict[str, str]],
    sampling: SamplingConfig,
    *,
    image: str | None,
    image_turn: int | None,
    system_prompt: str,
    thinking: bool,
) -> str:
    chunks: list[str] = []
    for chunk in engine.stream(
        messages,
        sampling,
        image=image,
        image_turn=image_turn,
        system_prompt=system_prompt,
        thinking=thinking,
    ):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()
    return "".join(chunks).strip()


def _run_one_shot(
    engine: OpenAsterEngine,
    args: argparse.Namespace,
    sampling: SamplingConfig,
) -> int:
    messages = [{"role": "user", "content": args.prompt.strip()}]
    started = time.perf_counter()
    if args.json:
        chunks = list(
            engine.stream(
                messages,
                sampling,
                image=args.image,
                image_turn=0 if args.image else None,
                system_prompt=args.system,
                thinking=args.thinking,
            )
        )
        answer = "".join(chunks).strip()
        result = engine.last_prompt_result
        print(
            json.dumps(
                {
                    "model": engine.model_name,
                    "mode": engine.kind,
                    "answer": answer,
                    "prompt_tokens": result.prompt_tokens if result else None,
                    "dropped_turns": result.dropped_turns if result else None,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                ensure_ascii=False,
            )
        )
    else:
        _stream_answer(
            engine,
            messages,
            sampling,
            image=args.image,
            image_turn=0 if args.image else None,
            system_prompt=args.system,
            thinking=args.thinking,
        )
    return 0


def _run_repl(
    engine: OpenAsterEngine,
    args: argparse.Namespace,
    sampling: SamplingConfig,
) -> int:
    messages: list[dict[str, str]] = []
    system_prompt = args.system
    image = args.image
    image_turn: int | None = None
    _print_banner(engine, sampling)

    while True:
        try:
            line = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not line:
            continue

        command = parse_repl_command(line)
        if command is not None:
            name, value = command
            if name in {"exit", "quit"}:
                print("bye")
                return 0
            if name == "help":
                print("/clear · /image PATH · /system TEXT · /params · /set KEY VALUE · /exit")
            elif name == "clear":
                messages.clear()
                image = None
                image_turn = None
                print("conversation cleared")
            elif name == "image":
                if engine.kind != "vision":
                    print("this model is text-only")
                elif not value:
                    print("usage: /image PATH")
                elif not Path(value).expanduser().is_file():
                    print(f"image not found: {value}")
                else:
                    messages.clear()
                    image = str(Path(value).expanduser())
                    image_turn = None
                    print(f"new visual conversation: {image}")
            elif name == "system":
                system_prompt = value
                print("system prompt updated")
            elif name == "params":
                print(_format_sampling(sampling))
            elif name == "set":
                try:
                    sampling = update_sampling_config(sampling, value)
                    print(_format_sampling(sampling))
                except ValueError as exc:
                    print(f"error: {exc}")
            else:
                print(f"unknown command: /{name}")
            continue

        messages.append({"role": "user", "content": line})
        if image is not None and image_turn is None:
            image_turn = len(messages) - 1
        print("Aster > ", end="", flush=True)
        try:
            answer = _stream_answer(
                engine,
                messages,
                sampling,
                image=image,
                image_turn=image_turn,
                system_prompt=system_prompt,
                thinking=args.thinking,
            )
        except Exception as exc:
            messages.pop()
            print(f"error: {exc}", file=sys.stderr)
            continue
        if answer:
            messages.append({"role": "assistant", "content": answer})
        else:
            messages.pop()
            print("(empty response)")


def main() -> int:
    args = build_parser().parse_args()
    sampling = _sampling_from_args(args)
    print(f"Loading {args.model} ...", flush=True)
    engine = OpenAsterEngine.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("--prompt must not be empty")
        return _run_one_shot(engine, args, sampling)
    return _run_repl(engine, args, sampling)


if __name__ == "__main__":
    raise SystemExit(main())
