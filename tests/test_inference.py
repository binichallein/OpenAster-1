from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference.app import (
    IMAGE_TOKEN,
    ContextLengthError,
    ModelLoaders,
    OpenAsterEngine,
    SamplingConfig,
    detect_model_kind,
    expand_image_tokens,
    fit_messages_to_context,
    render_prompt,
    _run_seeded_generate,
    validate_messages,
)


class FakeEncoding:
    def __init__(self, text: str) -> None:
        normalized = text.replace(IMAGE_TOKEN, " image_token ")
        self.input_ids = normalized.split()


class FakeTokenizer:
    def __init__(self, reject_thinking: bool = False) -> None:
        self.reject_thinking = reject_thinking
        self.calls: list[dict] = []

    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        self.calls.append({"messages": messages, **kwargs})
        if self.reject_thinking and "enable_thinking" in kwargs:
            raise TypeError("enable_thinking is unsupported")
        lines = [f"{item['role']} {item['content']}" for item in messages]
        if kwargs.get("add_generation_prompt"):
            lines.append("assistant")
        return "\n".join(lines)

    def __call__(self, text: str, **_kwargs) -> FakeEncoding:
        return FakeEncoding(text)


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            eos_token_id=2,
            pad_token_id=0,
            max_position_embeddings=4096,
            use_cache=False,
        )
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class LoaderRecorder:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        self.calls: list[tuple[str, dict]] = []

    def config(self, _model: str, **kwargs):
        self.calls.append(("config", kwargs))
        return SimpleNamespace(
            model_type=self.model_type,
            image_seq_length=576,
            max_position_embeddings=131072,
        )

    def tokenizer(self, _model: str, **kwargs):
        self.calls.append(("tokenizer", kwargs))
        return FakeTokenizer()

    def text_model(self, _model: str, **kwargs):
        self.calls.append(("text_model", kwargs))
        return FakeModel()

    def vision_model(self, _model: str, **kwargs):
        self.calls.append(("vision_model", kwargs))
        return FakeModel()

    def image_processor(self, _model: str, **kwargs):
        self.calls.append(("image_processor", kwargs))
        return object()

    def loaders(self) -> ModelLoaders:
        return ModelLoaders(
            config=self.config,
            tokenizer=self.tokenizer,
            text_model=self.text_model,
            vision_model=self.vision_model,
            image_processor=self.image_processor,
        )


def test_detects_supported_model_families() -> None:
    assert detect_model_kind("qwen3_moe") == "text"
    assert detect_model_kind("llava") == "vision"


def test_rejects_unsupported_model_family() -> None:
    with pytest.raises(ValueError, match="Unsupported model type"):
        detect_model_kind("bert")


def test_expands_exactly_one_image_marker() -> None:
    rendered = "a<|image_pad|>b"
    assert expand_image_tokens(rendered, 3) == (
        "a<|image_pad|><|image_pad|><|image_pad|>b"
    )


def test_image_expansion_leaves_text_without_marker_unchanged() -> None:
    assert expand_image_tokens("plain text", 576) == "plain text"


def test_image_expansion_rejects_invalid_sequence_length() -> None:
    with pytest.raises(ValueError, match="image_seq_len"):
        expand_image_tokens("<|image_pad|>", 0)


def test_temperature_zero_disables_sampling() -> None:
    config = SamplingConfig(temperature=0.0)
    kwargs = config.generation_kwargs()

    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_positive_temperature_enables_sampling_controls() -> None:
    config = SamplingConfig(temperature=0.65, top_p=0.82, top_k=37)

    assert config.generation_kwargs() == {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.65,
        "top_p": 0.82,
        "top_k": 37,
        "repetition_penalty": 1.05,
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.1}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_k": -1}, "top_k"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"seed": -1}, "seed"),
        ({"context_tokens": 16, "max_new_tokens": 16}, "context_tokens"),
    ],
)
def test_sampling_validation_rejects_invalid_ranges(changes: dict, message: str) -> None:
    config = SamplingConfig(**changes)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_sampling_validation_accepts_deterministic_seed() -> None:
    config = SamplingConfig(seed=2026, context_tokens=4096, max_new_tokens=256)

    config.validate()
    assert config.seed == 2026


def test_validate_messages_trims_content_and_preserves_order() -> None:
    assert validate_messages(
        [
            {"role": "user", "content": "  question  "},
            {"role": "assistant", "content": " answer "},
            {"role": "user", "content": "follow up"},
        ]
    ) == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow up"},
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "not allowed here"}],
        [{"role": "assistant", "content": "wrong first role"}],
        [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ],
        [{"role": "user", "content": "   "}],
    ],
)
def test_validate_messages_rejects_invalid_history(messages: list[dict]) -> None:
    with pytest.raises(ValueError):
        validate_messages(messages)


def test_render_prompt_places_system_first_and_forwards_thinking() -> None:
    tokenizer = FakeTokenizer()

    prompt = render_prompt(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        system_prompt="Be concise.",
        thinking=True,
    )

    assert prompt.startswith("system Be concise.")
    assert tokenizer.calls[-1]["messages"][0]["role"] == "system"
    assert tokenizer.calls[-1]["enable_thinking"] is True


def test_render_prompt_retries_tokenizers_without_thinking_argument() -> None:
    tokenizer = FakeTokenizer(reject_thinking=True)

    prompt = render_prompt(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        thinking=False,
    )

    assert prompt.endswith("assistant")
    assert len(tokenizer.calls) == 2
    assert "enable_thinking" not in tokenizer.calls[-1]


def test_render_prompt_attaches_one_expanded_vision_marker() -> None:
    tokenizer = FakeTokenizer()

    prompt = render_prompt(
        tokenizer,
        [
            {"role": "user", "content": "describe it"},
            {"role": "assistant", "content": "it is a chart"},
            {"role": "user", "content": "what color"},
        ],
        image_turn=0,
        image_seq_len=3,
    )

    assert prompt.count(IMAGE_TOKEN) == 3
    assert "<|vision_start|>" in prompt
    assert tokenizer.calls[-1]["messages"][0]["content"].endswith("describe it")
    assert IMAGE_TOKEN not in tokenizer.calls[-1]["messages"][2]["content"]


def test_context_fitting_drops_oldest_complete_pair() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "keep question"},
        {"role": "assistant", "content": "keep answer"},
        {"role": "user", "content": "latest"},
    ]

    result = fit_messages_to_context(
        tokenizer,
        messages,
        context_tokens=12,
        max_new_tokens=3,
    )

    assert result.messages == messages[2:]
    assert result.dropped_turns == 1
    assert result.prompt_tokens == 9


def test_context_fitting_preserves_visual_turn_and_drops_middle_pair() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "image question"},
        {"role": "assistant", "content": "image answer"},
        {"role": "user", "content": "middle question"},
        {"role": "assistant", "content": "middle answer"},
        {"role": "user", "content": "latest"},
    ]

    result = fit_messages_to_context(
        tokenizer,
        messages,
        image_turn=0,
        image_seq_len=2,
        context_tokens=16,
        max_new_tokens=3,
    )

    assert [item["content"] for item in result.messages] == [
        "image question",
        "image answer",
        "latest",
    ]
    assert result.image_turn == 0
    assert result.dropped_turns == 1
    assert result.prompt.count(IMAGE_TOKEN) == 2


def test_context_fitting_rejects_current_turn_that_cannot_fit() -> None:
    tokenizer = FakeTokenizer()

    with pytest.raises(ContextLengthError, match="current conversation"):
        fit_messages_to_context(
            tokenizer,
            [{"role": "user", "content": "one two three four five six"}],
            context_tokens=6,
            max_new_tokens=2,
        )


def test_engine_selects_text_model_loader() -> None:
    recorder = LoaderRecorder("qwen3_moe")

    engine = OpenAsterEngine.from_pretrained(
        "text-model",
        device="cpu",
        dtype="float32",
        loaders=recorder.loaders(),
    )

    assert engine.kind == "text"
    assert engine.image_processor is None
    assert "text_model" in [name for name, _kwargs in recorder.calls]
    assert "vision_model" not in [name for name, _kwargs in recorder.calls]
    assert engine.model.eval_called is True


def test_engine_selects_vision_model_and_processor_loaders() -> None:
    recorder = LoaderRecorder("llava")

    engine = OpenAsterEngine.from_pretrained(
        "vision-model",
        device="cpu",
        dtype="float32",
        loaders=recorder.loaders(),
    )

    names = [name for name, _kwargs in recorder.calls]
    assert engine.kind == "vision"
    assert engine.image_seq_len == 576
    assert "vision_model" in names
    assert "image_processor" in names
    assert "text_model" not in names


def test_text_engine_rejects_image_before_generation() -> None:
    recorder = LoaderRecorder("qwen3_moe")
    engine = OpenAsterEngine.from_pretrained(
        "text-model",
        device="cpu",
        dtype="float32",
        loaders=recorder.loaders(),
    )

    with pytest.raises(ValueError, match="text-only"):
        list(
            engine.stream(
                [{"role": "user", "content": "describe"}],
                SamplingConfig(max_new_tokens=8, context_tokens=128),
                image=object(),
                image_turn=0,
            )
        )


def test_seeded_generation_sets_seed_without_forwarding_generator() -> None:
    calls: dict = {}

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class FakeTorchModule:
        @staticmethod
        def inference_mode():
            return InferenceMode()

    class GeneratingModel:
        def generate(self, **kwargs):
            calls["kwargs"] = kwargs

    def set_seed(seed: int) -> None:
        calls["seed"] = seed

    _run_seeded_generate(
        GeneratingModel(),
        {"input_ids": "tokens"},
        seed=2026,
        torch_module=FakeTorchModule(),
        seed_fn=set_seed,
    )

    assert calls == {"seed": 2026, "kwargs": {"input_ids": "tokens"}}
