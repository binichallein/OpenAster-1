from __future__ import annotations

import pytest

from inference.inference import SamplingConfig, detect_model_kind, expand_image_tokens


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
