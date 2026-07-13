from __future__ import annotations

import base64
import json

import pytest

from inference.app import (
    DEFAULT_GUI_MODEL,
    HTML,
    MAX_IMAGE_BYTES,
    build_parser,
    parse_chat_request,
    serialize_event,
)


def valid_payload() -> dict:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "system_prompt": "Be concise.",
        "thinking": False,
        "sampling": {
            "max_new_tokens": 128,
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "seed": 42,
            "context_tokens": 8192,
        },
    }


def test_gui_defaults_to_openaster_vl() -> None:
    args = build_parser().parse_args([])

    assert DEFAULT_GUI_MODEL == "binichallein/OpenAster1-VL"
    assert args.model == DEFAULT_GUI_MODEL
    assert args.port == 7860


@pytest.mark.parametrize(
    "control_id",
    [
        "imageInput",
        "systemPrompt",
        "maxNewTokens",
        "contextTokens",
        "temperature",
        "topP",
        "topK",
        "repetitionPenalty",
        "seed",
        "thinking",
        "stopButton",
        "regenerateButton",
        "clearButton",
    ],
)
def test_gui_contains_required_conversation_controls(control_id: str) -> None:
    assert f'id="{control_id}"' in HTML


def test_parse_chat_request_accepts_sampling_and_history() -> None:
    request = parse_chat_request(valid_payload())

    assert request.messages[-1] == {"role": "user", "content": "hello"}
    assert request.system_prompt == "Be concise."
    assert request.sampling.max_new_tokens == 128
    assert request.sampling.context_tokens == 8192
    assert request.image_bytes is None
    assert request.image_turn is None


def test_parse_chat_request_defaults_image_to_latest_user_turn() -> None:
    payload = valid_payload()
    payload["image_data"] = "data:image/png;base64," + base64.b64encode(b"png").decode()

    request = parse_chat_request(payload)

    assert request.image_bytes == b"png"
    assert request.image_turn == 0


def test_parse_chat_request_rejects_image_turn_without_image() -> None:
    payload = valid_payload()
    payload["image_turn"] = 0

    with pytest.raises(ValueError, match="without image_data"):
        parse_chat_request(payload)


def test_parse_chat_request_rejects_invalid_message_order() -> None:
    payload = valid_payload()
    payload["messages"] = [{"role": "assistant", "content": "wrong"}]

    with pytest.raises(ValueError, match="role"):
        parse_chat_request(payload)


def test_parse_chat_request_rejects_oversized_image() -> None:
    payload = valid_payload()
    payload["image_data"] = "data:image/png;base64," + base64.b64encode(
        b"x" * (MAX_IMAGE_BYTES + 1)
    ).decode()

    with pytest.raises(ValueError, match="20 MiB"):
        parse_chat_request(payload)


def test_parse_chat_request_rejects_invalid_sampling() -> None:
    payload = valid_payload()
    payload["sampling"]["top_p"] = 0

    with pytest.raises(ValueError, match="top_p"):
        parse_chat_request(payload)


def test_error_event_is_newline_delimited_json() -> None:
    payload = serialize_event("error", message="bad request")

    assert payload.endswith(b"\n")
    assert json.loads(payload) == {"type": "error", "message": "bad request"}


def test_hidden_empty_state_has_explicit_display_override() -> None:
    assert ".empty-state[hidden] { display: none; }" in HTML


def test_gui_configures_mathjax_for_inline_and_display_tex() -> None:
    assert "window.MathJax" in HTML
    assert "inlineMath: [['\\\\(', '\\\\)']]" in HTML
    assert "displayMath: [['\\\\[', '\\\\]']]" in HTML
    assert "tex-chtml.js" in HTML


def test_gui_typesets_completed_assistant_math_without_html_injection() -> None:
    assert "async function typesetMath(bubble)" in HTML
    assert "await window.MathJax.typesetPromise([bubble])" in HTML
    assert "await typesetMath(pending.bubble)" in HTML
    assert "bubble.textContent = text" in HTML
    assert "bubble.innerHTML" not in HTML
