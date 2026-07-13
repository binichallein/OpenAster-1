from __future__ import annotations

import base64
import json

import pytest

from inference.app import (
    DEFAULT_GUI_MODEL,
    HTML,
    MAX_IMAGE_BYTES,
    ReasoningStreamParser,
    build_parser,
    parse_chat_request,
    serialize_event,
    stream_generation_events,
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


def test_reasoning_parser_passes_non_thinking_stream_through_as_answer() -> None:
    parser = ReasoningStreamParser(enabled=False)

    assert parser.feed("plain ") == [("answer", "plain ")]
    assert parser.feed("response") == [("answer", "response")]
    assert parser.finish() == []


def test_reasoning_parser_strips_split_opening_marker() -> None:
    parser = ReasoningStreamParser(enabled=True)

    assert parser.feed("<thi") == []
    assert parser.feed("nk>first step") == [("reasoning", "first step")]


def test_reasoning_parser_splits_reasoning_and_answer_across_chunk_boundary() -> None:
    parser = ReasoningStreamParser(enabled=True)

    events = []
    for chunk in ["check the premise</th", "ink>Final ", "answer"]:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())

    assert events == [
        ("reasoning", "check the premise"),
        ("reasoning_done", ""),
        ("answer", "Final "),
        ("answer", "answer"),
    ]
    assert "<think>" not in "".join(text for _kind, text in events)
    assert "</think>" not in "".join(text for _kind, text in events)


def test_reasoning_parser_completes_unclosed_reasoning_at_end_of_stream() -> None:
    parser = ReasoningStreamParser(enabled=True)

    assert parser.feed("unfinished reasoning") == [
        ("reasoning", "unfinished reasoning")
    ]
    assert parser.finish() == [("reasoning_incomplete", "")]


def test_generation_events_include_timed_reasoning_before_answer_tokens() -> None:
    clock_values = iter([10.0, 12.34])

    events = [
        json.loads(event)
        for event in stream_generation_events(
            ["work</thi", "nk>answer"],
            thinking=True,
            clock=lambda: next(clock_values),
        )
    ]

    assert events == [
        {"type": "reasoning", "text": "work"},
        {"type": "reasoning_done", "seconds": 2.3},
        {"type": "token", "text": "answer"},
    ]


def test_generation_events_mark_unclosed_reasoning_as_incomplete() -> None:
    clock_values = iter([5.0, 7.0])

    events = [
        json.loads(event)
        for event in stream_generation_events(
            ["unfinished reasoning"],
            thinking=True,
            clock=lambda: next(clock_values),
        )
    ]

    assert events == [
        {"type": "reasoning", "text": "unfinished reasoning"},
        {"type": "reasoning_incomplete", "seconds": 2.0},
    ]


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


def test_gui_has_collapsible_live_reasoning_and_separate_answer() -> None:
    assert 'details.className = "thinking-panel"' in HTML
    assert 'summary.className = "thinking-summary"' in HTML
    assert 'label.textContent = "思考中"' in HTML
    assert 'reasoning.className = "thinking-copy"' in HTML
    assert 'answer.className = "answer-copy"' in HTML
    assert 'spinner.className = "thinking-spinner"' in HTML
    assert "@keyframes spin" in HTML


def test_gui_collapses_reasoning_after_timed_server_event() -> None:
    assert 'event.type === "reasoning"' in HTML
    assert 'event.type === "reasoning_done"' in HTML
    assert 'pending.thinkingLabel.textContent = `已思考 ${seconds.toFixed(1)} 秒`' in HTML
    assert "pending.thinkingPanel.open = false" in HTML
    assert "currentAnswer += event.text" in HTML
    assert 'history.push({ role: "assistant", content: currentAnswer.trim() })' in HTML


def test_gui_preserves_incomplete_or_stopped_thinking_without_storing_reasoning() -> None:
    assert 'event.type === "reasoning_incomplete"' in HTML
    assert 'pending.thinkingLabel.textContent = `思考已截断 ${seconds.toFixed(1)} 秒`' in HTML
    assert 'const incompleteAnswer = "未生成最终回答。请提高 Max new tokens 后重新生成。"' in HTML
    assert 'const stoppedAnswer = "生成已停止，未得到最终回答。"' in HTML
    assert 'history.push({ role: "assistant", content: stoppedAnswer })' in HTML


def test_gui_clear_during_generation_does_not_restore_aborted_history() -> None:
    assert "let clearingConversation = false" in HTML
    assert 'if (error.name === "AbortError" && clearingConversation)' in HTML
    assert "clearingConversation = true" in HTML
    assert "clearingConversation = false" in HTML
