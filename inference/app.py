from __future__ import annotations

import argparse
import base64
import binascii
import json
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import urlparse

try:
    from inference.inference import OpenAsterEngine, SamplingConfig, validate_messages
except ModuleNotFoundError:
    from inference import OpenAsterEngine, SamplingConfig, validate_messages


DEFAULT_GUI_MODEL = "binichallein/OpenAster1-VL"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 28 * 1024 * 1024
MAX_MESSAGE_CHARS = 32_000
MAX_SYSTEM_CHARS = 8_000


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, str]]
    system_prompt: str
    thinking: bool
    sampling: SamplingConfig
    image_bytes: bytes | None
    image_turn: int | None


def _number(
    values: Mapping[str, Any],
    key: str,
    cast,
    default: int | float,
) -> int | float:
    value = values.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc


def _decode_image_data(image_data: Any) -> bytes | None:
    if image_data in {None, ""}:
        return None
    if not isinstance(image_data, str):
        raise ValueError("image_data must be a base64 data URL")
    encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_data is not valid base64") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 20 MiB limit")
    return raw


def parse_chat_request(payload: Any) -> ChatRequest:
    if not isinstance(payload, Mapping):
        raise ValueError("request body must be a JSON object")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be a list")
    messages = validate_messages(raw_messages)
    if messages[-1]["role"] != "user":
        raise ValueError("messages must end with the current user turn")
    if any(len(item["content"]) > MAX_MESSAGE_CHARS for item in messages):
        raise ValueError(f"message content exceeds {MAX_MESSAGE_CHARS} characters")

    system_prompt = payload.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    if len(system_prompt) > MAX_SYSTEM_CHARS:
        raise ValueError(f"system_prompt exceeds {MAX_SYSTEM_CHARS} characters")

    thinking = payload.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ValueError("thinking must be true or false")

    values = payload.get("sampling", {})
    if not isinstance(values, Mapping):
        raise ValueError("sampling must be an object")
    sampling = SamplingConfig(
        max_new_tokens=int(_number(values, "max_new_tokens", int, 512)),
        temperature=float(_number(values, "temperature", float, 0.7)),
        top_p=float(_number(values, "top_p", float, 0.9)),
        top_k=int(_number(values, "top_k", int, 50)),
        repetition_penalty=float(
            _number(values, "repetition_penalty", float, 1.05)
        ),
        seed=int(_number(values, "seed", int, 42)),
        context_tokens=int(_number(values, "context_tokens", int, 32768)),
    )
    sampling.validate()

    image_bytes = _decode_image_data(payload.get("image_data"))
    image_turn = payload.get("image_turn")
    if image_turn is not None:
        if isinstance(image_turn, bool) or not isinstance(image_turn, int):
            raise ValueError("image_turn must be an integer")
        if image_bytes is None:
            raise ValueError("image_turn was provided without image_data")
    elif image_bytes is not None:
        image_turn = len(messages) - 1
    if image_turn is not None:
        if not 0 <= image_turn < len(messages):
            raise ValueError("image_turn is outside messages")
        if messages[image_turn]["role"] != "user":
            raise ValueError("image_turn must point to a user message")

    return ChatRequest(
        messages=messages,
        system_prompt=system_prompt.strip(),
        thinking=thinking,
        sampling=sampling,
        image_bytes=image_bytes,
        image_turn=image_turn,
    )


def serialize_event(event_type: str, **payload: Any) -> bytes:
    return (
        json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"
    ).encode("utf-8")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenAster1 Inference Studio</title>
  <style>
    :root {
      color-scheme: light;
      --page: #f4f7f8;
      --surface: #ffffff;
      --surface-soft: #f8fafb;
      --ink: #18212f;
      --muted: #647184;
      --line: #d8e0e7;
      --line-strong: #bdc9d4;
      --green: #087f5b;
      --green-soft: #e8f7f0;
      --blue: #2563eb;
      --blue-soft: #edf3ff;
      --amber: #c66a08;
      --amber-soft: #fff5e4;
      --red: #c81e45;
      --red-soft: #fff0f3;
      --purple: #7542d8;
      --shadow: 0 10px 30px rgba(27, 45, 65, .08);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      overflow: hidden;
      background: var(--page);
      color: var(--ink);
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, textarea { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    .topbar {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }
    .brand { display: flex; align-items: center; gap: 11px; min-width: 0; }
    .brand-mark {
      width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      border: 2px solid var(--ink);
      border-radius: 7px;
      color: var(--green);
      font-weight: 850;
      font-size: 12px;
      background: var(--green-soft);
    }
    .brand-name { font-size: 16px; font-weight: 760; line-height: 1.15; }
    .brand-sub { color: var(--muted); font-size: 11px; margin-top: 2px; }
    .runtime { display: flex; align-items: center; gap: 8px; min-width: 0; }
    .status-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--amber); }
    .status-dot.ready { background: var(--green); }
    .status-dot.busy { background: var(--blue); animation: pulse 1s ease-in-out infinite; }
    .status-dot.error { background: var(--red); }
    @keyframes pulse { 50% { opacity: .35; } }
    .runtime-copy { min-width: 0; text-align: right; }
    #modelName { display: block; max-width: 420px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; font-weight: 700; }
    #runtimeStatus { color: var(--muted); font-size: 11px; }
    .workspace {
      height: calc(100dvh - 64px);
      display: grid;
      grid-template-columns: 276px minmax(420px, 1fr) 278px;
      min-height: 0;
    }
    .rail {
      min-height: 0;
      overflow: auto;
      padding: 18px;
      background: var(--surface-soft);
    }
    .rail.left { border-right: 1px solid var(--line); }
    .rail.right { border-left: 1px solid var(--line); }
    .rail-section { padding: 0 0 20px; margin: 0 0 20px; border-bottom: 1px solid var(--line); }
    .rail-section:last-child { border-bottom: 0; margin-bottom: 0; }
    .section-title { margin: 0 0 10px; font-size: 12px; font-weight: 800; color: var(--ink); }
    .section-kicker { margin: -6px 0 10px; color: var(--muted); font-size: 11px; }
    .dropzone {
      position: relative;
      min-height: 166px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px dashed var(--line-strong);
      border-radius: 8px;
      background: var(--surface);
      transition: border-color .15s, background .15s;
    }
    .dropzone.drag { border-color: var(--green); background: var(--green-soft); }
    .dropzone img { width: 100%; height: 166px; object-fit: contain; display: block; }
    .drop-copy { color: var(--muted); font-size: 12px; text-align: center; padding: 16px; }
    .drop-copy strong { display: block; color: var(--ink); margin-bottom: 3px; }
    #imageInput { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
    textarea, input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
      outline: none;
    }
    textarea:focus, input[type="number"]:focus { border-color: var(--blue); box-shadow: 0 0 0 2px rgba(37, 99, 235, .12); }
    #systemPrompt { min-height: 118px; resize: vertical; padding: 10px; }
    .chat-panel { min-width: 0; min-height: 0; display: grid; grid-template-rows: 1fr auto; background: var(--surface); }
    .messages { min-height: 0; overflow: auto; padding: 26px clamp(20px, 5vw, 72px); scroll-behavior: smooth; }
    .empty-state { height: 100%; display: grid; place-items: center; color: var(--muted); }
    .empty-state[hidden] { display: none; }
    .empty-core { text-align: center; }
    .empty-mark { width: 52px; height: 52px; display: grid; place-items: center; margin: 0 auto 12px; border: 2px solid var(--ink); border-radius: 8px; background: var(--amber-soft); color: var(--amber); font-weight: 900; }
    .empty-core strong { display: block; color: var(--ink); font-size: 16px; }
    .turn { margin: 0 auto 22px; max-width: 900px; }
    .turn-head { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; color: var(--muted); font-size: 11px; font-weight: 750; text-transform: uppercase; }
    .role-chip { width: 22px; height: 22px; display: grid; place-items: center; border-radius: 5px; color: #fff; font-size: 10px; }
    .turn.user .role-chip { background: var(--blue); }
    .turn.assistant .role-chip { background: var(--green); }
    .turn.error .role-chip { background: var(--red); }
    .image-chip { color: var(--purple); background: #f2edff; border-radius: 4px; padding: 2px 6px; text-transform: none; }
    .bubble {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      padding: 13px 15px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-soft);
      font-size: 14px;
    }
    .turn.user .bubble { border-color: #bfd0f7; background: var(--blue-soft); }
    .turn.assistant .bubble { border-color: #b9dfcf; background: #fbfffd; }
    .turn.error .bubble { border-color: #f3b5c2; background: var(--red-soft); color: #8d1632; }
    .cursor::after { content: ""; display: inline-block; width: 7px; height: 15px; margin-left: 3px; vertical-align: -2px; background: var(--green); animation: blink .8s steps(1) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .composer-wrap { padding: 14px clamp(18px, 4vw, 62px) 16px; border-top: 1px solid var(--line); background: var(--surface); }
    .composer { max-width: 960px; margin: auto; display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; }
    #messageInput { min-height: 58px; max-height: 180px; resize: vertical; padding: 11px 12px; }
    .send-button { height: 42px; min-width: 86px; border: 1px solid var(--green); border-radius: 6px; background: var(--green); color: white; font-weight: 760; }
    .send-button:hover { background: #066d4e; }
    .send-button:disabled { opacity: .48; cursor: not-allowed; }
    .composer-meta { max-width: 960px; margin: 8px auto 0; display: flex; justify-content: space-between; gap: 14px; color: var(--muted); font-size: 11px; }
    .icon-actions { display: flex; gap: 6px; }
    .icon-button {
      width: 34px;
      height: 32px;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
      font-size: 16px;
    }
    .icon-button:hover { border-color: var(--line-strong); background: var(--surface-soft); }
    .icon-button.stop { color: var(--red); }
    .icon-button:disabled { opacity: .35; cursor: not-allowed; }
    .control { margin-bottom: 14px; }
    .control:last-child { margin-bottom: 0; }
    .control-head { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 6px; color: var(--muted); font-size: 11px; }
    .control-head label { color: var(--ink); font-weight: 700; }
    .control output { color: var(--blue); font-weight: 800; }
    input[type="range"] { width: 100%; accent-color: var(--blue); }
    input[type="number"] { height: 34px; padding: 0 9px; }
    .toggle-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
    .switch { position: relative; width: 40px; height: 22px; flex: 0 0 auto; }
    .switch input { position: absolute; opacity: 0; }
    .switch span { position: absolute; inset: 0; border-radius: 11px; background: #c7d0d8; transition: .15s; }
    .switch span::after { content: ""; position: absolute; width: 16px; height: 16px; left: 3px; top: 3px; border-radius: 50%; background: white; transition: .15s; box-shadow: 0 1px 3px rgba(0,0,0,.25); }
    .switch input:checked + span { background: var(--purple); }
    .switch input:checked + span::after { transform: translateX(18px); }
    .mode-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 7px; border-radius: 5px; background: var(--amber-soft); color: var(--amber); font-size: 11px; font-weight: 800; }
    .mode-badge.vision { background: var(--green-soft); color: var(--green); }
    @media (max-width: 1080px) {
      .workspace { grid-template-columns: 238px minmax(380px, 1fr) 238px; }
      .rail { padding: 14px; }
    }
    @media (max-width: 820px) {
      body { overflow: auto; }
      .topbar { position: sticky; top: 0; z-index: 10; }
      .workspace { height: auto; display: grid; grid-template-columns: 1fr; }
      .rail { overflow: visible; }
      .rail.left { border-right: 0; border-bottom: 1px solid var(--line); display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .rail.right { border-left: 0; border-top: 1px solid var(--line); display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
      .rail-section { margin: 0; border: 0; padding: 0; }
      .chat-panel { min-height: 72dvh; }
      .messages { min-height: 54dvh; padding: 20px 16px; }
    }
    @media (max-width: 560px) {
      .runtime-copy { display: none; }
      .rail.left, .rail.right { grid-template-columns: 1fr; }
      .composer { grid-template-columns: 1fr; }
      .send-button { width: 100%; }
      .composer-meta { align-items: center; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">OA</div>
      <div><div class="brand-name">OpenAster1</div><div class="brand-sub">Inference Studio</div></div>
    </div>
    <div class="runtime">
      <div class="runtime-copy"><span id="modelName">loading model</span><span id="runtimeStatus">connecting</span></div>
      <span class="status-dot" id="statusDot"></span>
      <span class="mode-badge" id="modeBadge">...</span>
    </div>
  </header>

  <div class="workspace">
    <aside class="rail left">
      <section class="rail-section">
        <h2 class="section-title">Visual context</h2>
        <div class="dropzone" id="dropzone">
          <div class="drop-copy" id="dropCopy"><strong>Select an image</strong>PNG, JPEG, WEBP</div>
          <input id="imageInput" type="file" accept="image/png,image/jpeg,image/webp" aria-label="Select image">
        </div>
      </section>
      <section class="rail-section">
        <h2 class="section-title">System prompt</h2>
        <textarea id="systemPrompt" placeholder="Optional"></textarea>
      </section>
    </aside>

    <main class="chat-panel">
      <div class="messages" id="messages">
        <div class="empty-state" id="emptyState"><div class="empty-core"><div class="empty-mark">OA</div><strong>OpenAster1 is ready</strong></div></div>
      </div>
      <div class="composer-wrap">
        <form class="composer" id="chatForm">
          <textarea id="messageInput" placeholder="Ask OpenAster1" required></textarea>
          <button class="send-button" id="sendButton" type="submit">Send</button>
        </form>
        <div class="composer-meta">
          <span id="contextStatus">0 turns</span>
          <div class="icon-actions">
            <button class="icon-button stop" id="stopButton" type="button" title="Stop generation" disabled>&#9632;</button>
            <button class="icon-button" id="regenerateButton" type="button" title="Regenerate last answer" disabled>&#8635;</button>
            <button class="icon-button" id="clearButton" type="button" title="Clear conversation">&#215;</button>
          </div>
        </div>
      </div>
    </main>

    <aside class="rail right">
      <section class="rail-section">
        <h2 class="section-title">Generation</h2>
        <div class="control">
          <div class="control-head"><label for="maxNewTokens">Max new tokens</label></div>
          <input id="maxNewTokens" type="number" min="1" max="8192" value="512">
        </div>
        <div class="control">
          <div class="control-head"><label for="contextTokens">Context tokens</label></div>
          <input id="contextTokens" type="number" min="1024" max="131072" step="1024" value="32768">
        </div>
        <div class="control">
          <div class="control-head"><label for="temperature">Temperature</label><output id="temperatureValue">0.7</output></div>
          <input id="temperature" type="range" min="0" max="2" step="0.05" value="0.7">
        </div>
        <div class="control">
          <div class="control-head"><label for="topP">Top-p</label><output id="topPValue">0.90</output></div>
          <input id="topP" type="range" min="0.05" max="1" step="0.01" value="0.9">
        </div>
        <div class="control">
          <div class="control-head"><label for="topK">Top-k</label></div>
          <input id="topK" type="number" min="0" max="1000" value="50">
        </div>
        <div class="control">
          <div class="control-head"><label for="repetitionPenalty">Repetition penalty</label><output id="repetitionPenaltyValue">1.05</output></div>
          <input id="repetitionPenalty" type="range" min="0.5" max="2" step="0.01" value="1.05">
        </div>
        <div class="control">
          <div class="control-head"><label for="seed">Seed</label></div>
          <input id="seed" type="number" min="0" max="2147483647" value="42">
        </div>
      </section>
      <section class="rail-section">
        <div class="toggle-row"><div><h2 class="section-title">Thinking</h2></div><label class="switch"><input id="thinking" type="checkbox"><span></span></label></div>
      </section>
    </aside>
  </div>

  <script>
    const history = [];
    let imageData = null;
    let imageTurn = null;
    let controller = null;
    let busy = false;
    let currentBubble = null;
    let currentText = "";

    const byId = (id) => document.getElementById(id);
    const messagesEl = byId("messages");
    const emptyState = byId("emptyState");
    const form = byId("chatForm");
    const messageInput = byId("messageInput");
    const sendButton = byId("sendButton");
    const stopButton = byId("stopButton");
    const regenerateButton = byId("regenerateButton");
    const clearButton = byId("clearButton");
    const imageInput = byId("imageInput");
    const dropzone = byId("dropzone");
    const dropCopy = byId("dropCopy");
    const statusDot = byId("statusDot");
    const runtimeStatus = byId("runtimeStatus");

    function setRuntime(status, state = "") {
      runtimeStatus.textContent = status;
      statusDot.className = `status-dot ${state}`;
    }

    function updateContextStatus(meta = null) {
      const turns = history.filter((item) => item.role === "user").length;
      let text = `${turns} turn${turns === 1 ? "" : "s"}`;
      if (meta && Number.isFinite(meta.prompt_tokens)) text += ` · ${meta.prompt_tokens.toLocaleString()} prompt tokens`;
      if (meta && meta.dropped_turns) text += ` · ${meta.dropped_turns} trimmed`;
      byId("contextStatus").textContent = text;
    }

    function addTurn(role, text, withImage = false) {
      emptyState.hidden = true;
      const turn = document.createElement("article");
      turn.className = `turn ${role}`;
      const head = document.createElement("div");
      head.className = "turn-head";
      const chip = document.createElement("span");
      chip.className = "role-chip";
      chip.textContent = role === "assistant" ? "OA" : role === "user" ? "Y" : "!";
      const label = document.createElement("span");
      label.textContent = role === "assistant" ? "OpenAster1" : role === "user" ? "You" : "Error";
      head.append(chip, label);
      if (withImage) {
        const imageChip = document.createElement("span");
        imageChip.className = "image-chip";
        imageChip.textContent = "image";
        head.appendChild(imageChip);
      }
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      turn.append(head, bubble);
      messagesEl.appendChild(turn);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return { turn, bubble };
    }

    function clearConversation(clearImage = true) {
      history.length = 0;
      imageTurn = null;
      messagesEl.querySelectorAll(".turn").forEach((node) => node.remove());
      emptyState.hidden = false;
      regenerateButton.disabled = true;
      updateContextStatus();
      if (clearImage) setImage(null);
    }

    function setImage(dataUrl) {
      imageData = dataUrl;
      imageInput.value = "";
      dropzone.querySelectorAll("img").forEach((node) => node.remove());
      dropCopy.hidden = Boolean(dataUrl);
      if (dataUrl) {
        const image = document.createElement("img");
        image.src = dataUrl;
        image.alt = "Selected visual context";
        dropzone.insertBefore(image, imageInput);
      }
    }

    function loadImageFile(file) {
      if (!file) return;
      if (file.size > 20 * 1024 * 1024) {
        setRuntime("image exceeds 20 MiB", "error");
        return;
      }
      if (history.length) clearConversation(false);
      const reader = new FileReader();
      reader.onload = () => { setImage(reader.result); setRuntime("image ready", "ready"); };
      reader.readAsDataURL(file);
    }

    imageInput.addEventListener("change", () => loadImageFile(imageInput.files[0]));
    ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("drag"); }));
    dropzone.addEventListener("drop", (event) => loadImageFile(event.dataTransfer.files[0]));

    function samplingPayload() {
      return {
        max_new_tokens: Number(byId("maxNewTokens").value),
        context_tokens: Number(byId("contextTokens").value),
        temperature: Number(byId("temperature").value),
        top_p: Number(byId("topP").value),
        top_k: Number(byId("topK").value),
        repetition_penalty: Number(byId("repetitionPenalty").value),
        seed: Number(byId("seed").value)
      };
    }

    function setBusy(value) {
      busy = value;
      sendButton.disabled = value;
      stopButton.disabled = !value;
      regenerateButton.disabled = value || history.length < 2;
      if (!value) messageInput.focus();
    }

    async function generate(message) {
      const clean = message.trim();
      if (!clean || busy) return;
      history.push({ role: "user", content: clean });
      if (imageData && imageTurn === null) imageTurn = history.length - 1;
      addTurn("user", clean, imageTurn === history.length - 1);
      updateContextStatus();
      messageInput.value = "";
      const pending = addTurn("assistant", "");
      pending.bubble.classList.add("cursor");
      currentBubble = pending.bubble;
      currentText = "";
      controller = new AbortController();
      setBusy(true);
      setRuntime("generating", "busy");

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            messages: history,
            system_prompt: byId("systemPrompt").value,
            thinking: byId("thinking").checked,
            sampling: samplingPayload(),
            image_data: imageData,
            image_turn: imageTurn
          })
        });
        if (!response.ok) throw new Error(await response.text());
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let doneMeta = null;
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            if (event.type === "token") {
              currentText += event.text;
              pending.bubble.textContent = currentText;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            } else if (event.type === "done") {
              doneMeta = event;
            } else if (event.type === "error") {
              throw new Error(event.message);
            }
          }
        }
        pending.bubble.classList.remove("cursor");
        if (!currentText.trim()) throw new Error("The model returned an empty response.");
        history.push({ role: "assistant", content: currentText.trim() });
        regenerateButton.disabled = false;
        updateContextStatus(doneMeta);
        setRuntime("ready", "ready");
      } catch (error) {
        pending.bubble.classList.remove("cursor");
        if (error.name === "AbortError") {
          if (currentText.trim()) {
            history.push({ role: "assistant", content: currentText.trim() });
            pending.bubble.textContent = currentText.trim();
          } else {
            history.pop();
            pending.turn.remove();
          }
          setRuntime("stopped", "ready");
        } else {
          history.pop();
          pending.turn.remove();
          addTurn("error", String(error.message || error));
          setRuntime("generation failed", "error");
        }
        updateContextStatus();
      } finally {
        currentBubble = null;
        controller = null;
        setBusy(false);
      }
    }

    form.addEventListener("submit", (event) => { event.preventDefault(); generate(messageInput.value); });
    messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
    });
    stopButton.addEventListener("click", () => { if (controller) controller.abort(); });
    clearButton.addEventListener("click", () => { if (controller) controller.abort(); clearConversation(true); setRuntime("ready", "ready"); });
    regenerateButton.addEventListener("click", () => {
      if (busy || history.length < 2) return;
      const lastAnswer = history.pop();
      const lastUser = history.pop();
      if (!lastAnswer || !lastUser || lastUser.role !== "user") return;
      const turns = messagesEl.querySelectorAll(".turn");
      if (turns.length >= 2) { turns[turns.length - 1].remove(); turns[turns.length - 2].remove(); }
      generate(lastUser.content);
    });

    [["temperature", "temperatureValue", 1], ["topP", "topPValue", 2], ["repetitionPenalty", "repetitionPenaltyValue", 2]].forEach(([inputId, outputId, digits]) => {
      const input = byId(inputId);
      const output = byId(outputId);
      const sync = () => { output.textContent = Number(input.value).toFixed(digits); };
      input.addEventListener("input", sync); sync();
    });

    fetch("/health").then((response) => response.json()).then((data) => {
      byId("modelName").textContent = data.model;
      byId("modeBadge").textContent = data.mode;
      byId("modeBadge").classList.toggle("vision", data.mode === "vision");
      byId("contextTokens").max = data.max_context_tokens;
      setRuntime("ready", "ready");
    }).catch((error) => setRuntime(String(error), "error"));
    messageInput.focus();
  </script>
</body>
</html>
"""


def make_handler(engine: OpenAsterEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "OpenAsterHTTP/1.0"

        def _send_bytes(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_bytes(
                    HTTPStatus.OK,
                    HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            if path == "/health":
                body = json.dumps(
                    {
                        "ok": True,
                        "model": engine.model_name,
                        "mode": engine.kind,
                        "max_context_tokens": engine.max_context_tokens,
                    }
                ).encode("utf-8")
                self._send_bytes(HTTPStatus.OK, body, "application/json")
                return
            self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/chat":
                self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length < 1 or content_length > MAX_REQUEST_BYTES:
                self._send_bytes(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    b"invalid request size",
                    "text/plain",
                )
                return
            try:
                payload = json.loads(self.rfile.read(content_length))
                request = parse_chat_request(payload)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._send_bytes(
                    HTTPStatus.BAD_REQUEST,
                    str(exc).encode("utf-8"),
                    "text/plain; charset=utf-8",
                )
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            stop_event = threading.Event()

            def emit(event: bytes) -> bool:
                try:
                    self.wfile.write(event)
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    stop_event.set()
                    return False

            if not emit(serialize_event("start", model=engine.model_name, mode=engine.kind)):
                return
            try:
                for chunk in engine.stream(
                    request.messages,
                    request.sampling,
                    image=request.image_bytes,
                    image_turn=request.image_turn,
                    system_prompt=request.system_prompt,
                    thinking=request.thinking,
                    stop_event=stop_event,
                ):
                    if not emit(serialize_event("token", text=chunk)):
                        return
                result = engine.last_prompt_result
                emit(
                    serialize_event(
                        "done",
                        prompt_tokens=result.prompt_tokens if result else None,
                        dropped_turns=result.dropped_turns if result else 0,
                        retained_messages=len(result.messages) if result else None,
                    )
                )
            except Exception as exc:
                emit(serialize_event("error", message=str(exc)))

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} {fmt % args}")

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the OpenAster1 text/vision web chat.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_GUI_MODEL, help="Hub model ID or local path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
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
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"Loading {args.model} ...", flush=True)
    engine = OpenAsterEngine.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    url = f"http://{args.host}:{args.port}"
    print(f"OpenAster1 {engine.kind} chat: {url}", flush=True)
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
