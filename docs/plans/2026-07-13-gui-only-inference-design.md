# GUI-Only Inference Design

## Goal

Make the browser GUI the only supported OpenAster inference experience. The `inference/` directory must contain exactly one source file, `app.py`, while retaining text and vision inference, long conversations, streaming output, image upload, thinking control, and sampling settings.

## Approaches Considered

### 1. Merge the inference engine into `app.py`

Move the reusable model-loading, prompt-building, context-budgeting, and streaming code into `app.py`. Remove terminal-only parsing and REPL code. This is the selected approach because it satisfies the one-file requirement without weakening the GUI.

### 2. Keep a private backend module

Rename `inference.py` to `_engine.py` and expose only `app.py` in documentation. This would keep the source cleaner, but it violates the requirement that `inference/` contain only `app.py`.

### 3. Generate a bundled `app.py` from multiple source files

Maintain separate source modules elsewhere and build a generated single-file artifact. This adds build tooling and creates two representations of the same code, which is unnecessary for this repository.

## Architecture

`inference/app.py` will contain four ordered layers:

1. Inference contracts and prompt helpers: sampling validation, message validation, chat-template rendering, image-token expansion, and whole-turn context trimming.
2. `OpenAsterEngine`: model-family detection, text/vision checkpoint loading, image preprocessing, seeded generation, and token streaming.
3. GUI transport: request validation, NDJSON events, HTTP handlers, and server lifecycle.
4. Embedded HTML/CSS/JavaScript and the GUI-only command-line launcher used to select the model, host, port, device, dtype, and attention implementation.

The launcher starts a browser application only. It will not accept a one-shot prompt, print a terminal chat banner, or expose REPL commands.

## Repository Surface

- Keep `inference/app.py` as the only file under `inference/`.
- Delete `inference/inference.py` and `assets/terminal-demo.gif`.
- Remove terminal commands, terminal screenshots, and terminal capability descriptions from the English and Chinese README sections.
- Remove obsolete inference design documents that describe a two-script terminal/GUI interface.
- Keep `assets/gui-demo.gif` and document text and vision GUI launch commands.

## Compatibility

The GUI keeps the existing public behavior:

- automatic detection of `qwen3_moe` text and `llava` vision checkpoints;
- Hugging Face IDs and local checkpoint paths;
- streaming generation and generation cancellation;
- long multi-turn history with complete-turn eviction at the context boundary;
- one persistent image per visual conversation;
- system prompt, thinking mode, max tokens, temperature, top-p, top-k, repetition penalty, seed, and context budget;
- host, port, device, dtype, attention implementation, trust-remote-code, and optional browser opening controls.

## Testing

- Move inference-engine tests to import from `inference.app`.
- Delete tests for terminal parsers, REPL commands, one-shot output, and terminal README content.
- Add a repository contract asserting that `inference/` contains only `app.py` and that no terminal demo is referenced.
- Retain unit coverage for model loading, prompt rendering, context trimming, image handling, sampling, request validation, and GUI controls.
- Run the full test suite, compile `app.py`, verify `python inference/app.py --help`, launch a local fake-engine server, and inspect desktop/mobile browser screenshots.
