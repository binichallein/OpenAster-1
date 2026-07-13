# OpenAster Inference Demos Design

## Goal

Add a production-shaped inference experience to the OpenAster-1 release with only two user-facing Python scripts: one terminal entry point and one browser GUI. Both scripts must support the released text and vision checkpoints, multi-turn conversation, configurable sampling, and reproducible README demos.

## Constraints

- Keep the public surface to two scripts under `inference/`.
- Run directly against Hugging Face or local model paths.
- Support `Qwen3MoeForCausalLM` text checkpoints and the released `LlavaForConditionalGeneration` vision checkpoint.
- Preserve conversation history while dropping the oldest complete turns when the configured context budget is exceeded.
- Stream generated text in both terminal and GUI modes.
- Avoid private paths and server-specific assumptions.
- Record real OpenAster model output for the README GIFs.

## Approaches Considered

### 1. Shared Python backend plus custom standard-library web GUI

`inference/inference.py` owns model loading, prompt rendering, context budgeting, image preprocessing, streaming generation, and the terminal REPL. `inference/app.py` imports that backend and serves an embedded responsive HTML interface with Python's standard HTTP server.

This is the selected approach. It keeps model behavior in one place, avoids an additional GUI dependency, permits a polished project-specific interface, and still satisfies the two-script limit.

### 2. Shared backend plus Gradio

Gradio would reduce browser plumbing and provide built-in chat/image components. It would add a large dependency, make the visual result less distinctive, and introduce version-sensitive component behavior. It is not selected.

### 3. Separate text and vision demos

Independent scripts would be initially simple, but model loading, sampling validation, history handling, and documentation would be duplicated. It also scales poorly when users switch between OpenAster checkpoints. It is not selected.

## Architecture

### `inference/inference.py`

- Detects the model family from `AutoConfig.model_type`.
- Loads text checkpoints with `AutoModelForCausalLM` and vision checkpoints with `LlavaForConditionalGeneration` plus `CLIPImageProcessor`.
- Exposes a small `OpenAsterEngine` API used by both terminal and GUI code.
- Renders the tokenizer chat template and expands the single LLaVA image token to the model's configured image sequence length.
- Keeps an image attached to the turn where it was introduced, allowing follow-up questions over the same image.
- Drops the oldest user/assistant turn pair until prompt tokens plus requested output fit the selected context window.
- Streams output with `TextIteratorStreamer`.
- Provides a terminal REPL with `/image`, `/clear`, `/system`, `/params`, and `/exit` commands, plus a one-shot `--prompt` mode.

### `inference/app.py`

- Loads one model at process startup and exposes health and streaming chat endpoints.
- Embeds the complete HTML, CSS, and JavaScript interface in the Python file.
- Shows the active model, text/vision mode, image preview, retained-turn/token status, and generation state.
- Supports image upload, long multi-turn history, system prompt editing, sampling controls, stop, regenerate, and clear actions.
- Streams newline-delimited JSON events so the assistant response updates token by token.

## Conversation and Image Data Flow

1. The client sends the complete visible message history, optional image, image turn index, system prompt, and generation settings.
2. The backend validates role ordering and parameter ranges.
3. The engine associates the image marker with exactly one user turn, applies the model chat template, and expands image tokens for OpenAster1-VL.
4. Whole oldest turn pairs are removed if the prompt exceeds the requested context budget.
5. Text inputs and optional `pixel_values` are passed to `generate()`.
6. Streamed text chunks are emitted to the terminal or browser and the final answer is appended to client history.

The first public version supports one persistent image per conversation. Selecting a different image starts a new visual conversation, which avoids ambiguous image-to-token alignment.

## Sampling Surface

Both entry points expose:

- maximum new tokens
- temperature
- top-p
- top-k
- repetition penalty
- random seed
- thinking on/off when supported by the tokenizer template
- context-token budget

Temperature zero selects greedy decoding; positive temperature enables sampling.

## Error Handling

- Reject unsupported model types with a concrete message.
- Reject image input when the loaded model is text-only.
- Validate image size and decoding before inference.
- Validate sampling ranges before starting generation.
- Surface CUDA availability and out-of-memory errors without hiding the original cause.
- Stop browser generation when the client aborts the request.
- Keep history unchanged when generation fails.

## Testing

Unit tests use lightweight fake tokenizers/configs and do not download model weights. They cover model-family detection, sampling validation, image-token expansion, chat-template fallback, whole-turn context trimming, request validation, and GUI HTML controls. Runtime smoke tests load the real text and vision releases on the `train` A800 host.

## Demo Assets and README

- `assets/terminal-demo.gif`: real `OpenAster1-math` multi-turn terminal session.
- `assets/gui-demo.gif`: real `OpenAster1-VL` upload, visual answer, and follow-up session.
- README commands cover installation, one-shot text, interactive text, one-shot vision, text GUI, and vision GUI.
- GIFs are referenced with relative repository paths and kept small enough for direct GitHub rendering.
