# OpenAster Inference Demos Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two public inference scripts that run all released OpenAster text and vision checkpoints, provide long multi-turn terminal and browser chat, and document real terminal/VL demo GIFs.

**Architecture:** `inference/inference.py` is both the terminal entry point and the shared inference backend. `inference/app.py` imports it and serves an embedded streaming web UI using only the Python standard library. Unit tests isolate prompt/history/sampling behavior; real-model smoke tests and GIF recording run on the `train` A800 host.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, Pillow, standard-library `argparse`/`http.server`, pytest, Playwright, FFmpeg.

---

### Task 1: Pure inference contracts

**Files:**
- Create: `tests/test_inference.py`
- Create: `inference/inference.py`

**Step 1: Write failing tests for model detection, image-token expansion, and sampling validation**

Add tests equivalent to:

```python
from inference.inference import SamplingConfig, detect_model_kind, expand_image_tokens


def test_detects_supported_model_families():
    assert detect_model_kind("qwen3_moe") == "text"
    assert detect_model_kind("llava") == "vision"


def test_expands_exactly_one_image_marker():
    rendered = "a<|image_pad|>b"
    assert expand_image_tokens(rendered, 3) == "a<|image_pad|><|image_pad|><|image_pad|>b"


def test_temperature_zero_disables_sampling():
    cfg = SamplingConfig(temperature=0.0)
    assert cfg.generation_kwargs()["do_sample"] is False
    assert "temperature" not in cfg.generation_kwargs()
```

Also cover unsupported model types, top-p/top-k/repetition-penalty ranges, context budget, and deterministic seed handling.

**Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_inference.py`

Expected: collection fails because `inference.inference` does not exist.

**Step 3: Implement the minimal pure contracts**

Create:

```python
@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    seed: int = 42
    context_tokens: int = 32768

    def validate(self) -> None: ...
    def generation_kwargs(self) -> dict[str, Any]: ...


def detect_model_kind(model_type: str) -> Literal["text", "vision"]: ...
def expand_image_tokens(text: str, image_seq_len: int) -> str: ...
```

**Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_inference.py`

Expected: all Task 1 tests pass.

**Step 5: Commit**

```bash
git add inference/inference.py tests/test_inference.py
git commit -m "feat: add OpenAster inference contracts"
```

### Task 2: Prompt rendering and long-history budgeting

**Files:**
- Modify: `tests/test_inference.py`
- Modify: `inference/inference.py`

**Step 1: Write failing tests for prompt rendering and whole-turn trimming**

Cover:

- system prompt stays first;
- `enable_thinking` is forwarded when the chat template supports it and gracefully retried when it does not;
- a vision prefix is attached to exactly one user turn;
- the expanded vision token count equals `image_seq_length`;
- trimming removes the oldest complete user/assistant pair, never half a turn;
- current user input is never discarded;
- an overlong current turn raises a useful `ContextLengthError`.

Use a deterministic fake tokenizer that counts whitespace-separated tokens.

**Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_inference.py -k 'prompt or history or context or vision'`

Expected: failures because prompt/history helpers are missing.

**Step 3: Implement prompt/history helpers**

Add:

```python
def validate_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]: ...
def render_prompt(tokenizer, messages, *, system_prompt, thinking, image_turn, image_seq_len): ...
def fit_messages_to_context(tokenizer, messages, *, render, context_tokens, max_new_tokens): ...
```

Return prompt text, retained messages, prompt token count, and dropped-turn count so both UIs can report context behavior.

**Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_inference.py`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add inference/inference.py tests/test_inference.py
git commit -m "feat: add long conversation prompt budgeting"
```

### Task 3: Model engine and terminal interface

**Files:**
- Modify: `tests/test_inference.py`
- Modify: `inference/inference.py`

**Step 1: Write failing tests for engine loading decisions and CLI behavior**

Cover:

- text config selects `AutoModelForCausalLM`;
- LLaVA config selects `LlavaForConditionalGeneration` and `CLIPImageProcessor`;
- text model rejects image input;
- generation kwargs include only sampling-compatible fields;
- `/clear`, `/image`, `/system`, `/params`, and `/exit` parse deterministically;
- one-shot mode accepts `--prompt` and optional `--image`;
- default text model is `binichallein/OpenAster1-math`.

Use dependency injection for model/tokenizer loaders so tests do not allocate tensors or download weights.

**Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_inference.py -k 'engine or cli or command or image'`

Expected: failures because engine and CLI behavior are missing.

**Step 3: Implement `OpenAsterEngine` and terminal REPL**

Implement:

```python
class OpenAsterEngine:
    @classmethod
    def from_pretrained(cls, model_name_or_path, *, device_map, dtype, attn_implementation): ...

    def stream(self, messages, sampling, *, image=None, image_turn=None, system_prompt="", thinking=False): ...
```

Use `TextIteratorStreamer`, a background generation thread, an inference lock, `torch.inference_mode()`, and model-config EOS/pad IDs. The REPL prints streamed output and only appends a successful response to history.

**Step 4: Run tests and syntax checks**

Run:

```bash
pytest -q tests/test_inference.py
python -m py_compile inference/inference.py
python inference/inference.py --help
```

Expected: tests pass, syntax check exits 0, help lists text/vision and sampling options.

**Step 5: Commit**

```bash
git add inference/inference.py tests/test_inference.py
git commit -m "feat: add terminal text and vision inference"
```

### Task 4: Streaming browser GUI

**Files:**
- Create: `tests/test_app.py`
- Create: `inference/app.py`

**Step 1: Write failing tests for request validation and required UI controls**

Cover:

- chat requests validate roles, message length, image size, image turn, and sampling values;
- GUI HTML contains image upload, system prompt, max tokens, context tokens, temperature, top-p, top-k, repetition penalty, seed, thinking toggle, stop, regenerate, and clear controls;
- API errors serialize as newline-delimited JSON error events;
- default GUI model is `binichallein/OpenAster1-VL`.

**Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_app.py`

Expected: collection fails because `inference.app` does not exist.

**Step 3: Implement the GUI server and embedded interface**

Create a responsive light interface with:

- neutral work surface and distinct green/blue/amber accents;
- left utility rail for image and system prompt;
- central scrolling conversation;
- right generation controls on wide screens and collapsible controls on mobile;
- assistant streaming state, token/turn status, image preview, clear, stop, and regenerate commands;
- `/health` JSON and `/api/chat` NDJSON endpoints.

Use `ThreadingHTTPServer`. Do not add FastAPI, Gradio, or Node dependencies.

**Step 4: Run tests and launch a fake-engine UI smoke test**

Run:

```bash
pytest -q tests/test_app.py
python -m py_compile inference/app.py
python inference/app.py --help
```

Then inject a deterministic fake engine in a short test process, open the page at desktop and mobile sizes, and verify no overlap or clipped controls with Playwright screenshots.

**Step 5: Commit**

```bash
git add inference/app.py tests/test_app.py
git commit -m "feat: add OpenAster streaming web chat"
```

### Task 5: Dependencies and README quickstarts

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

**Step 1: Write a failing documentation contract test**

Add a test that asserts README includes commands for:

- one-shot text inference;
- interactive terminal chat;
- one-shot vision inference;
- text GUI;
- vision GUI;
- both demo image paths.

**Step 2: Run the test and verify RED**

Run: `pytest -q tests -k readme`

Expected: failure because inference commands and GIF references are absent.

**Step 3: Update dependencies and README**

Add `pytest` only to a documented development/test extra command rather than the runtime requirements. Ensure runtime requirements already contain `torch`, `transformers`, `accelerate`, and `pillow`.

Add concise English and Chinese inference sections. Use commands such as:

```bash
python inference/inference.py --model binichallein/OpenAster1-math
python inference/inference.py --model binichallein/OpenAster1-VL --image examples/demo.jpg --prompt "Describe this image."
python inference/app.py --model binichallein/OpenAster1-VL --port 7860
```

Explain that selecting a new image starts a new visual conversation and that context trimming removes oldest complete turns.

**Step 4: Run tests and verify GREEN**

Run: `pytest -q tests`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add README.md requirements.txt tests
git commit -m "docs: add OpenAster inference quickstarts"
```

### Task 6: Real-model smoke tests and demo recording

**Files:**
- Create: `assets/terminal-demo.gif`
- Create: `assets/gui-demo.gif`
- Modify: `README.md`

**Step 1: Sync the feature worktree to `train`**

Use `rsync` over the configured `train` SSH host into a temporary user directory. Do not copy credentials or private paths into the repository.

**Step 2: Run real text smoke tests**

Run one-shot and multi-turn `OpenAster1-math` inference on an A800 with conservative generation limits. Verify generated text is non-empty, history is included in the second turn, and no CUDA/shape error occurs.

**Step 3: Run real vision smoke tests**

Use a public repository-safe image. Verify OpenAster1-VL produces a non-empty answer, image-token count is 576, and a follow-up question reuses the same image/history without shape errors.

**Step 4: Record the terminal GIF**

Capture an actual terminal session showing model startup, one math question, a follow-up, `/params`, and `/exit`. Render at a readable 1000px width and optimize with FFmpeg to stay below GitHub's practical inline-image limits.

**Step 5: Record the GUI GIF**

Start the GUI on `train`, forward the port locally, and automate the browser with Playwright. Capture image upload, first visual answer, parameter adjustment, and a follow-up turn. Assemble frames with FFmpeg and verify text remains legible.

**Step 6: Update README references and commit**

```bash
git add assets/terminal-demo.gif assets/gui-demo.gif README.md
git commit -m "docs: add terminal and vision chat demos"
```

### Task 7: Completion verification and integration

**Files:**
- Verify all modified files

**Step 1: Run the complete local suite**

Run:

```bash
pytest -q
python -m compileall -q inference tests
git diff --check main...HEAD
```

Expected: zero test failures, zero syntax failures, zero whitespace errors.

**Step 2: Verify README links and artifacts**

Check that both GIFs exist, are valid animated GIFs with multiple frames, render below the intended width, and are referenced by relative paths in README.

**Step 3: Review the final diff against the design**

Confirm exactly two public inference scripts, support for text and vision, long history, all requested sampling settings, terminal/GUI GIFs, and startup commands.

**Step 4: Integrate and push**

Use the finishing-a-development-branch workflow, merge the feature branch into `main`, rerun verification on `main`, and push `main` to `origin`.
