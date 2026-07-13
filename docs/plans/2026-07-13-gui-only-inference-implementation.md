# GUI-Only Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `inference/app.py` the sole OpenAster inference source file and the browser GUI the only supported user interface.

**Architecture:** Move the reusable inference contracts and `OpenAsterEngine` from `inference/inference.py` into the top of `inference/app.py`, ahead of the HTTP and embedded-UI layers. Delete all terminal-only code and assets while preserving text/vision model support and GUI behavior.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, Pillow, standard-library HTTP server, embedded HTML/CSS/JavaScript, pytest, Playwright.

---

### Task 1: Define the GUI-only repository contract

**Files:**
- Modify: `tests/test_readme.py`
- Modify: `tests/test_inference.py`

**Step 1: Write failing repository-surface tests**

Replace terminal documentation assertions with contracts equivalent to:

```python
def test_readme_documents_only_gui_inference():
    assert "python inference/app.py --model binichallein/OpenAster1-math" in README
    assert "python inference/app.py --model binichallein/OpenAster1-VL" in README
    assert "inference/inference.py" not in README
    assert "assets/terminal-demo.gif" not in README


def test_release_has_one_inference_source_file():
    scripts = sorted(path.name for path in (ROOT / "inference").iterdir())
    assert scripts == ["app.py"]
```

Update engine imports in `tests/test_inference.py` to target `inference.app`. Remove only terminal-specific imports and tests: `DEFAULT_TEXT_MODEL`, `parse_repl_command`, `update_sampling_config`, and the terminal `build_parser` expectations.

**Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_readme.py tests/test_inference.py`

Expected: failures because `inference/inference.py`, terminal README content, and terminal GIF still exist.

**Step 3: Commit the failing contract tests**

```bash
git add tests/test_readme.py tests/test_inference.py
git commit -m "test: require GUI-only inference surface"
```

### Task 2: Merge the inference engine into `app.py`

**Files:**
- Modify: `inference/app.py`
- Delete: `inference/inference.py`

**Step 1: Move backend imports and contracts**

Add the backend dependencies needed by the existing engine (`os`, `queue`, `Path`, and typing helpers). Move these definitions into `app.py` before `ChatRequest`:

- `IMAGE_TOKEN`, `VISION_PREFIX`
- `ModelLoaders`, `ContextLengthError`, `PromptResult`, `SamplingConfig`
- `detect_model_kind`, `expand_image_tokens`, `validate_messages`
- prompt rendering and context fitting helpers
- model/image loader helpers and seeded generation helper
- `OpenAsterEngine`

Do not move terminal-only constants, REPL parsers, sampling mutation helpers, terminal banners, one-shot execution, REPL execution, or terminal `main()`.

**Step 2: Remove the cross-module import**

Delete the `try/except` import of `OpenAsterEngine`, `SamplingConfig`, and `validate_messages`. The GUI request and server layers use the definitions in the same file.

**Step 3: Delete the terminal source file**

Delete `inference/inference.py`. Confirm `inference/` contains only `app.py`.

**Step 4: Run focused tests and verify GREEN**

Run:

```bash
pytest -q tests/test_inference.py tests/test_app.py
python -m py_compile inference/app.py
python inference/app.py --help
```

Expected: engine and GUI tests pass; help exposes only GUI server/model runtime arguments.

**Step 5: Commit**

```bash
git add inference/app.py inference/inference.py tests/test_inference.py
git commit -m "refactor: consolidate inference into GUI app"
```

### Task 3: Remove terminal documentation and assets

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme.py`
- Delete: `assets/terminal-demo.gif`
- Delete: `docs/plans/2026-07-13-inference-demos-design.md`
- Delete: `docs/plans/2026-07-13-inference-demos-implementation.md`

**Step 1: Rewrite the inference documentation**

Describe `inference/app.py` as the single GUI entry point. Keep the GUI GIF and provide text, 128K text, and vision launch commands using `app.py`. Explain that model selection determines text or vision mode and that all conversation/sampling controls live in the GUI.

Remove terminal chat headings, one-shot commands, REPL commands, terminal authenticity text, and Chinese terminal examples.

**Step 2: Delete obsolete terminal artifacts**

Delete the terminal GIF and the two superseded design documents that specify a terminal interface.

**Step 3: Run documentation contracts**

Run: `pytest -q tests/test_readme.py`

Expected: all README and repository-surface tests pass.

**Step 4: Commit**

```bash
git add README.md tests/test_readme.py assets/terminal-demo.gif docs/plans
git commit -m "docs: retire terminal inference interface"
```

### Task 4: Full verification and browser smoke test

**Files:**
- Test: `tests/test_app.py`
- Test: `tests/test_inference.py`
- Test: `tests/test_readme.py`

**Step 1: Run the full automated suite**

Run:

```bash
pytest -q
python -m compileall -q inference tests
git diff --check
```

Expected: all tests pass with no compile or whitespace errors.

**Step 2: Verify the public surface**

Run:

```bash
find inference -maxdepth 1 -type f -printf '%f\n'
rg -n "inference/inference.py|terminal-demo|Terminal chat|terminal REPL" README.md inference tests docs
python inference/app.py --help
```

Expected: only `app.py`; no active terminal-interface references; GUI-only help output.

**Step 3: Launch a fake-engine GUI server**

Start the HTTP server with a deterministic fake engine, open it in Playwright at desktop and mobile viewports, submit a multi-turn text request, and confirm streaming, history, controls, and layout remain functional without loading model weights.

**Step 4: Review and commit any verification-only corrections**

Run `git status --short` and commit only corrections required by the verification results.

