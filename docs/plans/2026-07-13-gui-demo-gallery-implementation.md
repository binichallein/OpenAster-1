# GUI Demo Gallery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish separate real GUI GIFs for vision, text Thinking, and math reasoning with rendered formulas.

**Architecture:** Keep `inference/app.py` as the only inference source file. Configure MathJax in the embedded HTML and invoke it after a completed assistant response, then record the two missing public-checkpoint sessions through the real HTTP GUI.

**Tech Stack:** Python, embedded HTML/CSS/JavaScript, MathJax 3, pytest, Playwright, FFmpeg.

---

### Task 1: Formula rendering

1. Add failing HTML-contract tests for MathJax delimiters and completion-time typesetting.
2. Run the focused test and confirm RED.
3. Add safe MathJax configuration and a `typesetMath` helper to `inference/app.py`.
4. Invoke typesetting only after the assistant stream completes or retains a partial response after an error.
5. Run GUI tests and a browser formula-rendering smoke test.

### Task 2: Real text and math recordings

1. Sync the current app to `train` and launch the 4K base checkpoint.
2. Record the exact Chinese swimming prompt with Thinking enabled.
3. Launch the Math checkpoint and record GSM8K example 208 with the verified sampling seed.
4. Convert both Playwright videos to optimized GIFs and inspect representative frames.

### Task 3: README gallery and release

1. Rename the current visual GIF and add the two new GIFs.
2. Replace the single README demo with three labeled sections and authenticity notes.
3. Update README contract tests.
4. Run all tests, compile checks, GIF metadata checks, and remote-content verification.
