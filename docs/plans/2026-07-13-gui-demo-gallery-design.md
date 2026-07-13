# GUI Demo Gallery Design

## Goal

Present three separate, authentic GUI demonstrations: OpenAster1-VL visual understanding, OpenAster1-4k-base text Thinking, and OpenAster1-math mathematical reasoning with rendered formulas.

## Design

- Preserve the existing real visual recording and rename it `gui-vision-demo.gif`.
- Record `gui-text-thinking-demo.gif` from the public `OpenAster1-4k-base` checkpoint with Thinking enabled and the exact prompt `我想学习游泳，你能给我一些建议吗`.
- Record `gui-math-demo.gif` from the public `OpenAster1-math` checkpoint using GSM8K test example 208, whose verified answer is 76.
- Add MathJax configuration to the embedded GUI. Assistant text remains inserted with `textContent`; MathJax only typesets TeX delimiters after generation finishes, preserving the existing HTML-injection boundary.
- Update the README to show the three GIFs under explicit Vision, Text Thinking, and Math Reasoning headings.

## Verification

- Unit-test the MathJax configuration and completion hook.
- Confirm both new recordings show the public model ID, user prompt, live generation, and final answer.
- Confirm the math recording contains rendered formulas rather than raw TeX delimiters.
- Verify GIF dimensions, frame counts, duration, README references, full tests, and GitHub rendering paths.
