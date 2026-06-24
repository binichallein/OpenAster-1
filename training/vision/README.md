# Vision Tuning

OpenAster-VL follows a LLaVA-style recipe:

1. Load the OpenAster text checkpoint.
2. Attach a CLIP-compatible vision tower.
3. Train a projector on image-text pairs.
4. Run full visual instruction tuning.

`run_llava_style_vision_tuning.sh` is the sanitized launch entry point. It expects a local or hub `TEXT_MODEL`, a JSONL conversation file, and an output directory. The JSONL rows should contain `messages` and either direct image paths (`image`/`images`) or zip-backed fields (`image_zip`/`image_rel`).

The included trainer saves both model files and resumable training state (`training_state.pt` plus per-rank RNG state) at checkpoints.
