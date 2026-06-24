# Post-Training

This directory contains sanitized release entry points for the OpenAster post-training branches.

- `run_math_sft.sh` launches math supervised fine-tuning.
- `sft_train.py` is a minimal causal-LM SFT trainer for JSON/JSONL files with a `text` field.
- `run_text_rl_diagnostics.sh` records the GRPO/DPO diagnostic interface used in internal experiments. The released OpenAster-1 weights do not require running these diagnostic branches.

Use environment variables for all local paths. Do not hard-code cluster-specific paths in release configs.
