#!/usr/bin/env bash
set -euo pipefail

# Text-only RL diagnostic recorder used for GRPO/DPO-style experiments.
# The released OpenAster-1 models do not require running these branches.

BASE_MODEL=${BASE_MODEL:?set BASE_MODEL}
TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
METHOD=${METHOD:-grpo}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/rl_diagnostic_config.py" \
  --method "$METHOD" \
  --model "$BASE_MODEL" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-4096}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-8192}" \
  --num-generations "${NUM_GENERATIONS:-8}" \
  --beta "${DPO_BETA:-0.1}"
