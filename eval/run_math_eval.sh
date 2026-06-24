#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?set MODEL}
DATASET=${DATASET:-gsm8k}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/math_eval}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.0}

mkdir -p "$OUTPUT_DIR"

python eval_math_generate.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE"
