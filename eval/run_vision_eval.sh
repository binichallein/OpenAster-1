#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?set MODEL}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/vision_eval}
TASKS=${TASKS:-mme,pope,scienceqa_img,textvqa,mmbench_en_dev}

mkdir -p "$OUTPUT_DIR"

lmms-eval \
  --model llava \
  --model_args "pretrained=${MODEL},trust_remote_code=True" \
  --tasks "$TASKS" \
  --batch_size "${BATCH_SIZE:-1}" \
  --output_path "$OUTPUT_DIR"
