#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?set MODEL}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/text_eval}
TASKS=${TASKS:-mmlu,ceval-valid,cmmlu,openbookqa,arc_easy,arc_challenge,hellaswag,boolq,leaderboard_bbh}
NUM_FEWSHOT=${NUM_FEWSHOT:-5}

mkdir -p "$OUTPUT_DIR"

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16" \
  --tasks "$TASKS" \
  --num_fewshot "$NUM_FEWSHOT" \
  --batch_size "${BATCH_SIZE:-auto}" \
  --output_path "$OUTPUT_DIR"
