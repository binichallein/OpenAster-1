#!/usr/bin/env bash
set -euo pipefail

# Math SFT launch template for OpenAster-Math.
# Required: BASE_MODEL, TRAIN_FILE, OUTPUT_DIR.

BASE_MODEL=${BASE_MODEL:?set BASE_MODEL}
TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29611}

accelerate launch --num_processes "$GPUS_PER_NODE" --main_process_port "$MASTER_PORT" \
  training/post_training/sft_train.py \
  --model_name_or_path "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length "${MAX_SEQ_LENGTH:-20000}" \
  --num_train_epochs "${EPOCHS:-5}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-1.0e-5}" \
  --bf16 true \
  --gradient_checkpointing true \
  --save_steps "${SAVE_STEPS:-1000}" \
  --logging_steps 10
