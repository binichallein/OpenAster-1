#!/usr/bin/env bash
set -euo pipefail

# LLaVA-style vision tuning entry point for OpenAster-VL.
# Required: TEXT_MODEL, TRAIN_JSON, OUTPUT_DIR.
# Optional: IMAGE_ZIP for zip-backed image storage.

TEXT_MODEL=${TEXT_MODEL:?set TEXT_MODEL}
TRAIN_JSON=${TRAIN_JSON:?set TRAIN_JSON}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

torchrun --nproc_per_node="${GPUS_PER_NODE:-8}" --master_port="${MASTER_PORT:-29631}" \
  "$SCRIPT_DIR/train_vlm.py" \
  --stage "${STAGE:-mm_sft}" \
  --model "$TEXT_MODEL" \
  --train-jsonl "$TRAIN_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --image-zip "${IMAGE_ZIP:-}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --image-seq-len "${IMAGE_SEQ_LEN:-576}" \
  --epochs "${EPOCHS:-1}" \
  --per-device-batch-size "${PER_DEVICE_BATCH:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --learning-rate "${LR:-2.0e-5}" \
  --save-steps "${SAVE_STEPS:-1000}" \
  --logging-steps "${LOGGING_STEPS:-10}" \
  --gradient-checkpointing
