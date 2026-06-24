#!/usr/bin/env bash
set -euo pipefail

# Sanitized Megatron-style launch template for OpenAster-1.
# Required environment variables:
#   DATA_PREFIX: Megatron indexed-data prefix without .bin/.idx suffix
#   TOKENIZER_DIR: tokenizer directory
#   OUTPUT_DIR: checkpoint output directory
# Optional:
#   WANDB_PROJECT, GPUS_PER_NODE, MASTER_PORT, TP_SIZE, PP_SIZE, SEQ_LEN

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29577}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-32768}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
TRAIN_TOKENS=${TRAIN_TOKENS:-20000000000}
LR=${LR:-3.0e-4}
MIN_LR=${MIN_LR:-3.0e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}

: "${DATA_PREFIX:?set DATA_PREFIX}"
: "${TOKENIZER_DIR:?set TOKENIZER_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" \
  pretrain_gpt.py \
  --tensor-model-parallel-size "$TP_SIZE" \
  --pipeline-model-parallel-size "$PP_SIZE" \
  --sequence-parallel \
  --use-distributed-optimizer \
  --bf16 \
  --seq-length "$SEQ_LEN" \
  --max-position-embeddings "$SEQ_LEN" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --train-tokens "$TRAIN_TOKENS" \
  --lr "$LR" \
  --min-lr "$MIN_LR" \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --data-path "$DATA_PREFIX" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$TOKENIZER_DIR" \
  --save "$OUTPUT_DIR" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval 10 \
  --eval-interval 1000 \
  --eval-iters 10
