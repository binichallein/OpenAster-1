# Pretraining

`run_pretrain_megatron.sh` is a sanitized Megatron-style pretraining launch template for OpenAster-1. It covers:

- 20B-token scratch pretraining.
- Continued pretraining with replay/web/math/code/multilingual mixtures.
- 32K to 64K to 128K context extension by changing `SEQ_LEN` and the data prefix.

The released script intentionally uses environment variables for all paths:

```bash
DATA_PREFIX=/path/to/megatron/data_prefix \
TOKENIZER_DIR=/path/to/tokenizer \
OUTPUT_DIR=/path/to/checkpoints \
SEQ_LEN=131072 \
bash training/pretrain/run_pretrain_megatron.sh
```

The 20B data mixture is described in `openaster_pretrain_config.yaml`.
