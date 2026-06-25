# OpenAster-1 Release Manifest

## Weights

- `OpenAster1-4k-base`: 4K text base model before context extension.
- `OpenAster1-128k`: 128K text base model.
- `OpenAster1-Math`: math SFT checkpoint based on the 128K base.
- `OpenAster1-VL`: LLaVA-style vision tuned checkpoint.

## Data

- `OpenAster-1-data`: final 20B-token scratch pretraining mixture.
- The 60B continued-pretraining data is intentionally not included in this release manifest yet.

## Code

- `training/pretrain`: pretraining and context-extension launch templates.
- `training/post_training`: math SFT and text-only RL diagnostic launch templates.
- `training/vision`: vision tuning launch template.
- `eval`: text, math, and vision evaluation entry points.

All release scripts use environment variables instead of internal server paths.
