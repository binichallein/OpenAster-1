# OpenAster-1

OpenAster-1 is a fully open 2B-scale Mixture-of-Experts language model project. The release includes model weights, the 20B-token scratch pretraining data mixture, training/evaluation scripts, and the technical report.

[Code](https://github.com/binichallein/OpenAster-1) | [Hugging Face Collection](https://huggingface.co/collections/binichallein/openaster-1) | [ModelScope Collection](https://www.modelscope.cn/collections/TYFTYF/OpenAster-1) | [Paper](./OpenAster-1_Technical_Report.pdf)

## Highlights

- **Fully open release**: weights, data, recipes, training scripts, and evaluation scripts are released together.
- **Low-cost training target**: OpenAster-1 is trained from scratch and extended with about 100B total training tokens.
- **MoE text backbone**: 2B total parameters with about 1.3B active parameters.
- **Long context**: the released 128K base checkpoint uses a YaRN-style long-context curriculum.
- **Vision tuning branch**: OpenAster-VL follows a LLaVA-style CLIP/projector/decoder recipe.

## Model Zoo

| Model | Type | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| OpenAster-1 128K Base | text base | [binichallein/openaster1-128k](https://huggingface.co/binichallein/openaster1-128k) | [TYFTYF/openaster1-128k](https://www.modelscope.cn/models/TYFTYF/openaster1-128k) |
| OpenAster-Math | math SFT | [binichallein/openaster1-math](https://huggingface.co/binichallein/openaster1-math) | [TYFTYF/openaster1-math](https://www.modelscope.cn/models/TYFTYF/openaster1-math) |
| OpenAster-VL | vision instruction tuned | [binichallein/openaster-vl](https://huggingface.co/binichallein/openaster-vl) | [TYFTYF/openaster-vl](https://www.modelscope.cn/models/TYFTYF/openaster-vl) |

## Data

The 20B-token scratch pretraining mixture is released as `OpenAster-1-data`:

- Hugging Face: [binichallein/OpenAster-1-data](https://huggingface.co/datasets/binichallein/OpenAster-1-data)
- ModelScope: [TYFTYF/OpenAster-1-data](https://www.modelscope.cn/datasets/TYFTYF/OpenAster-1-data)

The 20B mix contains DCLM Edu, OpenCSG FineWeb Edu Chinese 4/5, FineMath 4+, FineWeb Edu EN, Chinese FineWeb Edu, and Pile Deduplicated replay data. Records use JSONL with `text`, `source`, and `language` fields.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "binichallein/openaster1-128k"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "Give a short introduction to OpenAster-1."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

## Training

Release scripts are in `training/`:

- `training/pretrain/run_pretrain_megatron.sh`: scratch/continued pretraining and YaRN context extension entry point.
- `training/post_training/run_math_sft.sh`: math SFT entry point for OpenAster-Math.
- `training/post_training/run_text_rl_diagnostics.sh`: text-only RL diagnostic branches.
- `training/vision/run_llava_style_vision_tuning.sh`: LLaVA-style vision tuning.

All scripts use environment variables for data/model paths and intentionally avoid private server paths.

## Evaluation

Evaluation scripts are in `eval/` and cover the benchmarks used in the report:

- Text/basic: MMLU, C-Eval, C-MMLU, OpenBookQA, ARC, HellaSwag, BoolQ, BBH.
- Math: GSM8K and MATH500.
- Vision: MME, POPE, ScienceQA-IMG, TextVQA, and MMBench-EN dev.

See `eval/configs/openaster_eval_tasks.yaml` for task names and suggested commands.

## 中文说明

OpenAster-1 是一个完全开源的 2B 级 MoE 大语言模型项目。我们同时开源模型权重、20B token 从零预训练数据、训练脚本、评测脚本和技术报告，目标是让小规模团队也能复现一个具备长上下文和视觉分支的开放模型。

### 模型

| 模型 | 类型 | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| OpenAster-1 128K Base | 文本基座 | [binichallein/openaster1-128k](https://huggingface.co/binichallein/openaster1-128k) | [TYFTYF/openaster1-128k](https://www.modelscope.cn/models/TYFTYF/openaster1-128k) |
| OpenAster-Math | 数学 SFT | [binichallein/openaster1-math](https://huggingface.co/binichallein/openaster1-math) | [TYFTYF/openaster1-math](https://www.modelscope.cn/models/TYFTYF/openaster1-math) |
| OpenAster-VL | 视觉指令微调 | [binichallein/openaster-vl](https://huggingface.co/binichallein/openaster-vl) | [TYFTYF/openaster-vl](https://www.modelscope.cn/models/TYFTYF/openaster-vl) |

### 数据

20B token 从零预训练数据发布在：

- Hugging Face: [binichallein/OpenAster-1-data](https://huggingface.co/datasets/binichallein/OpenAster-1-data)
- ModelScope: [TYFTYF/OpenAster-1-data](https://www.modelscope.cn/datasets/TYFTYF/OpenAster-1-data)

数据以 `jsonl.zst` 分片发布，每条样本包含 `text`、`source`、`language` 字段。

## License

Code in this repository is released under the MIT License. Dataset users should also respect the licenses and terms of the upstream data sources listed in the dataset card.
