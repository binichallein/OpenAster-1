# OpenAster-1

OpenAster-1 is a fully open 2B-scale Mixture-of-Experts language model project. The release includes model weights, the 20B-token scratch pretraining data mixture, training/evaluation scripts, and the technical report.

[Code](https://github.com/binichallein/OpenAster-1) | [Hugging Face Collection](https://huggingface.co/collections/binichallein/openaster-1) | [ModelScope Collection](https://www.modelscope.cn/collections/TYFTYF/OpenAster-1) | [Paper](./OpenAster-1_Technical_Report.pdf)

## Highlights

- **Fully open release**: weights, data, recipes, training scripts, and evaluation scripts are released together.
- **Low-cost training target**: OpenAster-1 is trained from scratch and extended with about 100B total training tokens.
- **MoE text backbone**: 2B total parameters with about 1.3B active parameters.
- **Long context**: the released 128K base checkpoint uses a YaRN-style long-context curriculum.
- **Vision tuning branch**: OpenAster1-VL follows a LLaVA-style CLIP/projector/decoder recipe.

## Model Zoo

| Model | Type | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| OpenAster1-4k-base | 4K text base | [binichallein/OpenAster1-4k-base](https://huggingface.co/binichallein/OpenAster1-4k-base) | [TYFTYF/OpenAster1-4k-base](https://www.modelscope.cn/models/TYFTYF/OpenAster1-4k-base) |
| OpenAster1-128k-base | 128K text base | [binichallein/OpenAster1-128k-base](https://huggingface.co/binichallein/OpenAster1-128k-base) | [TYFTYF/OpenAster1-128k-base](https://www.modelscope.cn/models/TYFTYF/OpenAster1-128k-base) |
| OpenAster1-math | math SFT | [binichallein/OpenAster1-math](https://huggingface.co/binichallein/OpenAster1-math) | [TYFTYF/OpenAster1-math](https://www.modelscope.cn/models/TYFTYF/OpenAster1-math) |
| OpenAster1-VL | vision instruction tuned | [binichallein/OpenAster1-VL](https://huggingface.co/binichallein/OpenAster1-VL) | [TYFTYF/OpenAster1-VL](https://www.modelscope.cn/models/TYFTYF/OpenAster1-VL) |

## Data

The 20B-token scratch pretraining mixture is released as `OpenAster-1-data`:

- Hugging Face: [binichallein/OpenAster-1-data](https://huggingface.co/datasets/binichallein/OpenAster-1-data)
- ModelScope: [TYFTYF/OpenAster-1-data](https://www.modelscope.cn/datasets/TYFTYF/OpenAster-1-data)

The 20B mix contains DCLM Edu, OpenCSG FineWeb Edu Chinese 4/5, FineMath 4+, FineWeb Edu EN, Chinese FineWeb Edu, and Pile Deduplicated replay data. Records use JSONL with `text`, `source`, and `language` fields.

## Inference

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

OpenAster provides one inference entry point: `inference/app.py`. It launches the browser GUI and automatically detects text (`Qwen3MoeForCausalLM`) and vision (`LlavaForConditionalGeneration`) checkpoints.

### Browser GUI

#### Vision

<p align="center">
  <img src="assets/gui-vision-demo.gif" alt="OpenAster1-VL visual conversation" width="1000">
</p>

This recording runs the public `binichallein/OpenAster1-VL` checkpoint with a real uploaded image and visual follow-up question.

#### Text Thinking

<p align="center">
  <img src="assets/gui-text-thinking-demo.gif" alt="OpenAster1-4k-base text Thinking conversation" width="1000">
</p>

This recording runs the public `binichallein/OpenAster1-4k-base` checkpoint with Thinking enabled. The exact user prompt is `我想学习游泳，你能给我一些建议吗`.

#### Math Reasoning

<p align="center">
  <img src="assets/gui-math-demo.gif" alt="OpenAster1-math GSM8K reasoning with rendered formulas" width="1000">
</p>

This recording runs the public `binichallein/OpenAster1-math` checkpoint on GSM8K test example 208 with `temperature=0.7`, `top_p=0.95`, and `seed=3`. The model derives the correct answer `76`, and the GUI renders the generated LaTeX equations and `\boxed{76}` after streaming completes.

Launch the 4K base model:

```bash
python inference/app.py --model binichallein/OpenAster1-4k-base --port 7860 --open-browser
```

Launch OpenAster1-math:

```bash
python inference/app.py --model binichallein/OpenAster1-math --port 7860 --open-browser
```

Launch the 128K base model:

```bash
python inference/app.py --model binichallein/OpenAster1-128k-base --port 7860 --open-browser
```

Set **Context tokens** to `131072` in the GUI when the full 128K context budget is required.

Launch OpenAster1-VL:

```bash
python inference/app.py --model binichallein/OpenAster1-VL --host 0.0.0.0 --port 7860 --open-browser
```

Open `http://localhost:7860`. The GUI streams tokens, preserves multi-turn history, exposes the sampling controls above, and keeps one image attached across visual follow-up turns. Selecting a new image starts a new visual conversation. If a prompt approaches the selected context budget, the oldest complete user/assistant pairs are removed while the current turn and visual anchor turn are retained.

## Training

Release scripts are in `training/`:

- `training/pretrain/run_pretrain_megatron.sh`: scratch/continued pretraining and YaRN context extension entry point.
- `training/post_training/run_math_sft.sh`: math SFT entry point for OpenAster1-math.
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
| OpenAster1-4k-base | 4K 文本基座 | [binichallein/OpenAster1-4k-base](https://huggingface.co/binichallein/OpenAster1-4k-base) | [TYFTYF/OpenAster1-4k-base](https://www.modelscope.cn/models/TYFTYF/OpenAster1-4k-base) |
| OpenAster1-128k-base | 128K 文本基座 | [binichallein/OpenAster1-128k-base](https://huggingface.co/binichallein/OpenAster1-128k-base) | [TYFTYF/OpenAster1-128k-base](https://www.modelscope.cn/models/TYFTYF/OpenAster1-128k-base) |
| OpenAster1-math | 数学 SFT | [binichallein/OpenAster1-math](https://huggingface.co/binichallein/OpenAster1-math) | [TYFTYF/OpenAster1-math](https://www.modelscope.cn/models/TYFTYF/OpenAster1-math) |
| OpenAster1-VL | 视觉指令微调 | [binichallein/OpenAster1-VL](https://huggingface.co/binichallein/OpenAster1-VL) | [TYFTYF/OpenAster1-VL](https://www.modelscope.cn/models/TYFTYF/OpenAster1-VL) |

### 数据

20B token 从零预训练数据发布在：

- Hugging Face: [binichallein/OpenAster-1-data](https://huggingface.co/datasets/binichallein/OpenAster-1-data)
- ModelScope: [TYFTYF/OpenAster-1-data](https://www.modelscope.cn/datasets/TYFTYF/OpenAster-1-data)

数据以 `jsonl.zst` 分片发布，每条样本包含 `text`、`source`、`language` 字段。

### 推理

文本与视觉模型统一使用浏览器 GUI：

```bash
# 4K 文本基座
python inference/app.py --model binichallein/OpenAster1-4k-base --port 7860 --open-browser

# 数学模型
python inference/app.py --model binichallein/OpenAster1-math --port 7860 --open-browser

# 128K 文本基座
python inference/app.py --model binichallein/OpenAster1-128k-base --port 7860 --open-browser

# 视觉模型
python inference/app.py --model binichallein/OpenAster1-VL --host 0.0.0.0 --port 7860 --open-browser
```

GUI 支持长对话、图片上传、thinking 开关和完整采样参数。运行 128K 模型时，可在界面中将 **Context tokens** 设为 `131072`。对话超过设定上下文预算时，会按完整问答轮次从最早位置裁剪；视觉模式会保留带图轮次，更换图片会开始新的视觉会话。

## License

Code in this repository is released under the MIT License. Dataset users should also respect the licenses and terms of the upstream data sources listed in the dataset card.
