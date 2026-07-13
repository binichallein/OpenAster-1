# OpenAster-1

OpenAster-1 是一个完全开源的 2B 级混合专家（MoE）语言模型项目。发布内容包括模型权重、20B token 从零预训练数据配方、训练与评测脚本，以及完整技术报告。

[English](./README.md) | **简体中文**

[代码](https://github.com/binichallein/OpenAster-1) | [Hugging Face 合集](https://huggingface.co/collections/binichallein/openaster-1) | [ModelScope 合集](https://www.modelscope.cn/collections/TYFTYF/OpenAster-1) | [技术报告](./OpenAster-1_Technical_Report.pdf)

## 项目亮点

- **完整开放**：模型权重、数据、训练配方、训练脚本和评测脚本一并发布。
- **低成本训练目标**：OpenAster-1 从零开始训练，总训练量约为 100B token。
- **MoE 文本基座**：总参数量 2B，每个 token 激活约 1.3B 参数。
- **长上下文**：发布的 128K 基座模型采用 YaRN 风格的渐进式扩窗训练。
- **视觉微调分支**：OpenAster1-VL 使用 LLaVA 风格的 CLIP、Projector 与语言模型组合方案。

## 模型列表

| 模型 | 类型 | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| OpenAster1-4k-base | 4K 文本基座 | [binichallein/OpenAster1-4k-base](https://huggingface.co/binichallein/OpenAster1-4k-base) | [TYFTYF/OpenAster1-4k-base](https://www.modelscope.cn/models/TYFTYF/OpenAster1-4k-base) |
| OpenAster1-128k-base | 128K 文本基座 | [binichallein/OpenAster1-128k-base](https://huggingface.co/binichallein/OpenAster1-128k-base) | [TYFTYF/OpenAster1-128k-base](https://www.modelscope.cn/models/TYFTYF/OpenAster1-128k-base) |
| OpenAster1-math | 数学 SFT | [binichallein/OpenAster1-math](https://huggingface.co/binichallein/OpenAster1-math) | [TYFTYF/OpenAster1-math](https://www.modelscope.cn/models/TYFTYF/OpenAster1-math) |
| OpenAster1-VL | 视觉指令微调 | [binichallein/OpenAster1-VL](https://huggingface.co/binichallein/OpenAster1-VL) | [TYFTYF/OpenAster1-VL](https://www.modelscope.cn/models/TYFTYF/OpenAster1-VL) |

## 数据

20B token 从零预训练数据以 `OpenAster-1-data` 名称发布：

- Hugging Face：[binichallein/OpenAster-1-data](https://huggingface.co/datasets/binichallein/OpenAster-1-data)
- ModelScope：[TYFTYF/OpenAster-1-data](https://www.modelscope.cn/datasets/TYFTYF/OpenAster-1-data)

20B 数据配方包括 DCLM Edu、OpenCSG FineWeb Edu Chinese 4/5、FineMath 4+、FineWeb Edu EN、Chinese FineWeb Edu 和 Pile Deduplicated 回放数据。数据以 JSONL 格式组织，每条记录包含 `text`、`source` 和 `language` 字段。

## 推理

安装运行依赖：

```bash
pip install -r requirements.txt
```

OpenAster 只提供一个推理入口：`inference/app.py`。该脚本启动浏览器 GUI，并自动识别文本模型（`Qwen3MoeForCausalLM`）和视觉模型（`LlavaForConditionalGeneration`）。

### 浏览器 GUI

#### 视觉对话

<p align="center">
  <img src="assets/gui-vision-demo.gif" alt="OpenAster1-VL 视觉对话演示" width="1000">
</p>

该演示使用公开的 `binichallein/OpenAster1-VL` 权重，上传真实图片并进行视觉追问。

#### 文本思考

<p align="center">
  <img src="assets/gui-text-thinking-demo.gif" alt="OpenAster1-4k-base 文本思考演示" width="1000">
</p>

该演示使用公开的 `binichallein/OpenAster1-4k-base` 权重，开启 Thinking 并设置 `seed=0`。用户问题为“我想学习游泳，你能给我一些建议吗”。思考内容会实时输出，完成后折叠为带耗时的下拉区域；用户可以重新展开查看，最终回答则独立显示。

#### 数学推理

<p align="center">
  <img src="assets/gui-math-demo.gif" alt="OpenAster1-math 数学推理与公式渲染演示" width="1000">
</p>

该演示使用公开的 `binichallein/OpenAster1-math` 权重，在 GSM8K 测试集第 208 个样例上设置 `temperature=0.7`、`top_p=0.95` 和 `seed=3`。模型推导出正确答案 `76`，GUI 在流式生成结束后渲染 LaTeX 公式和 `\boxed{76}`。

启动 4K 文本基座：

```bash
python inference/app.py --model binichallein/OpenAster1-4k-base --port 7860 --open-browser
```

启动 OpenAster1-math：

```bash
python inference/app.py --model binichallein/OpenAster1-math --port 7860 --open-browser
```

启动 128K 文本基座：

```bash
python inference/app.py --model binichallein/OpenAster1-128k-base --port 7860 --open-browser
```

需要完整 128K 上下文预算时，请在 GUI 中将 **Context tokens** 设置为 `131072`。

启动 OpenAster1-VL：

```bash
python inference/app.py --model binichallein/OpenAster1-VL --port 7860 --open-browser
```

浏览器访问 `http://localhost:7860`。GUI 支持流式生成、长对话、图片上传、Thinking 开关和完整采样参数。对话接近上下文上限时，会优先移除最早的完整问答轮次，并保留当前轮次和视觉锚点轮次；选择新图片会开始新的视觉会话。

内置服务不提供身份验证。仅在本地使用时请保持默认回环地址；如需开放到网络，应先配置带身份验证的反向代理。

## 训练

公开训练脚本位于 `training/`：

- `training/pretrain/run_pretrain_megatron.sh`：从零预训练、继续预训练与 YaRN 扩窗入口。
- `training/post_training/run_math_sft.sh`：OpenAster1-math 数学 SFT 入口。
- `training/post_training/run_text_rl_diagnostics.sh`：纯文本 RL 诊断分支。
- `training/vision/run_llava_style_vision_tuning.sh`：LLaVA 风格视觉指令微调入口。

所有脚本都通过环境变量接收数据和模型路径，不包含私有服务器路径。

## 评测

评测脚本位于 `eval/`，覆盖技术报告使用的主要基准：

- 文本基础能力：MMLU、C-Eval、C-MMLU、OpenBookQA、ARC、HellaSwag、BoolQ 和 BBH。
- 数学能力：GSM8K 和 MATH500。
- 视觉能力：MME、POPE、ScienceQA-IMG、TextVQA 和 MMBench-EN dev。

任务名称和推荐命令见 `eval/configs/openaster_eval_tasks.yaml`。

## 许可证

本仓库代码使用 MIT License 发布。使用数据集时，还应遵守数据卡中列出的上游数据源许可证与使用条款。
