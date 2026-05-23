# qwen-compress

面向 Qwen 模型家族的工业级压缩工具库。三阶段可组合流水线 —— **逐组蒸馏 → SparseGPT 剪枝 → INT8 QAT/QAD** —— 把 14B 量级的教师模型压到 3B INT8，适合端侧部署，同时保留思维链推理能力。

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org)

[English](./README.md) | 简体中文

---

## 特性

- **逐组蒸馏（Group-wise Distillation）**。把教师 decoder 栈切成 `G` 个连续组，每组选一层作为锚点，学生模型在对应深度比例的层上学习匹配，并通过可学习的线性投影解决维度不一致。在教师远深于学生时，效果显著好于只匹配末层 logits。
- **SparseGPT 剪枝**。逐 block 一次性剪枝，使用阻尼 Hessian 估计；支持非结构化、2:4、4:8 三种稀疏模式；峰值显存只占一个 decoder block。
- **QAT + QAD 量化**。即插即用的 `FakeQuantize` 模块、权重 per-channel + 激活 per-tensor/per-token 量化、可选 LSQ 可学习 scale、KV-Cache 量化（端到端 INT8 推理的关键）、可选用同一个教师模型做 Quantization-Aware Distillation 缓解精度损失。
- **保持稀疏的恢复微调**：剪枝完每次 optimizer step 后重新施加 mask，避免后续训练把零位置训出非零值。
- **CoT 友好的 SFT 加载器**：`direct` / `cot` / `dual` 三种模式，让学生压缩后还能保留推理能力。
- **每阶段一份 YAML 配置**，pydantic 强校验；顶层 `pipeline` 配置可一次串通三个阶段。
- **工程化基础设施**：loguru 结构化日志、safetensors 原子化 checkpoint、确定性种子、分布式训练辅助函数、ONNX QDQ 导出。

## 架构

```
┌────────────────────┐    阶段1：蒸馏           ┌────────────────────┐
│ Qwen2.5-14B-Inst.  │  ─────────────────────▶ │   Qwen2.5-3B       │
│  （教师，冻结）       │   逐组 CE+KD+MSE          │  （学生，FP16）       │
└────────────────────┘                          └─────────┬──────────┘
                                                          │
                                            阶段2：SparseGPT 剪枝
                                                          │
                                          ┌───────────────▼─────────────┐
                                          │   3B @ 50% 非结构化             │
                                          │    + 保持稀疏的恢复 FT          │
                                          └───────────────┬─────────────┘
                                                          │
                                    阶段3：INT8 QAT + QAD
                                    （继续用 14B 教师做监督）
                                                          │
                                          ┌───────────────▼─────────────┐
                                          │ 3B INT8（W8A8 + KV-Cache）   │
                                          │   safetensors / ONNX        │
                                          └─────────────────────────────┘
```

## 安装

```bash
git clone https://github.com/eating-and-drinking/Qwen-compress.git
cd Qwen-compress
pip install -e ".[dev]"
```

要求 Python ≥ 3.9，PyTorch ≥ 2.1，CUDA ≥ 11.8。

## 快速开始

```bash
# 阶段1：蒸馏 Qwen2.5-14B-Instruct -> Qwen2.5-3B
bash scripts/run_distill.sh configs/distill/qwen2_5_14b_to_3b.yaml

# 阶段2：SparseGPT 50% 剪枝 + 恢复微调
bash scripts/run_prune.sh   configs/prune/sparsegpt_50pct.yaml

# 阶段3：INT8 QAT + QAD
bash scripts/run_qat.sh     configs/qat/int8_qad.yaml

# 或者一条命令跑完全流程：
bash scripts/run_pipeline.sh configs/pipeline/full.yaml
```

## 数据格式

每个阶段都吃同一份思维链 SFT 数据（JSONL）：

```json
{
  "instruction": "...",
  "input": "",
  "chain_of_thought": "...",
  "answer": "..."
}
```

`dual` 模式（默认）逐样本在"直接答案"和"`<think>` + 答案"两种格式之间交替，让学生同时具备快速回答和推理能力。

## 多卡训练

```bash
NUM_GPUS=8 bash scripts/run_distill.sh configs/distill/qwen2_5_14b_to_3b.yaml
```

教师用 `device_map="auto"` 自动分片，学生在每张卡的 `cuda:0`；KD 与 hidden-state 损失会自动把教师张量搬到学生设备。

## 项目结构

详见英文 README。

## 引用

```bibtex
@software{qwen_compress_2024,
  title  = {qwen-compress: A Production-Grade Compression Toolkit for the Qwen Family},
  author = {eating-and-drinking},
  year   = {2024},
  url    = {https://github.com/eating-and-drinking/Qwen-compress},
}
```

## 作者

由 [eating-and-drinking](https://github.com/eating-and-drinking) 创建并维护。

仓库地址：[https://github.com/eating-and-drinking/Qwen-compress](https://github.com/eating-and-drinking/Qwen-compress)

## 许可证

Apache License 2.0。Qwen 是阿里巴巴集团的商标，本项目与阿里巴巴无关。
