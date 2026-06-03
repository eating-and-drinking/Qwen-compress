# qwen-compress

生产级Qwen大模型压缩工具包，整合三种核心压缩技术：**MOT-FD Group-wise Distillation** + **SparseGPT Pruning** + **QAT/QAD Quantization**。

## 🌟 核心特性

| 阶段 | 技术 | 功能描述 |
|------|------|----------|
| **Stage 1** | MOT-FD | 基于最优传输的组级功能蒸馏，将14B教师模型知识迁移到3B学生模型 |
| **Stage 2** | SparseGPT | 非结构化剪枝，支持50%+稀疏度，可选通道重排和恢复微调 |
| **Stage 3** | QAT + QAD | 量化感知训练结合量化感知蒸馏，输出INT8量化模型 |

## 🚀 快速开始

### 安装

```powershell
# 基础安装
pip install -e .

# 完整安装（包含导出和跟踪功能）
pip install -e ".[all]"
```

### 运行完整压缩流程

```powershell
# 使用预配置的完整流程
qwen-compress pipeline --config configs/pipeline/full.yaml
```

### 分步执行

```powershell
# 阶段1: 蒸馏 (14B → 3B)
qwen-compress distill --config configs/distill/qwen2_5_14b_to_3b.yaml

# 阶段2: 剪枝 (50%稀疏度)
qwen-compress prune --config configs/prune/sparsegpt_50pct.yaml

# 阶段3: 量化 (INT8 QAT)
qwen-compress qat --config configs/qat/int8_qad.yaml
```

### Python API 使用

```python
from qwen_compress.distill import GroupwiseDistillTrainer
from qwen_compress.qat import QADTrainer, export_quantized_model
from qwen_compress.utils.config import DistillConfig, QATConfig

# 蒸馏配置
distill_cfg = DistillConfig(
    teacher_model_name_or_path="Qwen/Qwen2.5-14B-Instruct",
    student_model_name_or_path="Qwen/Qwen2.5-3B",
    num_groups=12,
    # ... 其他配置
)

# 执行蒸馏
trainer = GroupwiseDistillTrainer(distill_cfg)
distilled_model_path = trainer.train()
```

## 📁 项目结构

```
qwen-compress/
├── configs/                    # 配置文件目录
│   ├── distill/               # 蒸馏配置
│   ├── pipeline/              # 完整流程配置
│   ├── prune/                 # 剪枝配置
│   └── qat/                   # 量化配置
├── examples/                  # 示例代码
│   └── quick_start.py         # 快速入门示例
├── scripts/                   # 辅助脚本
│   ├── run_distill.sh         # 蒸馏脚本
│   ├── run_prune.sh           # 剪枝脚本
│   ├── run_qat.sh             # 量化脚本
│   ├── run_pipeline.sh        # 完整流程脚本
│   └── split_dataset.py       # 数据集切分工具
├── src/qwen_compress/         # 核心源代码
│   ├── cot_generator/         # CoT训练数据生成器
│   ├── data/                  # 数据处理模块
│   ├── distill/               # 蒸馏模块
│   ├── models/                # 模型封装
│   ├── prune/                 # 剪枝模块
│   ├── qat/                   # 量化模块
│   ├── utils/                 # 工具函数
│   ├── cli.py                 # 命令行接口
│   └── __init__.py            # 模块导出
└── tests/                     # 测试用例
```

## 📊 压缩流程架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    qwen-compress Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Stage 1    │ -> │  Stage 2    │ -> │  Stage 3    │         │
│  │  MOT-FD     │    │  SparseGPT  │    │  QAT + QAD  │         │
│  │ Distillation│    │   Pruning   │    │ Quantization│         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│       │                   │                   │                 │
│       ▼                   ▼                   ▼                 │
│  14B → 3B            3B → 50%稀疏         INT8量化模型          │
│       │                   │                   │                 │
│       ▼                   ▼                   ▼                 │
│  教师功能分解         可选恢复微调          ONNX/Safetensors导出  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🎛️ 配置说明

### 蒸馏配置 (`configs/distill/*.yaml`)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `teacher_model_name_or_path` | - | 教师模型路径或HuggingFace名称 |
| `student_model_name_or_path` | - | 学生模型路径或HuggingFace名称 |
| `num_groups` | 12 | 功能分组数量 |
| `calibration_samples` | 256 | 教师分解校准样本数 |
| `alpha_ce` | 1.0 | 交叉熵损失权重 |
| `beta_kd` | 1.0 | 知识蒸馏损失权重 |
| `lambda_ot` | 1.0 | 最优传输损失权重 |
| `lambda_mono` | 0.1 | 单调性约束损失权重 |

### 剪枝配置 (`configs/prune/*.yaml`)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `model_name_or_path` | - | 待剪枝模型路径 |
| `sparsity` | 0.5 | 目标稀疏度 (0.0-1.0) |
| `sparsity_type` | unstructured | 剪枝类型 |
| `block_size` | 128 | SparseGPT块大小 |
| `nsamples` | 128 | 校准样本数 |
| `recovery_finetune` | true | 是否启用恢复微调 |

### 量化配置 (`configs/qat/*.yaml`)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `weight_bits` | 8 | 权重量化位数 |
| `activation_bits` | 8 | 激活量化位数 |
| `use_qad` | true | 是否启用量化感知蒸馏 |
| `nsamples_calib` | 512 | 量化校准样本数 |
| `export_format` | safetensors | 导出格式 (safetensors/onnx) |

## 🔧 CLI 命令参考

### distill 命令

```powershell
qwen-compress distill --config configs/distill/qwen2_5_14b_to_3b.yaml
```

### prune 命令

```powershell
qwen-compress prune --config configs/prune/sparsegpt_50pct.yaml
```

### qat 命令

```powershell
qwen-compress qat --config configs/qat/int8_qad.yaml
qwen-compress qat --config configs/qat/int8_qad.yaml --export-only
```

### pipeline 命令

```powershell
qwen-compress pipeline --config configs/pipeline/full.yaml
```

## 📚 数据准备

### CoT SFT 数据集格式

项目需要CoT风格的SFT训练数据，格式如下：

```json
{
  "id": "example_001",
  "sft_cot": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "<thinking>Let me think step by step...\nStep 1: Add 2 and 2.\nStep 2: The result is 4.</thinking>\n\n<answer>4</answer>"}
    ]
  },
  "sft_direct": {
    "messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"}
    ]
  }
}
```

### CoT 数据生成器

项目内置CoT训练数据生成器，可批量生成高质量训练数据：

```powershell
# 进入CoT生成器目录
cd src/qwen_compress/cot_generator

# 生成120K CoT训练样本
python run.py generate --target 120000

# 合并CoT和Direct模式数据
python run.py merge --output_dir outputs/cot_dataset --cot_ratio 0.7
```

## 🛠️ 依赖要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.9+ | 编程语言 |
| PyTorch | 2.1.0+ | 深度学习框架 |
| Transformers | 4.40.0+ | 模型库 |
| Accelerate | 0.27.0+ | 分布式训练 |
| Datasets | 2.14.0+ | 数据集处理 |
| Safetensors | 0.4.0+ | 模型存储 |
| PyYAML | 6.0+ | 配置文件解析 |
| Click | 8.1.0+ | CLI框架 |

## 🧪 测试

```powershell
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_losses.py -v

# 排除慢测试
python -m pytest tests/ -v -m "not slow"

# GPU测试
python -m pytest tests/ -v -m "gpu"
```

## 📈 性能参考

### 压缩效果

| 模型 | 原始大小 | 压缩后大小 | 压缩比 |
|------|----------|------------|--------|
| Qwen2.5-14B | ~28 GB (FP16) | ~4.5 GB (INT8) | ~6.2x |
| Qwen2.5-3B | ~6 GB (FP16) | ~1.5 GB (INT8) | ~4x |

### 推理性能

| 模型 | 推理速度 (tokens/s) | 内存占用 |
|------|---------------------|----------|
| Qwen2.5-14B FP16 | ~30 | ~32 GB |
| Qwen2.5-3B INT8 | ~150 | ~4 GB |

## 📝 示例代码

完整的Python API示例请参考 [examples/quick_start.py](file:///c:/python/qwen-compress/examples/quick_start.py)：

```powershell
python examples/quick_start.py --train-data ./data/cot_sft_120k.jsonl
```

## 📄 许可证

Apache License 2.0 — see [LICENSE](LICENSE). Includes an express patent grant.

## 👤 作者

Created and maintained by [eating-and-drinking](https://github.com/eating-and-drinking).

Repository: [https://github.com/eating-and-drinking/Qwen-compress](https://github.com/eating-and-drinking/Qwen-compress)

