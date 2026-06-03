# CoT Generator

批量生成高质量 Chain-of-Thought (CoT) 训练数据，用于大模型监督微调。

---

## 功能特性

- **批量生成**：基于 vLLM 的高效批量推理，支持多卡并行
- **Self-Instruct 扩充**：自动扩充问题池，达到目标数据量
- **质量过滤**：自动去重、步骤完整性检查
- **交叉验证**：可选双模型一致性校验（14B生成 + 7B验证）
- **双模式输出**：同时生成 CoT 模式和 Direct 模式训练数据
- **标准化格式**：输出标准 SFT 训练格式，无缝对接主流训练框架

---

## 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如需使用 HuggingFace datasets（可选）
pip install datasets
```

### 2. 运行生成

```bash
# 使用默认配置生成 12 万条数据
python run.py generate

# 生成指定数量
python run.py generate --target 50000

# 指定输出目录
python run.py generate --output_dir /data/my_cot_dataset

# 禁用交叉验证（更快，适合单卡）
python run.py generate --no_cross_val
```

### 3. 混合 CoT/Direct 训练集

```bash
python run.py merge --output_dir outputs/cot_dataset --cot_ratio 0.7
```

### 4. 查看统计

```bash
python run.py stats --output_dir outputs/cot_dataset
```

---

## 配置说明

配置文件：`configs/pipeline.json`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `target_total` | 120000 | 目标生成条数 |
| `self_instruct_count` | 35000 | Self-Instruct 扩充的问题数 |
| `cot_mix_ratio` | 0.7 | 最终混合集中 CoT 条目占比 |
| `enable_cross_validation` | false | 是否启用双模型交叉验证 |
| `flush_every` | 500 | 每 N 条刷盘一次 |

### 生成器配置

```json
{
  "generator": {
    "model_name": "Qwen/Qwen2.5-14B-Instruct",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.90,
    "max_model_len": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "batch_size": 64,
    "min_thinking_length": 100,
    "required_steps": ["步骤1", "步骤2", "步骤3"]
  }
}
```

### 数据源配置

支持 HuggingFace 数据集和本地 JSONL 文件：

```json
{
  "data_sources": {
    "hf_sources": {
      "math": 7500,
      "logiqa": 5000,
      "ceval": 15000,
      "mbpp": 8000
    },
    "local_sources": [
      {
        "path": "data/seeds/custom_math.jsonl",
        "domain": "math"
      }
    ]
  }
}
```

---

## 支持的数据源

| 数据源 | 类型 | 领域 | 说明 |
|-------|------|------|------|
| math | HuggingFace | 数学 | 竞赛数学题 |
| logiqa | HuggingFace | 逻辑 | 逻辑推理题 |
| ceval | HuggingFace | 综合 | 中文考试数据集 |
| mbpp | HuggingFace | 代码 | Python 编程题 |
| local | JSONL | 自定义 | 本地种子文件 |

---

## 输出文件

```
outputs/cot_dataset/
├── train_cot.jsonl      # CoT 模式训练集（含 <thinking>...</thinking>）
├── train_direct.jsonl   # Direct 模式训练集（只含答案）
├── train_mixed.jsonl    # 混合训练集（默认 7:3 比例）
├── val.jsonl            # 验证集（从训练集拆分 5%）
├── test.jsonl           # 测试集（从训练集拆分 5%）
└── metadata.json        # 统计信息和元数据
```

### 数据格式

每条数据格式（`train_mixed.jsonl`）：

```json
{
  "id": "abc12345",
  "domain": "math",
  "question": "求解方程：2x + 5 = 15",
  "thinking": "步骤1：理解问题...步骤2：移项...",
  "answer": "x = 5",
  "source": "math",
  "sft_cot": {
    "messages": [
      {"role": "system", "content": "你是一个逻辑推理专家..."},
      {"role": "user", "content": "求解方程：2x + 5 = 15"},
      {"role": "assistant", "content": "<thinking>...步骤...</thinking>\n\n<answer>x = 5</answer>"}
    ]
  },
  "sft_direct": {
    "messages": [
      {"role": "user", "content": "求解方程：2x + 5 = 15"},
      {"role": "assistant", "content": "x = 5"}
    ]
  }
}
```

---

## 添加自定义种子问题

在 `data/seeds/` 目录下放 JSONL 文件，每行格式：

```json
{"question": "你的问题", "domain": "math"}
```

支持的 domain：`math` / `logic` / `code` / `common_sense` / `science` / `language` / `general`

然后在 `configs/pipeline.json` 的 `data_sources.local_sources` 中添加路径即可。

---

## 硬件参考

| 配置 | 吞吐量估算 | 12万条耗时 |
|------|-----------|-----------|
| 1x A100 80G | ~800 tok/s | ~25 小时 |
| 2x A100 80G | ~1500 tok/s | ~13 小时 |
| 4x A100 80G | ~2800 tok/s | ~7 小时 |

> **建议**：原始生成量设为目标的 1.3 倍（过滤率约 20-30%）。

---

## 目录结构

```
cot_generator/
├── configs/
│   └── pipeline.json          # 默认配置文件
├── data/seeds/
│   └── example_seeds.jsonl    # 种子问题示例
├── __init__.py                # 模块导出
├── data_loader.py             # 多源数据加载器
├── expander.py                # Self-Instruct 扩充器
├── formatter.py               # SFT 格式输出
├── generator.py               # 核心生成器（vLLM）
├── pipeline.py                # 端到端主控流程
├── prompts.py                 # Prompt 模板库
├── validator.py               # 交叉验证模块
├── run.py                     # CLI 入口
├── requirements.txt           # 依赖列表
└── README.md                  # 本文件
```

---

## 作为模块导入

```python
from qwen_compress.cot_generator import run_pipeline

# 配置
cfg = {
    "output_dir": "outputs/my_dataset",
    "target_total": 50000,
    "cot_mix_ratio": 0.7,
    "data_sources": {
        "hf_sources": {"math": 5000},
        "local_sources": [{"path": "data/seeds/custom.jsonl"}]
    },
    "generator": {
        "tensor_parallel_size": 1,
        "batch_size": 32
    }
}

# 运行
run_pipeline(cfg)
```

---

## 许可证

Apache License 2.0
