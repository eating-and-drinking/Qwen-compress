# MOT-FD: Monotonic Optimal Transport Functional Distillation

本文档详细介绍了 qwen-compress 项目中采用的 **MOT-FD (Monotonic Optimal Transport Functional Distillation)** 蒸馏算法的完整实现原理与流程。

## 目录

- [核心思想](#核心思想)
- [完整流程](#完整流程)
- [阶段0: 教师模型功能分解](#阶段0-教师模型功能分解)
- [训练阶段: 复合损失函数](#训练阶段复合损失函数)
- [代码实现详解](#代码实现详解)
- [MOT-FD的优势](#MOT-FD的优势)

---

## 核心思想

MOT-FD 的核心创新在于将传统的**层对层映射**转变为**功能对齐**：

| 传统蒸馏 | MOT-FD |
|---------|--------|
| 固定层对层映射（如 S0→T0, S1→T4） | 动态功能对齐（软分配） |
| 关注「怎么做」（结构匹配） | 关注「做什么」（功能匹配） |
| 忽略层间关系 | 保持层顺序约束 |
| 教师知识碎片化 | 功能组聚合表示 |

这种方法特别适合跨架构蒸馏（如 MHA → MQA/GQA）或深度不匹配的场景。

---

## 完整流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MOT-FD 蒸馏流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  阶段0: 教师模型功能分解（预处理，一次性执行）                 │   │
│  │  1. 提取教师每层表示 z_l^T                                    │   │
│  │  2. 计算表示动力学能量 E(l)                                   │   │
│  │  3. 检测变化点，构建功能组 G_k                                │   │
│  │  4. 计算组表示 g_k^T                                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                           ↓                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  训练阶段（多次迭代）                                         │   │
│  │  1. 教师模型前向传播 → 获取教师 logits                        │   │
│  │  2. 学生模型前向传播 → 获取学生 logits + 所有层 hidden states │   │
│  │  3. 计算复合损失: L = CE + KD + OT + Monotonic                │   │
│  │  4. 反向传播，更新学生模型参数                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 阶段0: 教师模型功能分解

这是 MOT-FD 的关键预处理步骤，将教师模型的 48 层分解为 12 个功能组。

### 步骤1: 提取层表示

通过在少量校准数据上运行教师模型，获取每层隐藏状态的均值表示：

```python
# 代码位置: src/qwen_compress/distill/trainer.py # L280-L379
# 伪代码示意
def _run_teacher_decomposition():
    # 1. 注册钩子到所有教师层
    captures = [_LayerOutputCapture() for _ in teacher_layers]
    for cap, layer in zip(captures, teacher_layers):
        cap.attach(layer)
    
    # 2. 运行校准数据（如 256 个样本）
    layer_accum = [None] * num_teacher_layers
    for batch in calibration_data:
        with torch.no_grad():
            teacher(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        
        # 3. 计算每层的均值表示
        valid_mask = (batch.attention_mask == 1).unsqueeze(-1)
        for i, cap in enumerate(captures):
            h = cap.last_output  # [B, T, D]
            h_valid = h * valid_mask.to(h.dtype)
            rep = h_valid.sum(dim=(0, 1)) / valid_mask.sum().clamp_min(1.0)
            
            if layer_accum[i] is None:
                layer_accum[i] = rep
            else:
                layer_accum[i] += rep
    
    # 4. 平均后得到 z_l^T
    layer_reps = torch.stack([accum / num_samples for accum in layer_accum], dim=0)
    # layer_reps 形状: [48, hidden_dim]
    return layer_reps
```

**数学定义**：
```
z_l^T = E_{x~D}[h_l^T(x)]
```
其中 `h_l^T(x)` 是教师模型第 l 层在输入 x 上的隐藏状态，D 是校准数据分布。

---

### 步骤2: 计算表示动力学能量

通过三层能量信号检测层间变化：

```python
# 代码位置: src/qwen_compress/distill/groupwise.py # L70-L117
def compute_energy_signal(layer_reps, alpha=1.0, beta=0.5, gamma=0.3):
    z = layer_reps  # [L, D]
    
    # 1. 一阶差分: ||z_{l+1} - z_l||
    dz = z[1:] - z[:-1]
    first_order = dz.norm(dim=-1)
    
    # 2. 二阶差分: ||z_{l+1} - 2z_l + z_{l-1}||
    d2z = z[2:] - 2 * z[1:-1] + z[:-2]
    second_order = torch.cat([
        torch.zeros(1, device=z.device),
        d2z.norm(dim=-1),
    ])
    
    # 3. 余弦距离: 1 - cos(z_{l+1}, z_l)
    z_l = z[:-1]
    z_next = z[1:]
    cos_sim = (z_l * z_next).sum(dim=-1) / (
        z_l.norm(dim=-1) * z_next.norm(dim=-1) + 1e-8
    )
    cosine_dist = 1.0 - cos_sim
    
    # 4. 综合能量
    energy = alpha * first_order + beta * second_order + gamma * cosine_dist
    return energy  # [L-1]
```

**能量公式**：
```
E(l) = α·||z_{l+1} - z_l|| + β·||z_{l+1} - 2z_l + z_{l-1}|| + γ·(1 - cos(z_{l+1}, z_l))
```

- **一阶差分**：衡量相邻层表示的变化幅度
- **二阶差分**：衡量变化的加速度（拐点检测）
- **余弦距离**：衡量方向变化

---

### 步骤3: 检测变化点

在能量信号中找局部最大值作为断点：

```python
# 代码位置: src/qwen_compress/distill/groupwise.py # L120-L178
def detect_breakpoints(energy, num_breakpoints=11, min_distance=2):
    # 1. 找局部最大值
    candidates = []
    for i in range(1, len(energy)-1):
        if energy[i] > energy[i-1] and energy[i] > energy[i+1]:
            candidates.append((i, energy[i]))
    
    # 2. 按能量从大到小排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 3. 贪心选择，确保间距至少为 min_distance
    selected = []
    for idx, _ in candidates:
        bp = idx + 1  # 转换为层索引（1-indexed）
        if all(abs(bp - s) >= min_distance for s in selected):
            selected.append(bp)
        if len(selected) >= num_breakpoints:
            break
    
    # 4. 若不够，补充均匀分布的 fallback
    if len(selected) < num_breakpoints:
        step = (len(energy) + 1) // (num_breakpoints + 1)
        fallback = [step * (i + 1) for i in range(num_breakpoints)]
        for fb in fallback:
            if len(selected) >= num_breakpoints:
                break
            if fb not in selected and all(abs(fb - s) >= min_distance for s in selected):
                selected.append(fb)
    
    selected.sort()
    return selected
```

例如，可能检测到断点：
```
breakpoints = [5, 12, 18, 24, 30, 35, 39, 42, 44, 46, 47]
```
11 个断点 → 12 个功能组。

---

### 步骤4: 构建功能组并计算组表示

```python
# 代码位置: src/qwen_compress/distill/groupwise.py # L181-L230
def build_functional_groups(num_layers, breakpoints):
    boundaries = [0] + sorted(breakpoints) + [num_layers]
    groups = []
    for k in range(len(boundaries) - 1):
        start = boundaries[k]
        end = boundaries[k + 1]
        groups.append(list(range(start, end)))
    return groups

def compute_group_representations(groups, layer_reps):
    reps = []
    for g in groups:
        g_reps = layer_reps[g]  # [|G_k|, D]
        reps.append(g_reps.mean(dim=0))
    return torch.stack(reps, dim=0)  # [num_groups, D]
```

**组表示定义**：
```
g_k^T = mean{ z_l^T | l ∈ G_k }
```

示例分组结果：
```
G0: [0, 1, 2, 3, 4]
G1: [5, 6, 7, 8, 9, 10, 11]
G2: [12, 13, 14, 15, 16, 17]
...
G11: [47]
```

---

## 训练阶段: 复合损失函数

训练阶段使用四层损失函数：

```
L = α·CE + β·KD + λ_ot·L_OT + λ_mono·L_mono
```

### 1. CE 损失（交叉熵）

标准监督学习损失，确保学生能预测正确的 token：

```python
# 代码位置: src/qwen_compress/distill/losses.py # L394-L402
shift_logits = student_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
ce_loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

---

### 2. KD 损失（知识蒸馏）

匹配教师和学生的输出分布，使用 KL 散度：

```python
# 代码位置: src/qwen_compress/distill/losses.py # L113-L133
class KDLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits, valid_mask):
        T = self.temperature
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        t_probs = F.softmax(teacher_logits.detach() / T, dim=-1)
        
        per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)
        return masked_mean(per_token, valid_mask) * (T * T)
```

**公式**：
```
L_KD = T² × KL(softmax(S/T) || softmax(T/T))
```

温度 T 控制分布的平滑程度，T 越大分布越平滑。

---

### 3. OT 损失（最优传输对齐）

这是 MOT-FD 的核心，通过最优传输将学生层对齐到教师功能组：

```python
# 代码位置: src/qwen_compress/distill/losses.py # L180-L295
class OptimalTransportAlignLoss(nn.Module):
    def forward(self, student_hidden_states, teacher_group_reps, valid_mask):
        # 1. 计算学生每层的均值表示
        s_reps = []
        for h in student_hidden_states:
            if valid_mask is not None:
                mask = valid_mask.unsqueeze(-1).to(h.dtype)
                rep = (h * mask).sum(dim=(0, 1)) / mask.sum().clamp_min(1.0)
            else:
                rep = h.mean(dim=(0, 1))
            s_reps.append(rep)
        s_reps = torch.stack(s_reps, dim=0)  # [L, D]
        
        # 2. 对齐维度（若学生和教师隐藏维度不同）
        t_reps = teacher_group_reps.to(device=device, dtype=dtype)  # [G, D]
        if s_reps.shape[-1] != t_reps.shape[-1]:
            min_dim = min(s_reps.shape[-1], t_reps.shape[-1])
            s_reps = s_reps[..., :min_dim]
            t_reps = t_reps[..., :min_dim]
        
        # 3. 构建代价矩阵 C_{l,k} = ||h_l^S - g_k^T||² / D
        s_norm = s_reps.pow(2).sum(dim=-1, keepdim=True)  # [L, 1]
        t_norm = t_reps.pow(2).sum(dim=-1).unsqueeze(0)   # [1, G]
        s_t_dot = torch.mm(s_reps, t_reps.T)              # [L, G]
        C = (s_norm + t_norm - 2 * s_t_dot) / s_reps.shape[-1]
        
        # 4. Sinkhorn 算法求解最优传输计划
        gamma = sinkhorn(C, eps=self.ot_temperature, num_iters=self.sinkhorn_iters)
        
        # 5. OT 损失
        ot_loss = (gamma * C).sum()
        
        # 6. 软分配和期望位置（用于单调约束）
        pi = F.softmax(-C / self.soft_assign_temperature, dim=-1)  # [L, G]
        group_indices = torch.arange(num_groups, device=device)    # [G]
        expected_pos = (pi * group_indices.unsqueeze(0)).sum(dim=-1)  # [L]
        
        return ot_loss, expected_pos
```

#### Sinkhorn 算法

求解熵正则化最优传输问题：

```python
# 代码位置: src/qwen_compress/distill/losses.py # L49-L96
def sinkhorn(C, eps=0.1, num_iters=50, a=None, b=None):
    N, M = C.shape
    if a is None:
        a = torch.ones(N, device=C.device) / N
    if b is None:
        b = torch.ones(M, device=C.device) / M
    
    # Gibbs 核
    K = torch.exp(-C / eps)
    
    # Sinkhorn 迭代
    v = torch.ones(M, device=C.device) / M
    for _ in range(num_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    
    # 传输计划
    gamma = u.unsqueeze(1) * K * v.unsqueeze(0)
    return gamma
```

**问题定义**：
```
γ* = argmin_{γ∈Π(a,b)} Σ_{l,k} γ_{l,k} C_{l,k} + ε Σ γ_{l,k} log γ_{l,k}
```
其中 `Π(a,b)` 是满足边缘分布 a 和 b 的传输计划集合。

---

### 4. Monotonic 损失（单调性约束）

确保学生层按顺序学习教师功能：

```python
# 代码位置: src/qwen_compress/distill/losses.py # L291-L293
# 从 OT 损失计算中得到 expected_pos
mono_penalty = F.relu(expected_pos[:-1] - expected_pos[1:])
mono_loss = mono_penalty.sum()
```

**公式**：
```
L_mono = Σ_{l=1}^{L-1} max(0, μ_l - μ_{l+1})
```

其中 `μ_l = Σ_k k · π_{l,k}` 是学生层 l 的期望功能位置，`π_{l,k} = softmax(-C_{l,k}/τ)` 是软分配概率。

---

## 代码实现详解

### 主训练循环

```python
# 代码位置: src/qwen_compress/distill/trainer.py # L510-L643
def train(self):
    # 初始化优化器和学习率调度器
    optimizer = AdamW(param_groups, lr=opt.lr)
    scheduler = _make_lr_scheduler(optimizer, ...)
    
    global_step = 0
    while global_step < self.total_steps:
        for batch in self.train_loader:
            # 1. 前向传播
            with self._autocast():
                s_logits, t_logits, s_hidden = self._forward_pair(batch)
                labels = batch["labels"].to(s_logits.device)
                attention_mask = batch["attention_mask"].to(s_logits.device)
                
                # 2. 计算复合损失
                loss_out = self.loss_fn(
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    labels=labels,
                    attention_mask=attention_mask,
                    student_hidden_states=s_hidden,
                    teacher_hidden_states=None,  # MOT-FD 使用组表示
                )
                loss = loss_out.total / cfg.data.gradient_accumulation_steps
            
            # 3. 反向传播
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 4. 梯度累积和更新
            accumulated += 1
            if accumulated >= cfg.data.gradient_accumulation_steps:
                if self._scaler is not None:
                    self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in optimizer.param_groups for p in pg["params"] if p.grad is not None],
                    cfg.training.max_grad_norm,
                )
                if self._scaler is not None:
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                global_step += 1
                
                # 日志和评估
                if global_step % cfg.training.logging_steps == 0:
                    _logger.info(f"step={global_step}, loss={loss_out.total:.4f}")
                if global_step % cfg.training.eval_steps == 0:
                    eval_metrics = self.evaluate()
    
    # 保存最终模型
    final_path = self._save(output_dir, "final")
    return final_path
```

---

### 钩子机制：捕获隐藏状态

```python
# 代码位置: src/qwen_compress/distill/trainer.py # L56-L77
class _LayerOutputCapture:
    def __init__(self):
        self.last_output = None
        self._handle = None
    
    def attach(self, layer):
        def _hook(mod, inp, out):
            self.last_output = out[0] if isinstance(out, tuple) else out
        self._handle = layer.register_forward_hook(_hook)
    
    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.last_output = None
```

---

## MOT-FD 的优势

| 传统蒸馏 | MOT-FD |
|---------|--------|
| 固定层对层映射 | 动态功能对齐 |
| 忽略层间关系 | 保持顺序约束 |
| 教师知识碎片化 | 功能组聚合表示 |
| 学生被迫模仿教师结构 | 学生自主学习功能分布 |
| 跨架构蒸馏困难 | 特别适合跨架构/深度不匹配场景 |

### 适用场景

1. **跨架构蒸馏**：从 MHA 到 MQA/GQA
2. **深度不匹配**：48层教师到 24层学生
3. **模型压缩**：大模型知识迁移到小模型
4. **功能保留**：特别关注某些功能的保留

---

## 配置参考

```yaml
# configs/distill/qwen2_5_14b_to_3b.yaml
teacher_model_name_or_path: "Qwen/Qwen2.5-14B-Instruct"
student_model_name_or_path: "Qwen/Qwen2.5-3B"
num_groups: 12
calibration_samples: 256
alpha_ce: 1.0
beta_kd: 1.0
lambda_ot: 1.0
lambda_mono: 0.1
kd_temperature: 2.0
ot_temperature: 0.1
sinkhorn_iters: 50
```

---

## 相关代码文件

| 文件 | 功能 |
|------|------|
| [src/qwen_compress/distill/trainer.py](file:///c:/python/qwen-compress/src/qwen_compress/distill/trainer.py) | 主训练器，教师分解，训练循环 |
| [src/qwen_compress/distill/losses.py](file:///c:/python/qwen-compress/src/qwen_compress/distill/losses.py) | 复合损失函数实现 |
| [src/qwen_compress/distill/groupwise.py](file:///c:/python/qwen-compress/src/qwen_compress/distill/groupwise.py) | 教师功能分解实现 |
