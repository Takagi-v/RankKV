# RankKV: 基于Attention矩阵有效秩的KV Cache压缩方法

## 项目方案计划书

---

## 1. 项目概述

### 1.1 背景

大语言模型（LLM）在推理时需要存储KV Cache来避免重复计算，但KV Cache的大小随序列长度线性增长，成为推理效率的主要瓶颈。现有的KV Cache压缩方法通常对每层采用相同的压缩策略，忽略了不同层之间的差异。

### 1.2 核心观察

我们观察到：**不同层的Attention矩阵具有不同的"有效秩"（Effective Rank）**，这反映了该层attention分布的复杂程度：
- **有效秩低** → Attention分布集中在少数token上 → 可以激进压缩
- **有效秩高** → Attention分布分散在多个token上 → 需要保守压缩

### 1.3 方法概述

**RankKV** 是一种基于Attention矩阵有效秩的层级自适应KV Cache压缩方法：
1. 在Prefill阶段计算每层Attention矩阵的有效秩
2. 根据有效秩为每层分配不同的KV Cache预算（budget）
3. 使用现有的token selection方法（如SnapKV）选择具体保留哪些token

### 1.4 创新点

1. **首次提出使用Attention矩阵的Effective Rank指导层级budget分配**
2. **无训练方法**：只需在Prefill阶段进行一次分析，不需要额外训练
3. **与现有方法正交**：可以与任何token selection方法结合

---

## 2. 技术方案

### 2.1 整体流程

```
输入文本 → Prefill阶段 → 分析Rank → 分配Budget → 压缩KV Cache → Decode阶段 → 输出
              ↓              ↓            ↓              ↓
         生成Attention矩阵  计算每层rank  根据rank分配    选择token并压缩
         生成KV Cache      (SVD分解)     每层budget      (用SnapKV等)
```

### 2.2 Effective Rank 计算

**定义**：给定一个矩阵 $A$，其有效秩定义为：

$$\text{erank}(A) = \exp\left(-\sum_{i} \tilde{\sigma}_i \log \tilde{\sigma}_i\right)$$

其中 $\tilde{\sigma}_i = \sigma_i / \sum_j \sigma_j$ 是归一化的奇异值。

**直观理解**：
- 有效秩衡量矩阵"信息维度"的多少
- 如果奇异值集中在少数几个，有效秩低
- 如果奇异值分散在多个，有效秩高

**代码实现**：

```python
import torch

def compute_effective_rank(matrix):
    """
    计算矩阵的有效秩
    
    Args:
        matrix: [seq_len, seq_len] 的Attention矩阵
    
    Returns:
        effective_rank: float, 有效秩值
    """
    # SVD分解，获取奇异值
    U, S, V = torch.svd(matrix)
    
    # 归一化奇异值
    S_normalized = S / (S.sum() + 1e-10)
    
    # 计算熵
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    
    # 有效秩 = exp(熵)
    effective_rank = torch.exp(entropy)
    
    return effective_rank.item()
```

### 2.3 Budget 分配策略

**策略**：有效秩高的层分配更多budget，有效秩低的层分配更少budget。

```python
def allocate_budget(ranks, total_budget, min_budget=10):
    """
    根据每层的rank分配budget
    
    Args:
        ranks: list, 每层的有效秩
        total_budget: int, 总共要保留的token数
        min_budget: int, 每层最少保留的token数
    
    Returns:
        budgets: list, 每层分配的budget
    """
    ranks_tensor = torch.tensor(ranks)
    
    # 按rank比例分配
    ratios = ranks_tensor / ranks_tensor.sum()
    budgets = (ratios * total_budget).int().tolist()
    
    # 确保每层至少有min_budget
    budgets = [max(b, min_budget) for b in budgets]
    
    return budgets
```

### 2.4 Token Selection

使用基于Attention Score的方法选择每层保留哪些token：

```python
def select_tokens(K, V, attention_weights, budget):
    """
    基于attention score选择要保留的token
    
    Args:
        K: [seq_len, head_dim], Key矩阵
        V: [seq_len, head_dim], Value矩阵
        attention_weights: [seq_len, seq_len], Attention矩阵
        budget: int, 要保留的token数
    
    Returns:
        K_compressed, V_compressed: 压缩后的K和V
        selected_indices: 被选中的token索引
    """
    # 用最后几个token的attention作为观察窗口
    window_size = min(32, attention_weights.shape[0])
    observation = attention_weights[-window_size:, :]  # [window, seq_len]
    
    # 计算每个token的重要性分数
    importance = observation.sum(dim=0)  # [seq_len]
    
    # 选择分数最高的budget个token
    _, top_indices = torch.topk(importance, budget)
    top_indices = top_indices.sort().values  # 保持顺序
    
    # 压缩
    K_compressed = K[top_indices]
    V_compressed = V[top_indices]
    
    return K_compressed, V_compressed, top_indices
```

### 2.5 完整Pipeline

```python
class RankKV:
    def __init__(self, model, budget_ratio=0.3):
        """
        Args:
            model: HuggingFace模型
            budget_ratio: 压缩比例，0.3表示保留30%的token
        """
        self.model = model
        self.budget_ratio = budget_ratio
    
    def compress(self, input_ids):
        """
        执行RankKV压缩
        
        Args:
            input_ids: [batch, seq_len], 输入token ids
        
        Returns:
            compressed_kv_cache: 压缩后的KV Cache
        """
        seq_len = input_ids.shape[1]
        total_budget = int(seq_len * self.budget_ratio)
        
        # Step 1: Prefill，获取attention矩阵和KV cache
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_attentions=True,
                use_cache=True
            )
        
        attentions = outputs.attentions      # tuple of [batch, heads, seq, seq]
        kv_cache = outputs.past_key_values   # tuple of (K, V)
        num_layers = len(attentions)
        
        # Step 2: 计算每层的有效秩
        ranks = []
        for layer_idx in range(num_layers):
            # 对所有head取平均
            attn = attentions[layer_idx].mean(dim=1).squeeze(0)  # [seq, seq]
            rank = compute_effective_rank(attn)
            ranks.append(rank)
        
        # Step 3: 根据rank分配budget
        budgets = allocate_budget(ranks, total_budget)
        
        # Step 4: 压缩每层的KV Cache
        compressed_kv_cache = []
        for layer_idx in range(num_layers):
            K, V = kv_cache[layer_idx]
            attn = attentions[layer_idx].mean(dim=1).squeeze(0)
            budget = budgets[layer_idx]
            
            K_comp, V_comp, _ = select_tokens(
                K.squeeze(0).squeeze(0),  # 去掉batch和head维度进行选择
                V.squeeze(0).squeeze(0),
                attn,
                budget
            )
            compressed_kv_cache.append((K_comp, V_comp))
        
        return compressed_kv_cache, ranks, budgets
```

---

## 3. 实验设计

### 3.1 实验环境

| 项目 | 配置 |
|------|------|
| 模型 | Pythia-2.8B |
| 数据集 | WikiText-2, PG-19 |
| GPU | 根据实际情况 |
| 框架 | PyTorch, HuggingFace Transformers |

### 3.2 评测指标

| 指标 | 说明 |
|------|------|
| **PPL (Perplexity)** | 语言模型困惑度，越低越好 |
| **压缩率** | 压缩后KV Cache大小 / 原始大小 |
| **推理速度** | Tokens per second |
| **显存占用** | Peak GPU memory |

### 3.3 Baseline方法

| 方法 | 说明 |
|------|------|
| **Dense** | 不压缩，作为性能上界 |
| **StreamingLLM** | 保留sink tokens + recent tokens |
| **SnapKV** | 基于attention score选择token，每层相同budget |
| **PyramidKV** | 金字塔形状的固定budget分配 |
| **RankKV (Ours)** | 基于有效秩的自适应budget分配 |

### 3.4 实验列表

#### 实验1：分析实验（验证低秩性存在）

**目的**：展示Pythia-2.8B各层Attention矩阵的有效秩分布

**内容**：
- 在WikiText-2上采样多个文本
- 计算每层的平均有效秩
- 绘制 Layer vs Effective Rank 图

**预期输出**：
- 图表：各层有效秩分布
- 发现：不同层确实有不同的有效秩

#### 实验2：主实验（对比各方法）

**目的**：对比RankKV与baseline方法的性能

**设置**：
- 压缩率：30%, 50%, 70%
- 数据集：WikiText-2, PG-19

**输出**：

| Method | WikiText-2 PPL | PG-19 PPL | 压缩率 |
|--------|----------------|-----------|--------|
| Dense | - | - | 100% |
| StreamingLLM | - | - | 30% |
| SnapKV | - | - | 30% |
| PyramidKV | - | - | 30% |
| **RankKV** | - | - | 30% |

#### 实验3：Ablation Study

**目的**：验证各组件的贡献

**对比**：
1. RankKV (完整方法)
2. Uniform Budget (每层相同budget)
3. Random Budget (随机分配budget)
4. Inverse RankKV (rank低的层反而分配更多budget)

#### 实验4：可视化分析

**内容**：
- 各层有效秩分布柱状图
- Budget分配对比图（RankKV vs PyramidKV vs Uniform）
- PPL vs 压缩率曲线
- 不同序列长度下的rank变化

---

## 4. 代码结构

```
rankkv/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖
├── rankkv/
│   ├── __init__.py
│   ├── rank_analysis.py      # 有效秩计算
│   ├── budget_allocation.py  # Budget分配
│   ├── token_selection.py    # Token选择
│   ├── compression.py        # KV Cache压缩
│   └── pipeline.py           # 完整pipeline
├── baselines/
│   ├── streaming_llm.py
│   ├── snapkv.py
│   └── pyramidkv.py
├── experiments/
│   ├── analysis.py           # 实验1：分析实验
│   ├── main_exp.py           # 实验2：主实验
│   ├── ablation.py           # 实验3：消融实验
│   └── visualization.py      # 实验4：可视化
├── scripts/
│   ├── run_analysis.sh
│   ├── run_main_exp.sh
│   └── run_ablation.sh
└── results/
    ├── figures/
    └── tables/
```

---

## 5. 分工安排

### 5.1 成员分工

| 成员 | 职责 | 具体任务 | 工作量 |
|------|------|----------|--------|
| **成员A** | 算法实现 | 核心代码开发、Pipeline搭建、Debug | 40% |
| **成员B** | 实验运行 | 跑实验、收集数据、制作图表 | 35% |
| **成员C** | 论文撰写 | 写论文、制作示意图、整理仓库 | 25% |

### 5.2 详细任务分配

#### 成员A：算法实现

**Week 1**:
- [ ] 搭建项目框架
- [ ] 实现 `compute_effective_rank()` 函数
- [ ] 实现 `allocate_budget()` 函数
- [ ] 实现 `select_tokens()` 函数
- [ ] 实现完整的 `RankKV` 类

**Week 2**:
- [ ] 实现baseline方法（StreamingLLM, SnapKV, PyramidKV）
- [ ] Debug和优化
- [ ] 协助成员B跑实验

#### 成员B：实验运行

**Week 1**:
- [ ] 配置实验环境
- [ ] 下载模型和数据集
- [ ] 跑Dense baseline
- [ ] 跑StreamingLLM、SnapKV、PyramidKV baseline

**Week 2**:
- [ ] 跑RankKV主实验
- [ ] 跑Ablation实验
- [ ] 整理所有数据，制作表格
- [ ] 制作图表（matplotlib/seaborn）

#### 成员C：论文撰写

**Week 1**:
- [ ] 阅读相关论文，整理Related Work
- [ ] 撰写Introduction初稿
- [ ] 撰写Method初稿
- [ ] 制作方法示意图

**Week 2**:
- [ ] 根据实验结果撰写Experiments部分
- [ ] 撰写Abstract和Conclusion
- [ ] 整理GitHub仓库，写README
- [ ] 论文润色

### 5.3 时间线

```
Week 1 (Day 1-7):
├── Day 1-2: A搭建框架, B配环境, C读论文
├── Day 3-4: A实现核心函数, B跑baseline, C写Intro
├── Day 5-7: A完成Pipeline, B完成baseline, C写Method

Week 2 (Day 8-14):
├── Day 8-10: A实现baseline+debug, B跑主实验, C根据结果写Exp
├── Day 11-12: A优化代码, B跑Ablation+做图, C写Abstract/Conclusion
├── Day 13-14: 全员整合、检查、提交
```

---

## 6. 论文结构

### 6.1 大纲（4页 NeurIPS格式）

```
Abstract (0.3页)
- 问题：KV Cache是LLM推理瓶颈
- 观察：不同层Attention矩阵有效秩不同
- 方法：RankKV，基于有效秩的自适应压缩
- 结果：在Pythia-2.8B上优于baseline

1. Introduction (0.8页)
- 1.1 背景：LLM推理效率问题
- 1.2 现有方法局限：固定压缩策略忽略层间差异
- 1.3 我们的观察：有效秩反映层的压缩需求
- 1.4 贡献：(1)首次用有效秩指导压缩 (2)无训练方法 (3)与现有方法正交

2. Method (1.2页)
- 2.1 Preliminaries：KV Cache和压缩
- 2.2 Effective Rank分析
- 2.3 Rank-guided Budget Allocation
- 2.4 算法流程（配伪代码和示意图）

3. Experiments (1.5页)
- 3.1 实验设置（模型、数据、baseline、指标）
- 3.2 主实验结果（表格）
- 3.3 Ablation Study
- 3.4 分析与讨论

4. Conclusion (0.2页)
- 总结贡献
- 局限性和未来工作

References
```

### 6.2 关键图表

1. **Figure 1**: RankKV整体框架图
2. **Figure 2**: 各层有效秩分布
3. **Figure 3**: Budget分配对比（RankKV vs PyramidKV vs Uniform）
4. **Table 1**: 主实验结果对比
5. **Table 2**: Ablation实验结果

---

## 7. 参考文献

核心参考：
1. StreamingLLM: Efficient Streaming Language Models with Attention Sinks
2. SnapKV: LLM Knows What You Are Looking For Before Generation
3. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
4. DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs
5. Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation

有效秩参考：
6. Roy & Vetterli (2007): "The Effective Rank: A Measure of Effective Dimensionality"

---

## 8. 风险与应对

| 风险 | 应对策略 |
|------|----------|
| 各层rank差异不明显 | 如实报告，分析原因，这本身也是发现 |
| 性能提升不明显 | 作业要求"不追求高性能"，重点放在分析 |
| SVD计算开销大 | 只在Prefill做一次，可接受 |
| 代码bug | 预留debug时间，增量测试 |
| 时间不够 | 优先保证核心实验，砍掉部分Ablation |

---

## 9. Checklist

### 提交前检查

- [ ] 代码可运行，结果可复现
- [ ] README清晰，包含运行说明
- [ ] 论文格式正确（NeurIPS模板，≤4页）
- [ ] 所有图表清晰、有标注
- [ ] 参考文献完整
- [ ] 分工说明已包含在论文中
- [ ] GitHub仓库公开

---

**祝项目顺利！🚀**