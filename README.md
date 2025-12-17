# RankKV: 基于Attention矩阵有效秩的KV Cache压缩方法

## 项目简介

RankKV是一种基于Attention矩阵**有效秩(Effective Rank)**的**层级自适应KV Cache压缩方法**。

### 核心观察

不同层的Attention矩阵具有不同的有效秩：
- **有效秩低** → Attention分布集中 → 可激进压缩
- **有效秩高** → Attention分布分散 → 需保守压缩

### 方法概述

1. 在Prefill阶段计算每层Attention矩阵的有效秩
2. 根据有效秩为每层分配不同的KV Cache预算(Budget)
3. 使用现有token selection方法（如SnapKV）执行压缩

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rankkv import RankKVAnalyzer, quick_analyze

# 加载模型
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")

# 准备输入
text = "Your input text here..."
input_ids = tokenizer(text, return_tensors="pt").input_ids

# 方式1：使用分析器
analyzer = RankKVAnalyzer(model, model_name="pythia-2.8b")
result = analyzer.analyze(input_ids, compression_ratio=0.3)

print(result.summary())
print(f"Layer budgets: {result.budgets}")
print(f"Layer ranks: {result.layer_ranks}")

# 方式2：快速分析
budgets, ranks = quick_analyze(model, input_ids, compression_ratio=0.3)
```

## 核心模块

### 1. 有效秩计算 (rank_analysis.py)

```python
from rankkv import compute_effective_rank, analyze_attention_ranks

# 单个矩阵
rank = compute_effective_rank(attention_matrix)

# 分析所有层
analysis = analyze_attention_ranks(model_attentions)
```

### 2. Budget分配 (budget_allocation.py)

```python
from rankkv import BudgetAllocator

# 创建分配器
allocator = BudgetAllocator(
    strategy="proportional",  # 或 "softmax", "adaptive", "power"
    min_budget=4,
)

# 分配budget
result = allocator.allocate(
    layer_ranks=ranks,
    total_budget=1000,
)
```

### 3. 完整Pipeline (pipeline.py)

```python
from rankkv import RankKVAnalyzer

analyzer = RankKVAnalyzer(model)
result = analyzer.analyze(input_ids, compression_ratio=0.3)

# 获取结果供下游方法使用
budgets = result.budgets      # 每层的budget
ranks = result.layer_ranks    # 每层的有效秩
```

## Budget分配策略

| 策略 | 说明 |
|------|------|
| `proportional` | 按rank比例线性分配 |
| `softmax` | 使用softmax(rank/temperature)分配 |
| `adaptive` | 根据分位数分层处理 |
| `power` | 使用rank^power分配 |

## 运行实验

### 低秩性分析实验

```bash
python experiments/rank_analysis_exp.py \
    --model EleutherAI/pythia-2.8b \
    --dataset wikitext \
    --num_samples 10
```

### 测试

```bash
python tests/test_core.py
```

## 项目结构

```
rankkv/
├── README.md
├── requirements.txt
├── rankkv/
│   ├── __init__.py
│   ├── rank_analysis.py      # 有效秩计算
│   ├── budget_allocation.py  # Budget分配
│   └── pipeline.py           # 完整Pipeline
├── experiments/
│   └── rank_analysis_exp.py  # 分析实验
├── tests/
│   └── test_core.py          # 核心测试
└── results/
    └── figures/
```

## 接入下游方法

RankKV的输出（每层的budget）可以直接用于：
- **SnapKV**: 修改其每层的token selection数量
- **H2O**: 修改其每层的cache大小
- **PyramidKV**: 替换其固定的金字塔分配

示例：
```python
# 获取RankKV的budget分配
result = analyzer.analyze(input_ids, compression_ratio=0.3)
budgets = result.budgets

# 传递给下游方法
# snapkv_compress(kv_cache, budgets=budgets)
# h2o_compress(kv_cache, budgets=budgets)
```

## 参考文献

1. StreamingLLM: Efficient Streaming Language Models with Attention Sinks
2. SnapKV: LLM Knows What You Are Looking For Before Generation
3. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
4. Roy & Vetterli (2007): The Effective Rank: A Measure of Effective Dimensionality
