# RankKV: 基于Attention矩阵有效秩的KV Cache压缩框架

RankKV 是一个**层级自适应的 KV Cache 压缩框架**。它基于一个核心观察：**不同层的 Attention 矩阵具有不同的"有效秩" (Effective Rank)**，这反映了该层对上下文信息的依赖程度。

RankKV 通过在 Prefill 阶段分析每一层的有效秩，**动态分配**每一层的 KV Cache 预算 (Budget)，并将这些预算应用到现有的主流压缩方法（如 SnapKV, H2O, PyramidKV）中，从而在不增加显存开销的前提下显著提升模型在长文本任务上的表现。

---

## 🏗️ 项目架构

RankKV 采用 **动态分析 + 实时注入** 的工作模式，分为两个阶段：

1.  **Phase 1: Dynamic Analysis (动态分析)**
    - 在推理开始前（或首个batch），利用 `RankKVAnalyzer` 对输入样本的前 N 个 token 进行快速 Forward。
    - 计算每一层的 Effective Rank。
    - 使用 `BudgetAllocator` 为每一层分配个性化的 KV Cache 预算。
    - **特点**: 每个输入样本（Input-Aware）都会得到其专属的最佳 Budget 分配方案。

2.  **Phase 2: Execution (执行压缩)**
    - 将动态计算出的 `LAYER_BUDGETS` 注入到下游压缩器（如 SnapKV/PyramidKV）。
    - 启动标准推理流程，底层压缩器根据该 Budget 动态管理 KV Cache。

目录结构如下：

```text
RankKV/
├── rankkv/                 # [核心层] Brain: 分析与决策
│   ├── rank_analysis.py    # 计算 Attention 矩阵的 Effective Rank
│   └── budget_allocation.py# 策略算法 (Proportional, Softmax, Adaptive)
│   └── pipeline.py         # RankKVAnalyzer: 串联分析与分配的枢纽
│
├── RankKV_Snap/            # [集成层] SnapKV Integration
│   └── benchmark_rankkv_snap.py # 实现了上述 Phase 1 + Phase 2 的完整流程
│
├── RankKV_Pyramid/         # [集成层] PyramidKV Integration
│   └── benchmark_rankkv_vs_standard.py
│
├── RankKV_H2O/             # [集成层] H2O Integration
│
└── benchmarks/             # [评测层] Unified Benchmarking
    └── run_comprehensive.py# 一键运行所有评测并汇总报告
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 运行 Dynamic Benchmark

我们的 benchmark 脚本已经内置了动态分析逻辑，直接运行即可体验 **Effective Rank -> Dynamic Budget** 的全过程。

#### 运行 SnapKV 集成版

```bash
# 进入对应目录
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 脚本会自动：
# 1. 加载未经修改的模型进行 Phase 1 分析
# 2. 计算出针对该输入的最佳 Budget
# 3. 动态 Patch 模型并运行 Phase 2 压缩推理
python RankKV_Snap/benchmark_rankkv_snap.py --budget 128
```

#### 运行 PyramidKV 集成版

```bash
python RankKV_Pyramid/benchmark_rankkv_vs_standard.py --budget 128
```

---

## 🧩 核心模块详解

### `RankKVAnalyzer` (pipeline.py)
这是连接 Brain 和 Body 的核心 API。

```python
# 初始化
analyzer = RankKVAnalyzer(model)

# Phase 1: 动态分析
# input_ids: 当前输入的 tokens
# total_budget: 你希望的总显存预算
analysis_result = analyzer.analyze(input_ids, total_budget=4096)

# 获取分配结果
layer_budgets = analysis_result.budgets 
# e.g., [32, 32, 128, 512, 64, ...] -> 高秩层分得多，低秩层分得少
```

### 接入下游方法
我们通过全局变量注入或 MonkeyPatch 的方式，将 `layer_budgets` 传递给下游。

```python
# 以 SnapKV 为例
import snapkv.snapkv_utils as snapkv_utils
snapkv_utils.LAYER_BUDGETS = {i: budget for i, budget in enumerate(layer_budgets)}
```

---

## 📚 参考文献

1. **StreamingLLM**: Efficient Streaming Language Models with Attention Sinks
2. **SnapKV**: LLM Knows What You Are Looking For Before Generation
3. **PyramidKV**: Dynamic KV Cache Compression based on Pyramidal Information Funneling
4. **H2O**: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
