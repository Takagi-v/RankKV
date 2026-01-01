# RankKV with H2O Integration

本目录包含 **RankKV** 在 H2O 压缩方法上的实现与评测代码。RankKV 通过计算层级注意力矩阵的有效秩（Effective Rank），动态分配每一层的 KV Cache 预算，从而在保持显存占用不变的情况下提升长文本性能。

## 目录结构

```text
RankKV/
├── rankkv_h2o/                # 核心功能包
│   ├── config.py              # 路径、显存预算开关
│   ├── modeling.py            # Attention 修改与 Cache 驱逐逻辑
│   ├── strategy.py            # Rank 计算与 Budget 分配算法
│   └── ...
├── benchmark_h2o.py           # 运行 H2O 基线实验
├── benchmark_rankkv.py        # 运行 RankKV (Ours) 实验
└── benchmark_full.py          # 运行 Full KV (Upper Bound) 实验
```

## 文件功能说明

### 核心评测脚本

- **`benchmark_h2o.py`**
  - **功能**: H2O 基线测试。
  - **内容**: 运行标准的 H2O 压缩算法（Fixed Budget），复现论文中的基础指标，用于作为 RankKV 的对比基准。

- **`benchmark_rankkv.py`**
  - **功能**: RankKV 主评测脚本。
  - **内容**: 包含两个阶段：
    1. **Profiling**: 预运行模型计算各层的 Effective Rank。
    2. **Generation**: 根据 Rank 动态分配 Budget 并进行生成测试，记录 PPL 和 TPOT。

### 核心模块 (`rankkv_h2o/`)

- **`rankkv_h2o/config.py`**: 全局配置文件，管理模型路径、显存预算开关及全局状态变量。
- **`rankkv_h2o/modeling.py`**: 修改后的 GPT-NeoX Attention Forward 函数，集成了 H2O 驱逐策略与 Layer-wise Budget 支持。
- **`rankkv_h2o/strategy.py`**: 实现 RankKV 的核心算法，包括 Effective Rank 计算与 Budget 分配公式。
- **`rankkv_h2o/patch.py`**: 用于动态替换模型 Attention 方法的 Monkey Patch 工具。
- **`rankkv_h2o/utils.py`**: 数据集加载（Wikitext-2）及统一的 Benchmark 流程封装。

## 使用方法

无需安装额外包，在 **`RankKV_H2O/ `**目录下运行脚本即可。
`注意一定进入该目录运行，否则可能把model缓存到更上一级的目录中`

```bash
# 1. 运行 H2O 基线实验
python benchmark_h2o.py

# 2. 运行 RankKV 实验
python benchmark_rankkv.py
```

### config 配置说明

默认的模型和输出目录在RankKV根目录下的
```
model/
results/
```
文件夹下，如需改变缓存路径，可在/rankkv_h2o/config.py下进行修改

对于实验各参数，则需在相应的benchmark_*.py文件中修改