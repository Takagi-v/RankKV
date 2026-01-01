# RankKV with PyramidKV Integration

本目录包含将 **RankKV** 的预算分配策略与 **PyramidKV** 结合的相关代码和评测脚本。PyramidKV 是一种保留 Attention Sink（注意力汇聚点）的 KV Cache 压缩方法，RankKV 在此基础上根据层级 Effective Rank 动态分配预算。

## 文件功能说明

### 核心评测脚本

- **`benchmark_ablation_pyramid.py`**
  - **功能**: 执行消融实验，验证 RankKV 预算分配策略的有效性。
  - **内容**: 对比 "Random"（随机分配）和 "Inverse"（逆序分配）两种策略与标准 RankKV（正序分配）的性能差异。
  - **输出**: 结果通常记录在项目根目录的 `results/comprehensive_benchmark_results.md` 中。

- **`benchmark_original_pyramid.py`**
  - **功能**: 评测原始 PyramidKV 方法的性能。
  - **内容**: 使用固定的每层预算（例如 128/256 tokens）进行评测，作为基线（Baseline）参考。

- **`benchmark_rankkv_vs_standard.py`**
  - **功能**: 核心对比实验脚本。
  - **内容**: 直接对比 RankKV 增强版 PyramidKV 与标准 PyramidKV 的各项指标（PPL, TTFT, TPOT, Throughput 等）。

- **`benchmark_rankkv_datasets.py`**
  - **功能**: 多数据集评测。
  - **内容**: 在不同数据集（如 WikiText-2, PG-19）上评估 RankKV 的通用性和鲁棒性。

- **`benchmark_rankkv_pythia.py`**
  - **功能**: 模型层级分析与可视化。
  - **内容**: 专门针对 Pythia 模型（如 `pythia-2.8b`），计算并绘制各层的 Effective Rank，生成可视化图表（`layer_ranks.png`）。

### 核心模块 (`pyramidkv/`)

- **`pyramidkv/monkeypatch.py`**: 包含对 HuggingFace 模型（如 GPT-NeoX, Llama, Mistral）的 Monkey Patch 代码，用于替换标准的 Attention 实现以支持 PyramidKV 逻辑。
- **`pyramidkv/pyramidkv_utils.py`**: 实现 PyramidKV 的核心压缩逻辑，包括 KV Cache 的管理和驱逐策略。
- **`pyramidkv/*_model.py`**: 针对不同模型架构的具体 Attention 修改实现。

## 使用方法

所有脚本均可在项目根目录下通过 Python 直接运行。脚本内部会自动处理路径依赖，调用根目录的 `rankkv` 核心模块。

```bash
# 运行消融实验
python RankKV_Pyramid/benchmark_ablation_pyramid.py

# 运行原始 PyramidKV 基准测试
python RankKV_Pyramid/benchmark_original_pyramid.py

# 运行 RankKV 与 标准 PyramidKV 的对比
python RankKV_Pyramid/benchmark_rankkv_vs_standard.py
```

## 注意事项

- 运行前请确保已安装必要的依赖（`transformers`, `datasets`, `torch`, `matplotlib` 等）。
- 脚本默认使用 `cuda` 进行加速，请确保环境支持 GPU。
- 部分脚本会读取根目录 `results/rank_analysis.json` 中的秩信息，请确保该文件存在（通常由 `rank_analysis_exp.py` 生成）。
