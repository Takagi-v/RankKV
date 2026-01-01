# RankKV with SnapKV Integration

本目录包含将 **RankKV** 的预算分配策略与 **SnapKV** 结合的相关代码和评测脚本。SnapKV 是一种基于注意力模式自动选择重要 KV 对的压缩方法，RankKV 在此基础上引入层级预算动态分配以进一步提升性能。

## 文件功能说明

### 核心评测脚本

- **`benchmark_ablation_snap.py`**
  - **功能**: 执行消融实验，验证 RankKV 预算分配策略在 SnapKV 架构下的有效性。
  - **内容**: 对比 "Random"（随机分配）和 "Inverse"（逆序分配）策略，证明基于 Effective Rank 的分配策略的优越性。

- **`benchmark_original_snap.py`**
  - **功能**: 评测原始 SnapKV 方法的性能。
  - **内容**: 使用固定的每层预算进行评测，作为基线（Baseline）参考。

- **`benchmark_rankkv_snap.py`**
  - **功能**: RankKV (SnapKV版) 的主评测脚本。
  - **内容**: 评估结合了 RankKV 动态预算分配的 SnapKV 性能，计算 PPL、吞吐量、显存占用等关键指标。

- **`benchmark_full_kv.py`**
  - **功能**: Full KV Cache 基准测试。
  - **内容**: 在不进行任何压缩的情况下运行模型，提供性能的上限参考（最佳 PPL，最低吞吐量）。

### 核心模块 (`snapkv/`)

- **`snapkv/monkeypatch.py`**: 包含对 HuggingFace 模型（如 GPT-NeoX）的 Monkey Patch 代码，用于注入 SnapKV 的压缩逻辑。
- **`snapkv/snapkv_utils.py`**: 实现 SnapKV 的核心逻辑，包括重要 KV 对的选择、聚类和缓存更新。
- **`snapkv/gptneox_model.py`**: 针对 GPT-NeoX 模型的具体 Attention 修改实现。

## 使用方法

所有脚本均可在项目根目录下通过 Python 直接运行。脚本内部会自动处理路径依赖，调用根目录的 `rankkv` 核心模块。

```bash
# 运行消融实验
python RankKV_Snap/benchmark_ablation_snap.py

# 运行 RankKV (SnapKV集成) 评测
python RankKV_Snap/benchmark_rankkv_snap.py

# 运行原始 SnapKV 基准测试
python RankKV_Snap/benchmark_original_snap.py

# 运行 Full KV (无压缩) 基准测试
python RankKV_Snap/benchmark_full_kv.py
```

## 注意事项

- 脚本依赖根目录的 `rankkv` 模块提供的 `allocate_budget_proportional` 等函数。
- 确保根目录 `results/rank_analysis.json` 文件存在，以便脚本加载层级秩信息进行预算分配。
- 默认评测模型通常为 `EleutherAI/pythia-2.8b`，可在脚本中修改 `MODEL_NAME` 变量以更换模型。
