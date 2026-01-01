# Comprehensive Benchmarks Script

## 简介

`run_comprehensive_benchmarks.py` 是 RankKV 项目的核心自动化评测脚本。它旨在简化和标准化对不同 KV Cache 压缩方法（包括 SnapKV、PyramidKV 以及它们与 RankKV 结合后的版本）的性能评估流程。

## 功能特性

1.  **一键运行**:
    - 自动化执行多个独立的评测脚本，无需手动逐个运行。
    - 统一管理实验配置，如 KV Cache 预算（Budget）。

2.  **多维度评测**:
    - 支持多种压缩方法的对比：
        - **Original SnapKV**: 原始 SnapKV 方法。
        - **RankKV + SnapKV**: 结合了 RankKV 动态预算分配的 SnapKV。
        - **Original PyramidKV**: 原始 PyramidKV 方法。
        - **RankKV + PyramidKV**: 结合了 RankKV 动态预算分配的 PyramidKV。
    - 包含 **Full KV Baseline**（全量 KV Cache）作为性能上限参考。

3.  **结果汇总**:
    - 自动解析各子脚本输出的 JSON 格式结果。
    - 将关键指标（PPL, TTFT, TPOT, Throughput, GFLOPs, 显存占用等）汇总并生成 Markdown 格式的表格报告。
    - 报告默认保存在项目根目录的 `results/comprehensive_benchmark_results.md`。

## 使用方法

在 `benchmarks` 目录下直接运行脚本：

```bash
python run_comprehensive_benchmarks.py
```

## 配置说明

脚本顶部的常量定义了实验的关键参数，可根据需要进行修改：

- **`BUDGETS`**: 定义要测试的 KV Cache 预算列表（例如 `[256, 512, 1024]`）。
- **`BENCHMARKS`**: 定义要运行的评测任务列表，包括脚本路径、方法名称等。
- **`OUTPUT_FILE`**: 定义最终汇总报告的输出路径。

## 输出示例

脚本运行完成后，生成的 Markdown 报告将包含如下格式的表格：

| Method | Type | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | ... |
|---|---|---|---|---|---|---|---|
| full_kv | baseline | wikitext | 10.5 | 50.2 | 45.1 | 22.1 | ... |
| snapkv | original | wikitext | 12.1 | 48.5 | 40.2 | 24.8 | ... |
| snapkv | rankkv | wikitext | 11.8 | 49.0 | 41.0 | 24.3 | ... |

## 注意事项

- **路径依赖**: 脚本假设项目结构保持标准形式，依赖相对路径 `../RankKV_Snap/` 和 `../RankKV_Pyramid/` 等来定位子脚本。
- **环境要求**: 请确保运行环境已安装所有必要的 Python 依赖库（`torch`, `transformers`, `datasets` 等），并建议在支持 GPU 的环境中运行以获得准确的性能数据。
