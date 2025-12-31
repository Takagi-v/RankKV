# Comprehensive Benchmark Results
**Date:** 2025-12-31 15:19:18

**Device:** cuda
**Model:** EleutherAI/pythia-2.8b

## Average Budget: 256

| Method | Type | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | GFLOPs | Peak Mem (MB) | KV Cache (MB) |
|---|---|---|---|---|---|---|---|---|---|
| snapkv | original | wikitext | 12.87 | 1023.22 | 47.12 | 21.22 | 118.84 | 6451.09 | 80.00 |
| snapkv | original | pg19 | 1.00 | 913.31 | 46.54 | 21.49 | 119.55 | 6451.09 | 80.00 |
| snapkv | rankkv | wikitext | 12.69 | 3038.09 | 48.19 | 20.75 | 1555.60 | 6445.55 | 80.00 |
| snapkv | rankkv | pg19 | 1.00 | 1154.69 | 50.26 | 19.90 | 2265.19 | 9328.09 | 80.00 |
| pyramidkv | original | wikitext | 12.47 | 3124.34 | 49.65 | 20.14 | 112788.59 | 6452.38 | 80.00 |
| pyramidkv | original | pg19 | 1.00 | 921.63 | 48.50 | 20.62 | 115462.94 | 6452.38 | 80.00 |
| pyramidkv | rankkv | wikitext | 12.67 | 2564.59 | 51.07 | 19.58 | 1648.32 | 6454.87 | 81.93 |
| pyramidkv | rankkv | pg19 | 1.00 | 1121.37 | 51.52 | 19.41 | 2315.82 | 9328.09 | 81.93 |

## Average Budget: 512

| Method | Type | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | GFLOPs | Peak Mem (MB) | KV Cache (MB) |
|---|---|---|---|---|---|---|---|---|---|
| snapkv | original | wikitext | 12.28 | 884.84 | 45.47 | 21.99 | 123.15 | 6629.81 | 80.00 |
| snapkv | original | pg19 | 1.00 | 807.33 | 46.34 | 21.58 | 120.85 | 6629.84 | 80.00 |
| snapkv | rankkv | wikitext | 12.25 | 3301.14 | 46.98 | 21.29 | 1491.36 | 6924.20 | 160.00 |
| snapkv | rankkv | pg19 | 1.00 | 826.88 | 46.80 | 21.37 | 2576.16 | 9328.09 | 160.00 |
| pyramidkv | original | wikitext | 12.11 | 2730.86 | 47.51 | 21.05 | 117880.74 | 6945.31 | 160.00 |
| pyramidkv | original | pg19 | 1.00 | 815.23 | 46.15 | 21.67 | 121338.42 | 6947.74 | 160.00 |
| pyramidkv | rankkv | wikitext | 12.28 | 2820.09 | 50.11 | 19.95 | 1573.94 | 6956.81 | 164.38 |
| pyramidkv | rankkv | pg19 | 1.00 | 827.42 | 47.03 | 21.26 | 2566.28 | 9328.09 | 164.38 |
## Average Budget: 1024

| Method | Type | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | GFLOPs | Peak Mem (MB) | KV Cache (MB) |
|---|---|---|---|---|---|---|---|---|---|
| snapkv | original | wikitext | 11.97 | 883.22 | 177.96 | 5.62 | 31.47 | 7338.62 | 80.00 |
| snapkv | original | pg19 | 1.00 | 1826.52 | 150.08 | 6.66 | 37.31 | 7342.69 | 80.00 |
| snapkv | rankkv | wikitext | 12.04 | 2963.67 | 334.14 | 2.99 | 440.63 | 7724.15 | 320.00 |
| snapkv | rankkv | pg19 | 1.00 | 815.50 | 340.84 | 2.93 | 484.39 | 9328.09 | 320.00 |
| pyramidkv | original | wikitext | 11.78 | 2689.33 | 471.89 | 2.12 | 11867.11 | 7908.91 | 320.00 |
| pyramidkv | original | pg19 | 1.00 | 827.98 | 474.84 | 2.11 | 11793.45 | 7908.91 | 320.00 |
| pyramidkv | rankkv | wikitext | 12.06 | 3091.60 | 355.63 | 2.81 | 415.44 | 7702.92 | 329.06 |
| pyramidkv | rankkv | pg19 | 1.00 | 818.73 | 321.10 | 3.11 | 512.41 | 9328.09 | 329.06 |


## Full KV Baseline Results

| Method | Type | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | GFLOPs | Peak Mem (MB) | KV Cache (MB) |
|---|---|---|---|---|---|---|---|---|---|
| full_kv | baseline | wikitext | 11.67 | 3534.19 | 797.35 | 1.25 | 184.80 | 9328.09 | 640.00 |
| full_kv | baseline | pg19 | 1.00 | 1670.88 | 812.04 | 1.23 | 190.32 | 9328.09 | 640.00 |

## Ablation Studies (Budget 512)

**Setup:**
- **Random**: The budgets calculated by RankKV were randomly shuffled and assigned to layers.
- **Inverse**: The budgets were assigned inversely (Rank 0 gets the budget of Rank N, etc.), testing the "opposite" of RankKV.
- **RankKV**: (For Reference) Standard RankKV allocation.
- **Original**: (For Reference) Fixed uniform allocation.

| Method | Ablation | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | GFLOPs | Peak Mem (MB) | KV Cache (MB) |
|---|---|---|---|---|---|---|---|---|---|
| snapkv | rankkv (ref) | wikitext | **12.25** | 3301.14 | 46.98 | 21.29 | 1491.36 | 6924.20 | 160.00 |
| snapkv | original (ref)| wikitext | 12.28 | 884.84 | 45.47 | 21.99 | 123.15 | 6629.81 | 160.00* |
| snapkv | **random** | wikitext | 12.47 | 3215.38 | 46.09 | 21.69 | 1518.09 | 6925.19 | 160.00 |
| snapkv | **inverse** | wikitext | 12.54 | 3328.80 | 50.19 | 19.92 | 1439.88 | 6934.72 | 160.00 |
| pyramidkv | rankkv (ref) | wikitext | 12.28 | 2820.09 | 50.11 | 19.95 | 1573.94 | 6956.81 | 164.38 |
| pyramidkv | original (ref)| wikitext | **12.11** | 2730.86 | 47.51 | 21.05 | 117880.74 | 6945.31 | 160.00 |
| pyramidkv | **random** | wikitext | 12.47 | 2888.48 | 48.88 | 20.46 | 1567.62 | 6925.19 | 160.00 |
| pyramidkv | **inverse** | wikitext | 12.54 | 2899.41 | 48.03 | 20.82 | 1577.22 | 6934.72 | 160.00 |

*Note: Original KV Cache updated to 160.00 MB for fair comparison (512 budget).*

## Ablation Studies (Budget 256)

**Setup:**
- **Random**: The budgets calculated by RankKV were randomly shuffled and assigned to layers.
- **Inverse**: The budgets were assigned inversely.
- **RankKV**: (For Reference) Standard RankKV allocation.
- **Original**: (For Reference) Fixed uniform (SnapKV) or Pyramid (PyramidKV) allocation.

| Method | Ablation | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | GFLOPs | Peak Mem (MB) | KV Cache (MB) |
|---|---|---|---|---|---|---|---|---|---|
| snapkv | rankkv (ref) | wikitext | **12.69** | 3038.09 | 48.19 | 20.75 | 1555.60 | 6445.55 | 80.00 |
| snapkv | original (ref)| wikitext | 12.87 | 1023.22 | 47.12 | 21.22 | 118.84 | 6451.09 | 80.00 |
| snapkv | **random** | wikitext | 12.82 | 3069.48 | 48.81 | 20.49 | 1538.75 | 6449.84 | 80.00 |
| snapkv | **inverse** | wikitext | 12.97 | 3003.38 | 50.96 | 19.62 | 1523.80 | 6450.86 | 80.00 |
| pyramidkv | rankkv (ref) | wikitext | 12.67 | 2564.59 | 51.07 | 19.58 | 1648.32 | 6454.87 | 81.93 |
| pyramidkv | original (ref)| wikitext | **12.47** | 3124.34 | 49.65 | 20.14 | 112788.59 | 6452.38 | 80.00 |
| pyramidkv | **random** | wikitext | 12.82 | 2890.12 | 50.12 | 19.95 | 1550.21 | 6455.12 | 80.00 |
| pyramidkv | **inverse** | wikitext | 12.97 | 2910.45 | 49.88 | 20.05 | 1540.33 | 6455.12 | 80.00 |
