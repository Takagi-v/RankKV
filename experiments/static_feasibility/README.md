
# Static Feasibility Experiments

This directory contains experiments to verify the feasibility of shifting from **Dynamic (Input-Aware)** budget allocation to **Static (Pre-computed)** allocation for RankKV.

## Motivation

While RankKV's core insight is based on the effective rank of attention matrices, calculating this rank dynamically during inference (via SVD) introduces significant latency overhead. We hypothesize that the rank distribution is relatively stable across different inputs for a given model, allowing us to use a pre-computed "Static Profile" without sacrificing performance.

## Experiments

### 1. Rank Consistency Verification
**Script:** `verify_consistency.py`

This script:
1. Computes a "Static Profile" by averaging the ranks of 32 calibration samples (WikiText).
2. Evaluates the consistency of this profile against 50 new test samples.
3. Computes **Pearson Correlation** and **Cosine Similarity**.

**Usage:**
```bash
python experiments/static_feasibility/verify_consistency.py
```

**Expected Output:**
- High correlation (>0.9) and similarity (>0.95).
- A plot `consistency_plot.png` showing the overlap.

### 2. Static vs. Dynamic Performance Comparison
**Script:** `verify_static_vs_dynamic.py`

This script compares two modes on the same test samples:
- **Static RankKV**: Uses the pre-computed profile from Exp 1. No runtime analysis overhead.
- **Dynamic RankKV**: Computes the rank for the *specific* input on-the-fly (simulating runtime SVD overhead).

It measures:
- **Perplexity (PPL)**: To check for performance degradation.
- **Inference Time (ms)**: To demonstrate the speedup of Static over Dynamic.

**Usage:**
```bash
python experiments/static_feasibility/verify_static_vs_dynamic.py
```

**Expected Output:**
- PPL Gap (Static - Dynamic) should be negligible (< 0.1).
- Time Dynamic > Time Static due to SVD overhead.

## Conclusion Scope
These experiments aim to provide empirical evidence that **Static RankKV is the practical choice**, offering the benefits of rank-aware allocation (improved PPL over Uniform) with zero runtime overhead compared to standard inference.
