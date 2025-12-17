"""
RankKV: 基于Attention矩阵有效秩的KV Cache压缩方法

核心模块:
- rank_analysis: 有效秩计算与分析
- budget_allocation: 基于有效秩的Budget分配
- pipeline: 完整的分析流程
"""

from .rank_analysis import (
    compute_effective_rank,
    compute_effective_rank_batch,
    analyze_attention_ranks,
)
from .budget_allocation import (
    allocate_budget_proportional,
    allocate_budget_softmax,
    allocate_budget_adaptive,
    BudgetAllocator,
)
from .pipeline import RankKVAnalyzer

__version__ = "0.1.0"
__all__ = [
    "compute_effective_rank",
    "compute_effective_rank_batch",
    "analyze_attention_ranks",
    "allocate_budget_proportional",
    "allocate_budget_softmax",
    "allocate_budget_adaptive",
    "BudgetAllocator",
    "RankKVAnalyzer",
]
