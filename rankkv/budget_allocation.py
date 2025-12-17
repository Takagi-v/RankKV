"""
Budget 分配模块

根据每层的有效秩(Effective Rank)分配不同的KV Cache预算(Budget)

核心策略:
    - 有效秩高 → Attention分布分散 → 需要更多token → 分配更多budget
    - 有效秩低 → Attention分布集中 → 少数token足够 → 分配更少budget
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Callable
from dataclasses import dataclass


@dataclass
class BudgetAllocationResult:
    """Budget分配结果"""
    budgets: List[int]              # 每层分配的budget
    layer_ranks: List[float]        # 每层的有效秩
    total_budget: int               # 总budget
    compression_ratio: float        # 压缩率
    allocation_strategy: str        # 使用的分配策略
    
    def __repr__(self):
        return (
            f"BudgetAllocationResult(\n"
            f"  strategy={self.allocation_strategy},\n"
            f"  total_budget={self.total_budget},\n"
            f"  compression_ratio={self.compression_ratio:.2%},\n"
            f"  budgets={self.budgets}\n"
            f")"
        )
    
    def to_dict(self) -> Dict:
        return {
            "budgets": self.budgets,
            "layer_ranks": self.layer_ranks,
            "total_budget": self.total_budget,
            "compression_ratio": self.compression_ratio,
            "allocation_strategy": self.allocation_strategy,
        }


def allocate_budget_proportional(
    ranks: List[float],
    total_budget: int,
    min_budget: int = 4,
    max_budget: Optional[int] = None,
) -> List[int]:
    """
    按比例分配Budget（线性映射）
    
    budget_i = total_budget * (rank_i / sum(ranks))
    
    有效秩高的层获得更多budget
    
    Args:
        ranks: 每层的有效秩列表
        total_budget: 总共要分配的budget
        min_budget: 每层最少的budget
        max_budget: 每层最多的budget（None表示不限制）
    
    Returns:
        budgets: 每层分配的budget列表
    """
    num_layers = len(ranks)
    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    
    # 避免除以0
    rank_sum = ranks_tensor.sum()
    if rank_sum < 1e-10:
        # 如果所有rank都接近0，均匀分配
        base_budget = total_budget // num_layers
        return [base_budget] * num_layers
    
    # 按比例分配
    ratios = ranks_tensor / rank_sum
    budgets_float = ratios * total_budget
    
    # 转换为整数
    budgets = budgets_float.int().tolist()
    
    # 应用最小和最大约束
    budgets = [max(b, min_budget) for b in budgets]
    if max_budget is not None:
        budgets = [min(b, max_budget) for b in budgets]
    
    # 调整使总和接近total_budget
    current_sum = sum(budgets)
    diff = total_budget - current_sum
    
    if diff > 0:
        # 需要增加budget，优先给rank高的层
        sorted_indices = sorted(range(num_layers), key=lambda i: ranks[i], reverse=True)
        for i in range(abs(diff)):
            idx = sorted_indices[i % num_layers]
            if max_budget is None or budgets[idx] < max_budget:
                budgets[idx] += 1
    elif diff < 0:
        # 需要减少budget，优先从rank低的层减
        sorted_indices = sorted(range(num_layers), key=lambda i: ranks[i])
        for i in range(abs(diff)):
            idx = sorted_indices[i % num_layers]
            if budgets[idx] > min_budget:
                budgets[idx] -= 1
    
    return budgets


def allocate_budget_softmax(
    ranks: List[float],
    total_budget: int,
    temperature: float = 1.0,
    min_budget: int = 4,
    max_budget: Optional[int] = None,
) -> List[int]:
    """
    使用Softmax分配Budget
    
    budget_i = total_budget * softmax(rank_i / temperature)
    
    温度参数控制分配的平滑程度:
        - temperature高 → 分配更均匀
        - temperature低 → 分配更极端（高rank层获得更多）
    
    Args:
        ranks: 每层的有效秩列表
        total_budget: 总共要分配的budget
        temperature: softmax温度参数
        min_budget: 每层最少的budget
        max_budget: 每层最多的budget
    
    Returns:
        budgets: 每层分配的budget列表
    """
    num_layers = len(ranks)
    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    
    # Softmax with temperature
    probs = torch.softmax(ranks_tensor / temperature, dim=0)
    budgets_float = probs * total_budget
    
    # 转换为整数
    budgets = budgets_float.int().tolist()
    
    # 应用约束
    budgets = [max(b, min_budget) for b in budgets]
    if max_budget is not None:
        budgets = [min(b, max_budget) for b in budgets]
    
    # 调整总和
    current_sum = sum(budgets)
    diff = total_budget - current_sum
    
    sorted_indices = sorted(range(num_layers), key=lambda i: ranks[i], reverse=True)
    
    if diff > 0:
        for i in range(abs(diff)):
            idx = sorted_indices[i % num_layers]
            if max_budget is None or budgets[idx] < max_budget:
                budgets[idx] += 1
    elif diff < 0:
        sorted_indices = sorted_indices[::-1]  # 从低rank开始减
        for i in range(abs(diff)):
            idx = sorted_indices[i % num_layers]
            if budgets[idx] > min_budget:
                budgets[idx] -= 1
    
    return budgets


def allocate_budget_adaptive(
    ranks: List[float],
    total_budget: int,
    low_rank_ratio: float = 0.5,      # 低秩层使用的budget比例
    high_rank_ratio: float = 1.5,     # 高秩层使用的budget比例
    percentile_low: float = 25,        # 低秩层的百分位阈值
    percentile_high: float = 75,       # 高秩层的百分位阈值
    min_budget: int = 4,
    max_budget: Optional[int] = None,
) -> List[int]:
    """
    自适应分配Budget
    
    根据rank的分布将层分为三类:
        - 低秩层 (rank < q1): 分配 base_budget * low_rank_ratio
        - 中等层 (q1 <= rank <= q3): 分配 base_budget
        - 高秩层 (rank > q3): 分配 base_budget * high_rank_ratio
    
    Args:
        ranks: 每层的有效秩列表
        total_budget: 总共要分配的budget
        low_rank_ratio: 低秩层的budget缩放比例
        high_rank_ratio: 高秩层的budget缩放比例
        percentile_low: 定义低秩层的百分位数
        percentile_high: 定义高秩层的百分位数
        min_budget: 每层最少的budget
        max_budget: 每层最多的budget
    
    Returns:
        budgets: 每层分配的budget列表
    """
    num_layers = len(ranks)
    ranks_array = np.array(ranks)
    
    # 计算阈值
    q_low = np.percentile(ranks_array, percentile_low)
    q_high = np.percentile(ranks_array, percentile_high)
    
    # 为每层确定权重
    weights = []
    for rank in ranks:
        if rank < q_low:
            weights.append(low_rank_ratio)
        elif rank > q_high:
            weights.append(high_rank_ratio)
        else:
            weights.append(1.0)
    
    weights = np.array(weights)
    
    # 按权重分配budget
    weight_sum = weights.sum()
    budgets_float = (weights / weight_sum) * total_budget
    budgets = budgets_float.astype(int).tolist()
    
    # 应用约束
    budgets = [max(b, min_budget) for b in budgets]
    if max_budget is not None:
        budgets = [min(b, max_budget) for b in budgets]
    
    # 调整总和
    current_sum = sum(budgets)
    diff = total_budget - current_sum
    
    sorted_indices = sorted(range(num_layers), key=lambda i: ranks[i], reverse=True)
    
    if diff > 0:
        for i in range(abs(diff)):
            idx = sorted_indices[i % num_layers]
            if max_budget is None or budgets[idx] < max_budget:
                budgets[idx] += 1
    elif diff < 0:
        sorted_indices = sorted_indices[::-1]
        for i in range(abs(diff)):
            idx = sorted_indices[i % num_layers]
            if budgets[idx] > min_budget:
                budgets[idx] -= 1
    
    return budgets


def allocate_budget_power(
    ranks: List[float],
    total_budget: int,
    power: float = 2.0,
    min_budget: int = 4,
    max_budget: Optional[int] = None,
) -> List[int]:
    """
    使用幂次方分配Budget
    
    budget_i = total_budget * (rank_i^power / sum(rank^power))
    
    power > 1: 放大高rank层的优势
    power < 1: 平滑分配
    
    Args:
        ranks: 每层的有效秩列表
        total_budget: 总共要分配的budget
        power: 幂次参数
        min_budget: 每层最少的budget
        max_budget: 每层最多的budget
    
    Returns:
        budgets: 每层分配的budget列表
    """
    num_layers = len(ranks)
    ranks_array = np.array(ranks)
    
    # 避免负数或零的问题
    ranks_array = np.maximum(ranks_array, 1e-10)
    
    # 幂次变换
    powered_ranks = np.power(ranks_array, power)
    
    # 归一化并分配
    ratios = powered_ranks / powered_ranks.sum()
    budgets_float = ratios * total_budget
    budgets = budgets_float.astype(int).tolist()
    
    # 应用约束
    budgets = [max(b, min_budget) for b in budgets]
    if max_budget is not None:
        budgets = [min(b, max_budget) for b in budgets]
    
    # 调整总和
    current_sum = sum(budgets)
    diff = total_budget - current_sum
    
    sorted_indices = sorted(range(num_layers), key=lambda i: ranks[i], reverse=True)
    
    if diff > 0:
        for i in range(abs(diff)):
            idx = sorted_indices[i % num_layers]
            if max_budget is None or budgets[idx] < max_budget:
                budgets[idx] += 1
    elif diff < 0:
        sorted_indices = sorted_indices[::-1]
        for i in range(abs(diff)):
            idx = sorted_indices[i % num_layers]
            if budgets[idx] > min_budget:
                budgets[idx] -= 1
    
    return budgets


class BudgetAllocator:
    """
    Budget分配器类
    
    统一接口管理不同的分配策略
    
    Example:
        >>> allocator = BudgetAllocator(
        ...     strategy="proportional",
        ...     min_budget=4,
        ...     max_budget=None
        ... )
        >>> result = allocator.allocate(
        ...     layer_ranks=[0.3, 0.5, 0.8, 0.2, 0.6],
        ...     total_budget=100
        ... )
        >>> print(result.budgets)
    """
    
    STRATEGIES = {
        "proportional": allocate_budget_proportional,
        "softmax": allocate_budget_softmax,
        "adaptive": allocate_budget_adaptive,
        "power": allocate_budget_power,
    }
    
    def __init__(
        self,
        strategy: str = "proportional",
        min_budget: int = 4,
        max_budget: Optional[int] = None,
        **strategy_kwargs,
    ):
        """
        初始化Budget分配器
        
        Args:
            strategy: 分配策略名称
                - "proportional": 按比例分配
                - "softmax": Softmax分配
                - "adaptive": 自适应分配（分层处理）
                - "power": 幂次方分配
            min_budget: 每层最少的budget
            max_budget: 每层最多的budget
            **strategy_kwargs: 特定策略的额外参数
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(self.STRATEGIES.keys())}"
            )
        
        self.strategy = strategy
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.strategy_kwargs = strategy_kwargs
    
    def allocate(
        self,
        layer_ranks: List[float],
        total_budget: Optional[int] = None,
        seq_len: Optional[int] = None,
        compression_ratio: float = 0.3,
    ) -> BudgetAllocationResult:
        """
        分配Budget
        
        Args:
            layer_ranks: 每层的有效秩列表
            total_budget: 总budget（如果指定，忽略seq_len和compression_ratio）
            seq_len: 序列长度（与compression_ratio一起使用计算total_budget）
            compression_ratio: 压缩比率（0-1之间，表示保留的比例）
        
        Returns:
            BudgetAllocationResult: 分配结果
        """
        # 确定total_budget
        if total_budget is None:
            if seq_len is None:
                raise ValueError("Must provide either total_budget or seq_len")
            total_budget = int(seq_len * compression_ratio)
        
        # 计算实际压缩率
        if seq_len is not None:
            actual_compression_ratio = total_budget / seq_len
        else:
            actual_compression_ratio = compression_ratio
        
        # 调用对应的分配函数
        allocate_fn = self.STRATEGIES[self.strategy]
        
        # 构建参数
        kwargs = {
            "ranks": layer_ranks,
            "total_budget": total_budget,
            "min_budget": self.min_budget,
            "max_budget": self.max_budget,
        }
        kwargs.update(self.strategy_kwargs)
        
        budgets = allocate_fn(**kwargs)
        
        return BudgetAllocationResult(
            budgets=budgets,
            layer_ranks=layer_ranks,
            total_budget=total_budget,
            compression_ratio=actual_compression_ratio,
            allocation_strategy=self.strategy,
        )
    
    @classmethod
    def compare_strategies(
        cls,
        layer_ranks: List[float],
        total_budget: int,
        strategies: Optional[List[str]] = None,
        min_budget: int = 4,
        max_budget: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """
        比较不同分配策略的结果
        
        Args:
            layer_ranks: 每层的有效秩列表
            total_budget: 总budget
            strategies: 要比较的策略列表（None表示比较所有）
            min_budget: 每层最少的budget
            max_budget: 每层最多的budget
        
        Returns:
            每种策略的分配结果字典
        """
        if strategies is None:
            strategies = list(cls.STRATEGIES.keys())
        
        results = {}
        for strategy in strategies:
            allocator = cls(
                strategy=strategy,
                min_budget=min_budget,
                max_budget=max_budget,
            )
            result = allocator.allocate(layer_ranks, total_budget=total_budget)
            results[strategy] = result.budgets
        
        return results


def allocate_budget_uniform(
    num_layers: int,
    total_budget: int,
) -> List[int]:
    """
    均匀分配Budget（作为baseline对比）
    
    每层分配相同的budget
    
    Args:
        num_layers: 层数
        total_budget: 总budget
    
    Returns:
        budgets: 每层分配的budget列表
    """
    base_budget = total_budget // num_layers
    remainder = total_budget % num_layers
    
    budgets = [base_budget] * num_layers
    
    # 把余数分配给前几层
    for i in range(remainder):
        budgets[i] += 1
    
    return budgets


def allocate_budget_pyramid(
    num_layers: int,
    total_budget: int,
    direction: str = "decreasing",
) -> List[int]:
    """
    金字塔形分配Budget（PyramidKV baseline）
    
    Args:
        num_layers: 层数
        total_budget: 总budget
        direction: 分配方向
            - "decreasing": 从底层到顶层递减
            - "increasing": 从底层到顶层递增
    
    Returns:
        budgets: 每层分配的budget列表
    """
    # 生成权重序列（1, 2, 3, ..., num_layers）
    if direction == "decreasing":
        weights = list(range(num_layers, 0, -1))
    elif direction == "increasing":
        weights = list(range(1, num_layers + 1))
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    weight_sum = sum(weights)
    budgets = [int(w / weight_sum * total_budget) for w in weights]
    
    # 调整余数
    remainder = total_budget - sum(budgets)
    for i in range(remainder):
        budgets[i] += 1
    
    return budgets
