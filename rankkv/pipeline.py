"""
RankKV 完整分析流程
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from .rank_analysis import analyze_attention_ranks, analyze_rank_distribution
from .budget_allocation import BudgetAllocator, allocate_budget_uniform, allocate_budget_pyramid


@dataclass
class RankKVAnalysisResult:
    """RankKV分析结果"""
    layer_ranks: List[float]
    rank_statistics: Dict[str, float]
    rank_distribution: Dict
    budgets: List[int]
    total_budget: int
    compression_ratio: float
    allocation_strategy: str
    num_layers: int
    seq_len: int
    model_name: Optional[str]
    
    def __repr__(self):
        return (
            f"RankKVAnalysisResult(model={self.model_name}, "
            f"layers={self.num_layers}, compression={self.compression_ratio:.2%})"
        )
    
    def to_dict(self) -> Dict:
        return {
            "layer_ranks": self.layer_ranks,
            "rank_statistics": self.rank_statistics,
            "budgets": self.budgets,
            "total_budget": self.total_budget,
            "compression_ratio": self.compression_ratio,
            "num_layers": self.num_layers,
            "seq_len": self.seq_len,
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "RankKV Analysis Summary",
            "=" * 50,
            f"Model: {self.model_name or 'Unknown'}",
            f"Layers: {self.num_layers}, Seq Length: {self.seq_len}",
            f"Compression: {self.compression_ratio:.2%}, Budget: {self.total_budget}",
            f"Strategy: {self.allocation_strategy}",
            "-" * 50,
            f"Rank - Mean: {self.rank_statistics['mean']:.4f}, Std: {self.rank_statistics['std']:.4f}",
            f"Rank - Min: {self.rank_statistics['min']:.4f} (L{self.rank_statistics['min_layer']})",
            f"Rank - Max: {self.rank_statistics['max']:.4f} (L{self.rank_statistics['max_layer']})",
            "-" * 50,
        ]
        for i, (r, b) in enumerate(zip(self.layer_ranks, self.budgets)):
            lines.append(f"  Layer {i:2d}: rank={r:.4f}, budget={b:4d}")
        lines.append("=" * 50)
        return "\n".join(lines)


class RankKVAnalyzer:
    """RankKV分析器 - 在Prefill阶段分析有效秩并分配Budget"""
    
    def __init__(self, model: nn.Module, model_name: Optional[str] = None, 
                 device: Optional[Union[str, torch.device]] = None):
        self.model = model
        self.model_name = model_name
        self.device = device or (next(model.parameters()).device if hasattr(model, 'parameters') else torch.device("cpu"))
        self.budget_allocator = BudgetAllocator(strategy="proportional")
    
    def set_budget_strategy(self, strategy: str = "proportional", min_budget: int = 4, 
                           max_budget: Optional[int] = None, **kwargs):
        self.budget_allocator = BudgetAllocator(strategy=strategy, min_budget=min_budget, 
                                                 max_budget=max_budget, **kwargs)
    
    @torch.no_grad()
    def get_attention_weights(self, input_ids: torch.Tensor, 
                              attention_mask: Optional[torch.Tensor] = None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            output_attentions=True, use_cache=False)
        return outputs.attentions, input_ids.shape[1]
    
    def analyze(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                compression_ratio: float = 0.3, total_budget: Optional[int] = None,
                aggregation: str = "mean", normalize_rank: bool = True,
                show_progress: bool = True) -> RankKVAnalysisResult:
        """执行完整的RankKV分析"""
        attentions, seq_len = self.get_attention_weights(input_ids, attention_mask)
        return self.analyze_from_attentions(attentions, seq_len, compression_ratio, 
                                            total_budget, aggregation, normalize_rank, show_progress)
    
    def analyze_from_attentions(self, attentions: Tuple[torch.Tensor, ...], seq_len: int,
                                compression_ratio: float = 0.3, total_budget: Optional[int] = None,
                                aggregation: str = "mean", normalize_rank: bool = True,
                                show_progress: bool = True) -> RankKVAnalysisResult:
        """从已有的Attention权重进行分析"""
        num_layers = len(attentions)
        
        rank_analysis = analyze_attention_ranks(attentions, aggregation=aggregation,
                                                normalize=normalize_rank, show_progress=show_progress)
        layer_ranks = rank_analysis["layer_ranks"]
        rank_distribution = analyze_rank_distribution(layer_ranks)
        
        if total_budget is None:
            total_budget = int(seq_len * compression_ratio)
        
        budget_result = self.budget_allocator.allocate(layer_ranks=layer_ranks,
                                                        total_budget=total_budget, seq_len=seq_len)
        
        return RankKVAnalysisResult(
            layer_ranks=layer_ranks, rank_statistics=rank_analysis["statistics"],
            rank_distribution=rank_distribution, budgets=budget_result.budgets,
            total_budget=budget_result.total_budget, compression_ratio=budget_result.compression_ratio,
            allocation_strategy=budget_result.allocation_strategy, num_layers=num_layers,
            seq_len=seq_len, model_name=self.model_name,
        )
    
    def compare_with_baselines(self, input_ids: torch.Tensor, 
                               compression_ratio: float = 0.3) -> Dict[str, List[int]]:
        """与Baseline方法比较Budget分配"""
        result = self.analyze(input_ids, compression_ratio=compression_ratio, show_progress=False)
        n, total = result.num_layers, result.total_budget
        
        return {
            "RankKV": result.budgets,
            "Uniform": allocate_budget_uniform(n, total),
            "Pyramid_Dec": allocate_budget_pyramid(n, total, "decreasing"),
            "Pyramid_Inc": allocate_budget_pyramid(n, total, "increasing"),
        }


def quick_analyze(model: nn.Module, input_ids: torch.Tensor, compression_ratio: float = 0.3,
                  strategy: str = "proportional", **kwargs) -> Tuple[List[int], List[float]]:
    """快速分析 - 一行代码获取Budget分配"""
    analyzer = RankKVAnalyzer(model)
    analyzer.set_budget_strategy(strategy, **kwargs)
    result = analyzer.analyze(input_ids, compression_ratio=compression_ratio, show_progress=False)
    return result.budgets, result.layer_ranks
