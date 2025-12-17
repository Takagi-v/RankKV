"""
有效秩 (Effective Rank) 计算与分析模块

有效秩定义:
    erank(A) = exp(-Σ σ̃_i * log(σ̃_i))
    其中 σ̃_i = σ_i / Σ_j σ_j 是归一化的奇异值

直观理解:
    - 有效秩衡量矩阵"信息维度"的多少
    - 奇异值集中 → 有效秩低 → Attention分布集中 → 可激进压缩
    - 奇异值分散 → 有效秩高 → Attention分布分散 → 需保守压缩
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm


def compute_effective_rank(
    matrix: torch.Tensor,
    eps: float = 1e-10,
    normalize: bool = True,
) -> float:
    """
    计算单个矩阵的有效秩 (Effective Rank)
    
    基于 Roy & Vetterli (2007) 的定义:
    erank(A) = exp(H(σ̃)) = exp(-Σ σ̃_i * log(σ̃_i))
    
    Args:
        matrix: 输入矩阵，形状为 [M, N]
        eps: 数值稳定性的小常数
        normalize: 是否对有效秩进行归一化（除以min(M,N)使其在0-1之间）
    
    Returns:
        effective_rank: 有效秩值
            - 如果normalize=True，返回归一化的有效秩 (0-1)
            - 如果normalize=False，返回原始有效秩 (1 - min(M,N))
    
    Example:
        >>> attn = torch.randn(100, 100)
        >>> rank = compute_effective_rank(attn)
        >>> print(f"Effective Rank: {rank:.4f}")
    """
    # 确保输入是2D矩阵
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.dim()}D tensor")
    
    # 转换为float32以保证数值稳定性
    matrix = matrix.float()
    
    # SVD分解，只需要奇异值
    try:
        # 使用torch.linalg.svdvals更高效（只计算奇异值）
        singular_values = torch.linalg.svdvals(matrix)
    except RuntimeError:
        # 回退到完整SVD
        _, singular_values, _ = torch.svd(matrix)
    
    # 过滤掉过小的奇异值
    singular_values = singular_values[singular_values > eps]
    
    if len(singular_values) == 0:
        return 0.0
    
    # 归一化奇异值（得到概率分布）
    sv_sum = singular_values.sum()
    if sv_sum < eps:
        return 0.0
    
    sv_normalized = singular_values / sv_sum
    
    # 计算香农熵: H = -Σ p_i * log(p_i)
    log_sv = torch.log(sv_normalized + eps)
    entropy = -(sv_normalized * log_sv).sum()
    
    # 有效秩 = exp(熵)
    effective_rank = torch.exp(entropy).item()
    
    # 可选：归一化到0-1范围
    if normalize:
        max_rank = min(matrix.shape[0], matrix.shape[1])
        effective_rank = effective_rank / max_rank
    
    return effective_rank


def compute_effective_rank_batch(
    matrices: torch.Tensor,
    eps: float = 1e-10,
    normalize: bool = True,
) -> List[float]:
    """
    批量计算多个矩阵的有效秩
    
    Args:
        matrices: 输入矩阵batch，形状为 [B, M, N]
        eps: 数值稳定性的小常数
        normalize: 是否归一化有效秩
    
    Returns:
        effective_ranks: 每个矩阵的有效秩列表
    """
    if matrices.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, M, N], got {matrices.dim()}D tensor")
    
    ranks = []
    for i in range(matrices.shape[0]):
        rank = compute_effective_rank(matrices[i], eps=eps, normalize=normalize)
        ranks.append(rank)
    
    return ranks


def compute_layer_effective_rank(
    attention_weights: torch.Tensor,
    aggregation: str = "mean",
    eps: float = 1e-10,
    normalize: bool = True,
) -> Dict[str, Union[float, List[float]]]:
    """
    计算单层Attention矩阵的有效秩
    
    支持多种聚合方式处理多头注意力
    
    Args:
        attention_weights: Attention权重，形状为 [batch, num_heads, seq_len, seq_len]
        aggregation: 聚合方式
            - "mean": 先平均所有head再计算rank
            - "per_head": 计算每个head的rank
            - "median": 取所有head rank的中位数
            - "max": 取所有head rank的最大值
            - "min": 取所有head rank的最小值
        eps: 数值稳定性的小常数
        normalize: 是否归一化有效秩
    
    Returns:
        结果字典，包含:
            - "rank": 聚合后的有效秩值
            - "per_head_ranks": 每个head的有效秩（如果计算了的话）
    """
    # 处理输入维度
    if attention_weights.dim() == 3:
        # [num_heads, seq_len, seq_len] -> [1, num_heads, seq_len, seq_len]
        attention_weights = attention_weights.unsqueeze(0)
    
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    result = {"rank": 0.0, "per_head_ranks": None}
    
    if aggregation == "mean":
        # 平均所有batch和head
        avg_attn = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
        result["rank"] = compute_effective_rank(avg_attn, eps=eps, normalize=normalize)
    
    elif aggregation == "per_head":
        # 计算每个head的rank
        per_head_ranks = []
        for b in range(batch_size):
            for h in range(num_heads):
                rank = compute_effective_rank(
                    attention_weights[b, h], eps=eps, normalize=normalize
                )
                per_head_ranks.append(rank)
        result["per_head_ranks"] = per_head_ranks
        result["rank"] = np.mean(per_head_ranks)
    
    elif aggregation in ["median", "max", "min"]:
        # 先计算每个head的rank
        per_head_ranks = []
        for b in range(batch_size):
            for h in range(num_heads):
                rank = compute_effective_rank(
                    attention_weights[b, h], eps=eps, normalize=normalize
                )
                per_head_ranks.append(rank)
        
        result["per_head_ranks"] = per_head_ranks
        
        if aggregation == "median":
            result["rank"] = float(np.median(per_head_ranks))
        elif aggregation == "max":
            result["rank"] = max(per_head_ranks)
        elif aggregation == "min":
            result["rank"] = min(per_head_ranks)
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return result


def analyze_attention_ranks(
    attentions: Tuple[torch.Tensor, ...],
    aggregation: str = "mean",
    eps: float = 1e-10,
    normalize: bool = True,
    show_progress: bool = True,
) -> Dict[str, Union[List[float], Dict]]:
    """
    分析模型所有层的Attention矩阵有效秩
    
    这是分析低秩性的主要入口函数
    
    Args:
        attentions: 模型输出的attention weights
            tuple of tensors, 每个tensor形状为 [batch, num_heads, seq_len, seq_len]
        aggregation: 聚合方式 ("mean", "per_head", "median", "max", "min")
        eps: 数值稳定性的小常数
        normalize: 是否归一化有效秩
        show_progress: 是否显示进度条
    
    Returns:
        分析结果字典:
            - "layer_ranks": List[float], 每层的有效秩
            - "per_head_ranks": Dict[int, List[float]], 每层每个head的有效秩（如果计算了）
            - "statistics": Dict, 统计信息
                - "mean": 平均有效秩
                - "std": 标准差
                - "min": 最小值
                - "max": 最大值
                - "min_layer": 最小值所在层
                - "max_layer": 最大值所在层
    
    Example:
        >>> outputs = model(input_ids, output_attentions=True)
        >>> analysis = analyze_attention_ranks(outputs.attentions)
        >>> print(f"Layer ranks: {analysis['layer_ranks']}")
    """
    num_layers = len(attentions)
    layer_ranks = []
    per_head_ranks = {}
    
    iterator = range(num_layers)
    if show_progress:
        iterator = tqdm(iterator, desc="Analyzing attention ranks")
    
    for layer_idx in iterator:
        attn = attentions[layer_idx]
        
        # 计算该层的有效秩
        layer_result = compute_layer_effective_rank(
            attn,
            aggregation=aggregation,
            eps=eps,
            normalize=normalize,
        )
        
        layer_ranks.append(layer_result["rank"])
        
        if layer_result["per_head_ranks"] is not None:
            per_head_ranks[layer_idx] = layer_result["per_head_ranks"]
    
    # 计算统计信息
    ranks_array = np.array(layer_ranks)
    statistics = {
        "mean": float(np.mean(ranks_array)),
        "std": float(np.std(ranks_array)),
        "min": float(np.min(ranks_array)),
        "max": float(np.max(ranks_array)),
        "min_layer": int(np.argmin(ranks_array)),
        "max_layer": int(np.argmax(ranks_array)),
    }
    
    return {
        "layer_ranks": layer_ranks,
        "per_head_ranks": per_head_ranks if per_head_ranks else None,
        "statistics": statistics,
    }


def analyze_rank_distribution(
    layer_ranks: List[float],
    num_bins: int = 10,
) -> Dict[str, Union[np.ndarray, List[int]]]:
    """
    分析有效秩的分布特征
    
    Args:
        layer_ranks: 每层的有效秩列表
        num_bins: 直方图的bin数量
    
    Returns:
        分布分析结果:
            - "histogram": 直方图计数
            - "bin_edges": bin的边界
            - "low_rank_layers": 低秩层的索引
            - "high_rank_layers": 高秩层的索引
    """
    ranks_array = np.array(layer_ranks)
    
    # 计算直方图
    histogram, bin_edges = np.histogram(ranks_array, bins=num_bins)
    
    # 识别低秩和高秩层（使用四分位数）
    q1 = np.percentile(ranks_array, 25)
    q3 = np.percentile(ranks_array, 75)
    
    low_rank_layers = [i for i, r in enumerate(layer_ranks) if r < q1]
    high_rank_layers = [i for i, r in enumerate(layer_ranks) if r > q3]
    
    return {
        "histogram": histogram.tolist(),
        "bin_edges": bin_edges.tolist(),
        "low_rank_layers": low_rank_layers,
        "high_rank_layers": high_rank_layers,
        "q1": float(q1),
        "q3": float(q3),
    }
