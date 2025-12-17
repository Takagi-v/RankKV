"""
低秩性分析实验

目的：
1. 验证不同层的Attention矩阵确实具有不同的有效秩
2. 分析有效秩的分布特征
3. 可视化结果

使用方法：
    python experiments/rank_analysis_exp.py --model EleutherAI/pythia-2.8b --dataset wikitext
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rankkv import compute_effective_rank, analyze_attention_ranks
from rankkv.budget_allocation import BudgetAllocator


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """加载模型和tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_sample_texts(dataset_name: str, num_samples: int = 10, max_length: int = 512):
    """从数据集加载样本文本"""
    from datasets import load_dataset
    
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = dataset["text"]
    elif dataset_name == "pg19":
        dataset = load_dataset("pg19", split="test")
        texts = dataset["text"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 过滤空文本和过短文本
    texts = [t for t in texts if len(t.strip()) > 100]
    
    # 采样
    if len(texts) > num_samples:
        indices = np.random.choice(len(texts), num_samples, replace=False)
        texts = [texts[i] for i in indices]
    
    # 截断到max_length个字符（大约）
    texts = [t[:max_length * 4] for t in texts]  # 粗略估计每个token约4字符
    
    return texts


@torch.no_grad()
def analyze_single_text(model, tokenizer, text: str, device: str = "cuda"):
    """分析单个文本的有效秩"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    
    outputs = model(input_ids, output_attentions=True, use_cache=False)
    attentions = outputs.attentions
    
    analysis = analyze_attention_ranks(attentions, aggregation="mean", 
                                       normalize=True, show_progress=False)
    
    return {
        "layer_ranks": analysis["layer_ranks"],
        "statistics": analysis["statistics"],
        "seq_len": input_ids.shape[1],
    }


def run_analysis(
    model_name: str,
    dataset_name: str,
    num_samples: int = 10,
    output_dir: str = "results",
    device: str = "auto",
):
    """运行分析实验"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    device = next(model.parameters()).device
    
    # 加载文本
    texts = load_sample_texts(dataset_name, num_samples)
    
    # 分析每个文本
    all_results = []
    all_layer_ranks = []
    
    for i, text in enumerate(tqdm(texts, desc="Analyzing texts")):
        result = analyze_single_text(model, tokenizer, text, device)
        all_results.append(result)
        all_layer_ranks.append(result["layer_ranks"])
    
    # 聚合结果
    all_layer_ranks = np.array(all_layer_ranks)  # [num_samples, num_layers]
    mean_ranks = all_layer_ranks.mean(axis=0)
    std_ranks = all_layer_ranks.std(axis=0)
    
    num_layers = len(mean_ranks)
    
    # 统计信息
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "num_layers": num_layers,
        "mean_ranks": mean_ranks.tolist(),
        "std_ranks": std_ranks.tolist(),
        "overall_mean": float(mean_ranks.mean()),
        "overall_std": float(mean_ranks.std()),
        "min_layer": int(np.argmin(mean_ranks)),
        "max_layer": int(np.argmax(mean_ranks)),
        "min_rank": float(mean_ranks.min()),
        "max_rank": float(mean_ranks.max()),
    }
    
    # 保存结果
    with open(os.path.join(output_dir, "rank_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {num_samples}")
    print(f"Layers: {num_layers}")
    print(f"Overall Mean Rank: {summary['overall_mean']:.4f}")
    print(f"Rank Range: [{summary['min_rank']:.4f}, {summary['max_rank']:.4f}]")
    print(f"Min Rank Layer: {summary['min_layer']}")
    print(f"Max Rank Layer: {summary['max_layer']}")
    print("=" * 60)
    
    # 绘制图表
    plot_rank_analysis(mean_ranks, std_ranks, summary, output_dir)
    
    return summary


def plot_rank_analysis(mean_ranks, std_ranks, summary, output_dir):
    """绘制有效秩分析图表"""
    num_layers = len(mean_ranks)
    
    # 设置风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 柱状图：每层有效秩
    ax1 = axes[0, 0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_layers))
    bars = ax1.bar(range(num_layers), mean_ranks, yerr=std_ranks, 
                   color=colors, capsize=2, alpha=0.8)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Effective Rank (Normalized)")
    ax1.set_title("Effective Rank per Layer")
    ax1.axhline(y=mean_ranks.mean(), color='r', linestyle='--', label='Mean')
    ax1.legend()
    
    # 2. 折线图：趋势
    ax2 = axes[0, 1]
    ax2.plot(range(num_layers), mean_ranks, 'b-o', markersize=4, label='Mean Rank')
    ax2.fill_between(range(num_layers), 
                     mean_ranks - std_ranks, 
                     mean_ranks + std_ranks, 
                     alpha=0.3)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Effective Rank")
    ax2.set_title("Effective Rank Trend")
    ax2.legend()
    
    # 3. 直方图：秩分布
    ax3 = axes[1, 0]
    ax3.hist(mean_ranks, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=mean_ranks.mean(), color='r', linestyle='--', label='Mean')
    ax3.set_xlabel("Effective Rank")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Effective Ranks")
    ax3.legend()
    
    # 4. Budget分配对比
    ax4 = axes[1, 1]
    total_budget = 100  # 假设总budget为100
    
    # 计算不同策略的分配
    allocator_prop = BudgetAllocator(strategy="proportional")
    allocator_soft = BudgetAllocator(strategy="softmax", temperature=0.5)
    
    budgets_prop = allocator_prop.allocate(mean_ranks.tolist(), total_budget=total_budget).budgets
    budgets_soft = allocator_soft.allocate(mean_ranks.tolist(), total_budget=total_budget).budgets
    budgets_uniform = [total_budget // num_layers] * num_layers
    
    x = np.arange(num_layers)
    width = 0.25
    
    ax4.bar(x - width, budgets_uniform, width, label='Uniform', alpha=0.8)
    ax4.bar(x, budgets_prop, width, label='Proportional', alpha=0.8)
    ax4.bar(x + width, budgets_soft, width, label='Softmax', alpha=0.8)
    
    ax4.set_xlabel("Layer Index")
    ax4.set_ylabel("Budget")
    ax4.set_title("Budget Allocation Comparison")
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "rank_analysis.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "figures", "rank_analysis.pdf"))
    print(f"Figures saved to {output_dir}/figures/")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="RankKV - Effective Rank Analysis")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m",
                       help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       choices=["wikitext", "pg19"], help="Dataset name")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to analyze")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 运行分析
    run_analysis(
        model_name=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
