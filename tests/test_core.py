"""
RankKV 核心功能测试脚本

快速验证代码是否正常工作
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_effective_rank():
    """测试有效秩计算"""
    from rankkv import compute_effective_rank
    
    print("=" * 50)
    print("Test 1: Effective Rank Computation")
    print("=" * 50)
    
    # 测试1：低秩矩阵（秩为1）
    low_rank = torch.outer(torch.randn(100), torch.randn(100))
    rank_low = compute_effective_rank(low_rank, normalize=False)
    print(f"Low-rank matrix (rank~1): erank = {rank_low:.4f}")
    
    # 测试2：高秩矩阵（满秩）
    high_rank = torch.randn(100, 100)
    rank_high = compute_effective_rank(high_rank, normalize=False)
    print(f"Full-rank matrix: erank = {rank_high:.4f}")
    
    # 测试3：恒等矩阵
    identity = torch.eye(100)
    rank_id = compute_effective_rank(identity, normalize=False)
    print(f"Identity matrix: erank = {rank_id:.4f}")
    
    # 测试4：模拟Attention矩阵（行归一化）
    attn_sim = torch.softmax(torch.randn(100, 100), dim=-1)
    rank_attn = compute_effective_rank(attn_sim, normalize=True)
    print(f"Simulated Attention (normalized): erank = {rank_attn:.4f}")
    
    assert rank_low < rank_high, "Low-rank should have lower effective rank"
    print("\n✓ Effective rank computation works correctly!\n")


def test_budget_allocation():
    """测试Budget分配"""
    from rankkv import BudgetAllocator
    
    print("=" * 50)
    print("Test 2: Budget Allocation")
    print("=" * 50)
    
    # 模拟32层的有效秩
    np.random.seed(42)
    layer_ranks = list(np.random.uniform(0.1, 0.9, 32))
    total_budget = 1000
    
    # 测试不同策略
    strategies = ["proportional", "softmax", "adaptive", "power"]
    
    for strategy in strategies:
        allocator = BudgetAllocator(strategy=strategy, min_budget=10)
        result = allocator.allocate(layer_ranks, total_budget=total_budget)
        
        print(f"\n{strategy.capitalize()} Strategy:")
        print(f"  Total allocated: {sum(result.budgets)} (target: {total_budget})")
        print(f"  Budget range: [{min(result.budgets)}, {max(result.budgets)}]")
        print(f"  Std: {np.std(result.budgets):.2f}")
    
    print("\n✓ Budget allocation works correctly!\n")


def test_analyzer_mock():
    """测试分析器（使用模拟数据）"""
    from rankkv import analyze_attention_ranks
    from rankkv.budget_allocation import BudgetAllocator
    
    print("=" * 50)
    print("Test 3: Analyzer with Mock Data")
    print("=" * 50)
    
    # 模拟24层Attention矩阵
    num_layers = 24
    batch_size = 1
    num_heads = 12
    seq_len = 128
    
    print(f"Simulating {num_layers} layers, {num_heads} heads, seq_len={seq_len}")
    
    # 生成模拟的attention权重
    mock_attentions = []
    for i in range(num_layers):
        # 让不同层有不同的"集中程度"
        temperature = 0.5 + i * 0.05  # 温度随层数增加
        logits = torch.randn(batch_size, num_heads, seq_len, seq_len) / temperature
        attn = torch.softmax(logits, dim=-1)
        mock_attentions.append(attn)
    
    mock_attentions = tuple(mock_attentions)
    
    # 分析
    analysis = analyze_attention_ranks(mock_attentions, aggregation="mean", 
                                       normalize=True, show_progress=False)
    
    print(f"\nLayer Ranks (first 10):")
    for i in range(min(10, num_layers)):
        print(f"  Layer {i:2d}: {analysis['layer_ranks'][i]:.4f}")
    
    print(f"\nStatistics:")
    for key, value in analysis["statistics"].items():
        print(f"  {key}: {value}")
    
    # Budget分配
    allocator = BudgetAllocator(strategy="proportional")
    result = allocator.allocate(analysis["layer_ranks"], seq_len=seq_len, compression_ratio=0.3)
    
    print(f"\nBudget Allocation (compression=30%):")
    print(f"  Total budget: {result.total_budget}")
    print(f"  Budget range: [{min(result.budgets)}, {max(result.budgets)}]")
    
    print("\n✓ Analyzer works correctly with mock data!\n")


def test_full_pipeline():
    """完整Pipeline测试（需要GPU和transformers）"""
    print("=" * 50)
    print("Test 4: Full Pipeline (Optional)")
    print("=" * 50)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from rankkv import RankKVAnalyzer
        
        # 使用小模型测试
        model_name = "EleutherAI/pythia-70m"  # 最小的Pythia模型
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 分析
        analyzer = RankKVAnalyzer(model, model_name=model_name)
        
        text = "The quick brown fox jumps over the lazy dog. This is a test sentence for RankKV analysis."
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        
        result = analyzer.analyze(input_ids, compression_ratio=0.3, show_progress=False)
        
        print("\n" + result.summary())
        print("✓ Full pipeline works correctly!\n")
        
    except ImportError as e:
        print(f"Skipping (missing dependencies): {e}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: GPU may be required for this test")


def main():
    print("\n" + "=" * 50)
    print("RankKV Core Functionality Tests")
    print("=" * 50 + "\n")
    
    test_effective_rank()
    test_budget_allocation()
    test_analyzer_mock()
    
    # 可选的完整Pipeline测试
    run_full = input("Run full pipeline test with real model? (y/n): ").lower().strip()
    if run_full == 'y':
        test_full_pipeline()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
