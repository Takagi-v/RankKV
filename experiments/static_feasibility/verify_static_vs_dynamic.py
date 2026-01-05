import os
import sys
import torch
import time
import json
import math
import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import RankKV modules
from rankkv.pipeline import RankKVAnalyzer
from rankkv.budget_allocation import allocate_budget_proportional

# Import SnapKV Integration
try:
    from RankKV_Snap.snapkv.monkeypatch import replace_gptneox
    import RankKV_Snap.snapkv.snapkv_utils as snapkv_utils
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../RankKV_Snap')))
    from snapkv.monkeypatch import replace_gptneox
    import snapkv.snapkv_utils as snapkv_utils


def load_test_texts(min_length=1500, num_samples=10):
    """
    Load test texts from WikiText dataset.
    
    Args:
        min_length: minimum character length for texts
        num_samples: number of samples to load
    
    Returns:
        list of text strings
    """
    print("Loading test samples from WikiText...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # 尝试不同的长度阈值来获取足够的样本
    thresholds = [min_length, 1000, 500, 200]
    
    for threshold in thresholds:
        test_texts = [t for t in ds["text"] if len(t) > threshold]
        print(f"Found {len(test_texts)} texts with length > {threshold}")
        
        if len(test_texts) >= num_samples:
            return test_texts[:num_samples]
    
    # 如果还是不够，返回所有非空文本
    test_texts = [t for t in ds["text"] if len(t.strip()) > 0]
    print(f"Warning: Using all {len(test_texts)} non-empty texts")
    
    return test_texts[:num_samples] if len(test_texts) >= num_samples else test_texts


def measure_ppl_and_time(model, input_ids, mode_name="Unknown"):
    """
    Measure perplexity and inference time.
    
    Args:
        model: HuggingFace model
        input_ids: tokenized input tensor
        mode_name: name for logging
    
    Returns:
        (perplexity, time_in_ms)
    """
    torch.cuda.synchronize()
    t_start = time.time()
    
    seq_len = input_ids.shape[1]
    split_idx = int(seq_len * 0.75)  # Use last 25% for PPL
    
    # 确保 split_idx 合理
    if split_idx < 1:
        split_idx = 1
    if split_idx >= seq_len:
        split_idx = seq_len - 1
    
    # 1. Prefill
    context_ids = input_ids[:, :split_idx]
    target_ids = input_ids[:, split_idx:]
    
    if target_ids.shape[1] == 0:
        print(f"Warning: target_ids is empty for {mode_name}")
        return float('inf'), 0.0
    
    with torch.no_grad():
        outputs = model(context_ids, use_cache=True)
    
    past_key_values = outputs.past_key_values
    
    # 2. PPL Calculation
    past_length = split_idx
    position_ids = torch.arange(
        past_length, seq_len, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)
    
    with torch.no_grad():
        outputs_ppl = model(
            target_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
            labels=target_ids
        )
        ppl_loss = outputs_ppl.loss.item()
    
    torch.cuda.synchronize()
    t_end = time.time()
    
    ppl = math.exp(ppl_loss) if ppl_loss < 20 else 1e9
    total_time_ms = (t_end - t_start) * 1000
    
    return ppl, total_time_ms


def main():
    model_name = "EleutherAI/pythia-2.8b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    replace_gptneox("snapkv")  # Enable SnapKV monkeypatch
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Patch layer_idx if missing
    for i, layer in enumerate(model.gpt_neox.layers):
        if hasattr(layer, "attention"):
            layer.attention.layer_idx = i
    
    model.eval()
    
    # Load test samples
    test_texts = load_test_texts(min_length=1500, num_samples=10)
    
    if len(test_texts) == 0:
        print("Error: No test texts loaded! Cannot proceed.")
        return
    
    print(f"Loaded {len(test_texts)} test samples")
    
    # Load Static Profile
    profile_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 
                     "../../results/static_verification/static_profile_wikitext.json")
    )
    
    if not os.path.exists(profile_path):
        print(f"Error: Static profile not found at {profile_path}")
        print("Please run verify_consistency.py first to generate the static profile.")
        return
    
    with open(profile_path, 'r') as f:
        static_ranks = json.load(f)
    print(f"Loaded static profile with {len(static_ranks)} layers.")
    
    analyzer = RankKVAnalyzer(model)
    
    results = []
    
    TOTAL_BUDGET = 256 * 32  # Avg 256 per layer
    
    for i, text in enumerate(test_texts):
        print(f"\nProcessing sample {i+1}/{len(test_texts)}...")
        
        # Tokenize with reasonable max length
        input_ids = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,
            max_length=2048
        ).input_ids.to(device)
        
        # 检查序列长度是否足够
        if input_ids.shape[1] < 10:
            print(f"  Skipping sample {i+1}: sequence too short ({input_ids.shape[1]} tokens)")
            continue
        
        try:
            # --- Mode 1: Static RankKV (Proposed) ---
            snapkv_utils.DISABLE_COMPRESSION = False
            static_budgets = allocate_budget_proportional(static_ranks, TOTAL_BUDGET, min_budget=32)
            snapkv_utils.LAYER_BUDGETS = {idx: b for idx, b in enumerate(static_budgets)}
            
            torch.cuda.empty_cache()
            ppl_static, time_static = measure_ppl_and_time(model, input_ids, mode_name="Static")
            
            # --- Mode 2: Dynamic RankKV (Oracle) ---
            t_analyze_start = time.time()
            rank_res = analyzer.analyze(input_ids, show_progress=False)
            dynamic_ranks = rank_res.layer_ranks
            t_analyze_len = (time.time() - t_analyze_start) * 1000  # ms
            
            dynamic_budgets = allocate_budget_proportional(dynamic_ranks, TOTAL_BUDGET, min_budget=32)
            snapkv_utils.LAYER_BUDGETS = {idx: b for idx, b in enumerate(dynamic_budgets)}
            
            torch.cuda.empty_cache()
            ppl_dynamic, time_dynamic_inference = measure_ppl_and_time(model, input_ids, mode_name="Dynamic")
            
            # Total dynamic time = Analysis Overhead + Inference Time
            total_time_dynamic = time_dynamic_inference + t_analyze_len
            
            results.append({
                "sample_id": i,
                "seq_len": input_ids.shape[1],
                "ppl_static": ppl_static,
                "time_static": time_static,
                "ppl_dynamic": ppl_dynamic,
                "time_dynamic": total_time_dynamic,
                "analysis_overhead_ms": t_analyze_len
            })
            
            print(f"  Static PPL: {ppl_static:.2f}, Time: {time_static:.0f}ms")
            print(f"  Dynamic PPL: {ppl_dynamic:.2f}, Time: {total_time_dynamic:.0f}ms (Overhead: {t_analyze_len:.0f}ms)")
            
        except Exception as e:
            print(f"  Error processing sample {i+1}: {e}")
            continue
    
    # Check if we have any results
    if len(results) == 0:
        print("\nError: No samples were successfully processed!")
        print("Please check your data and model configuration.")
        return
    
    # Summary
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 50)
    print("=== Final Comparison ===")
    print("=" * 50)
    print(f"\nProcessed {len(results)} samples successfully")
    print("\nMean values:")
    print(df[['ppl_static', 'time_static', 'ppl_dynamic', 'time_dynamic', 'analysis_overhead_ms']].mean())
    
    # Save results
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../results/static_verification")
    )
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "static_vs_dynamic_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Calculate gap
    avg_ppl_static = df['ppl_static'].mean()
    avg_ppl_dynamic = df['ppl_dynamic'].mean()
    ppl_gap = avg_ppl_static - avg_ppl_dynamic
    
    avg_time_static = df['time_static'].mean()
    avg_time_dynamic = df['time_dynamic'].mean()
    time_speedup = avg_time_dynamic / avg_time_static if avg_time_static > 0 else 0
    
    print(f"\n--- Summary ---")
    print(f"Average Static PPL:  {avg_ppl_static:.4f}")
    print(f"Average Dynamic PPL: {avg_ppl_dynamic:.4f}")
    print(f"PPL Gap (Static - Dynamic): {ppl_gap:.4f}")
    print(f"\nAverage Static Time:  {avg_time_static:.2f}ms")
    print(f"Average Dynamic Time: {avg_time_dynamic:.2f}ms")
    print(f"Static Speedup: {time_speedup:.2f}x")
    
    if abs(ppl_gap) < 0.5:
        print("\n✓ SUCCESS: Static profile performs comparably to Dynamic oracle!")
    elif ppl_gap > 0:
        print("\n⚠ WARNING: Static profile has higher PPL (worse) than Dynamic.")
    else:
        print("\n✓ Static profile actually outperforms Dynamic oracle!")
    
    # Save summary as JSON
    summary = {
        "num_samples": len(results),
        "avg_ppl_static": float(avg_ppl_static),
        "avg_ppl_dynamic": float(avg_ppl_dynamic),
        "ppl_gap": float(ppl_gap),
        "avg_time_static_ms": float(avg_time_static),
        "avg_time_dynamic_ms": float(avg_time_dynamic),
        "speedup": float(time_speedup)
    }
    
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()