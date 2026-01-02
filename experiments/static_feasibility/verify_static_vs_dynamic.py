
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
# We need to ensure we can control SnapKV via global variables
try:
    from RankKV_Snap.snapkv.monkeypatch import replace_gptneox
    import RankKV_Snap.snapkv.snapkv_utils as snapkv_utils
except ImportError:
    # Need to add RankKV_Snap to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../RankKV_Snap')))
    from snapkv.monkeypatch import replace_gptneox
    import snapkv.snapkv_utils as snapkv_utils

def measure_ppl_and_time(model, input_ids, mode_name="Unknown"):
    torch.cuda.synchronize()
    t_start = time.time()
    
    seq_len = input_ids.shape[1]
    split_idx = int(seq_len * 0.75) # Use last 25% for PPL
    
    # 1. Prefill (where dynamic rank analysis would happen + KV cache creation)
    context_ids = input_ids[:, :split_idx]
    target_ids = input_ids[:, split_idx:]
    
    with torch.no_grad():
        outputs = model(context_ids, use_cache=True)
    
    past_key_values = outputs.past_key_values
    
    # 2. PPL Calculation
    past_length = split_idx
    position_ids = torch.arange(past_length, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    
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
    replace_gptneox("snapkv") # Enable SnapKV monkeypatch
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    # Patch layer_idx if missing
    for i, layer in enumerate(model.gpt_neox.layers):
        if hasattr(layer, "attention"): layer.attention.layer_idx = i
            
    model.eval()
    
    # Load 10 samples from WikiText
    print("Loading test samples...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_texts = [t for t in ds["text"] if len(t) > 3000][:10]
    
    # Load Static Profile
    profile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/static_verification/static_profile_wikitext.json"))
    if not os.path.exists(profile_path):
        print(f"Error: Static profile not found at {profile_path}. Run verify_consistency.py first.")
        return
        
    with open(profile_path, 'r') as f:
        static_ranks = json.load(f)
    print(f"Loaded static profile with {len(static_ranks)} layers.")
    
    analyzer = RankKVAnalyzer(model)
    
    results = []
    
    TOTAL_BUDGET = 256 * 32 # Avg 256 per layer
    
    for i, text in enumerate(test_texts):
        print(f"\nProcessing sample {i+1}/{len(test_texts)}...")
        input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :2048].to(device)
        
        # --- Mode 1: Static RankKV (Proposed) ---
        # No analysis time overhead during inference (done offline)
        snapkv_utils.DISABLE_COMPRESSION = False
        static_budgets = allocate_budget_proportional(static_ranks, TOTAL_BUDGET, min_budget=32)
        snapkv_utils.LAYER_BUDGETS = {acc: b for acc, b in enumerate(static_budgets)}
        
        torch.cuda.empty_cache()
        ppl_static, time_static = measure_ppl_and_time(model, input_ids)
        
        # --- Mode 2: Dynamic RankKV (Oracle) ---
        # Analyze first!
        t_analyze_start = time.time()
        # Note: We need to forward pass to get ranks. This is an EXTRA forward pass.
        # In a real "online" implementation, we might hook into the first prefill, 
        # but that's complex. Here we simulate the overhead of SVD analysis.
        
        # To be fair to "Dynamic", we assume an efficient implementation where
        # analysis happens during the prefill. But SVD is still extra computation.
        # We will measure the SVD time explicitly using our analyzer.
        rank_res = analyzer.analyze(input_ids, show_progress=False) 
        dynamic_ranks = rank_res.layer_ranks
        t_analyze_len = (time.time() - t_analyze_start) * 1000 # ms
        
        dynamic_budgets = allocate_budget_proportional(dynamic_ranks, TOTAL_BUDGET, min_budget=32)
        snapkv_utils.LAYER_BUDGETS = {acc: b for acc, b in enumerate(dynamic_budgets)}
        
        torch.cuda.empty_cache()
        ppl_dynamic, time_dynamic_inference = measure_ppl_and_time(model, input_ids)
        
        # Total dynamic time = Analysis Overhead + Inference Time
        # (Though strictly analysis overlaps with prefill, SVD is the dominant cost)
        total_time_dynamic = time_dynamic_inference + t_analyze_len 
        
        results.append({
            "sample_id": i,
            "ppl_static": ppl_static,
            "time_static": time_static,
            "ppl_dynamic": ppl_dynamic,
            "time_dynamic": total_time_dynamic,
            "analysis_overhead_ms": t_analyze_len
        })
        
        print(f"  Static PPL: {ppl_static:.2f}, Time: {time_static:.0f}ms")
        print(f"  Dynamic PPL: {ppl_dynamic:.2f}, Time: {total_time_dynamic:.0f}ms (Overhead: {t_analyze_len:.0f}ms)")

    # Summary
    df = pd.DataFrame(results)
    print("\n=== Final Comparison ===")
    print(df.mean())
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/static_verification/static_vs_dynamic_results.csv"))
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Calculate gap
    avg_ppl_static = df['ppl_static'].mean()
    avg_ppl_dynamic = df['ppl_dynamic'].mean()
    ppl_gap = avg_ppl_static - avg_ppl_dynamic
    
    print(f"PPL Gap (Static - Dynamic): {ppl_gap:.4f}")
    if abs(ppl_gap) < 0.1:
        print("SUCCESS: Static profile performs comparably to Dynamic oracle!")
    else:
        print("WARNING: Significant performance drop with static profile.")

if __name__ == "__main__":
    main()
