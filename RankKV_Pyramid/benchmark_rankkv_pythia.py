
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from pyramidkv.monkeypatch import replace_gptneox
import pyramidkv.pyramidkv_utils as pyramidkv_utils
from rankkv.rank_analysis import compute_effective_rank
from rankkv.budget_allocation import allocate_budget_proportional

def main():
    # Configuration
    model_name = "EleutherAI/pythia-2.8b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load Model (with PyramidKV patch)
    print("Loading model and patching with PyramidKV...")
    replace_gptneox("pyramidkv")
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Patch: Inject layer_idx to attention modules (required for PyramidKV)
    print("Injecting layer_idx to attention modules...")
    if hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    else:
        # Fallback for other architectures if needed
        layers = model.model.layers if hasattr(model, "model") else []
        
    for i, layer in enumerate(layers):
        if hasattr(layer, "attention"):
            layer.attention.layer_idx = i
            # Also set for kv_cluster if it needs it later
    
    # Prepare Input Data (Long Context)
    # Using repeated text to simulate long context
    text = "The quick brown fox jumps over the lazy dog. " * 200 
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]
    print(f"Input Sequence Length: {seq_len}")
    
    if seq_len < 100:
        print("Sequence too short for compression analysis. Extending...")
        inputs = tokenizer(text * 5, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]
        print(f"New Sequence Length: {seq_len}")

    # --- Step 1: Analyze Ranks (Full Attention) ---
    print("\n--- Step 1: Rank Analysis (Full Attention) ---")
    pyramidkv_utils.DISABLE_COMPRESSION = True
    
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_attentions=True, use_cache=True)
    
    ranks = []
    # outputs.attentions is tuple of (batch, heads, seq, seq)
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # layer_attn: [batch, heads, seq, seq]
        # Average over heads and batch
        attn_avg = layer_attn[0].mean(dim=0).float() # [seq, seq]
        rank = compute_effective_rank(attn_avg)
        ranks.append(rank)
        print(f"Layer {layer_idx} Effective Rank: {rank:.4f}")
        
    # Plot Ranks
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, marker='o')
        plt.title("Layer Effective Ranks (Pythia-2.8B)")
        plt.xlabel("Layer Index")
        plt.ylabel("Effective Rank")
        plt.grid(True)
        plt.savefig("layer_ranks.png")
        print("Rank plot saved to layer_ranks.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # --- Step 2: Allocate Budgets ---
    print("\n--- Step 2: Budget Allocation ---")
    # Target compression: e.g., Keep 10% of tokens (approx) or fixed number
    # Let's say we want to keep 128 tokens per layer on average (Total 32 * 128 = 4096 budget units? No, total budget)
    # If seq_len is ~2000. 10% is 200.
    # Let's set a small budget to see the effect: Total Budget = 1000 tokens (across 32 layers)
    # Avg per layer = 31.
    total_budget = 2048 # Total tokens to keep across all layers
    budgets = allocate_budget_proportional(ranks, total_budget, min_budget=16)
    
    print(f"Total Budget: {total_budget}")
    print(f"Allocated Budgets: {budgets}")
    print(f"Sum of Budgets: {sum(budgets)}")
    
    # --- Step 3: Run RankKV ---
    print("\n--- Step 3: Running RankKV ---")
    pyramidkv_utils.DISABLE_COMPRESSION = False
    pyramidkv_utils.LAYER_BUDGETS.clear()
    for i, b in enumerate(budgets):
        pyramidkv_utils.LAYER_BUDGETS[i] = b
        
    # Clear previous outputs to save memory
    del outputs
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        # Running model again with RankKV budgets
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        outputs_rankkv = model(inputs.input_ids, labels=inputs.input_ids, use_cache=True)
        end_time.record()
        torch.cuda.synchronize()
        
        loss_rankkv = outputs_rankkv.loss.item()
        ppl_rankkv = torch.exp(outputs_rankkv.loss).item()
        time_rankkv = start_time.elapsed_time(end_time)
        
    print(f"RankKV PPL: {ppl_rankkv:.4f}")
    print(f"RankKV Time: {time_rankkv:.2f} ms")
    
    # --- Step 4: Run Uniform Budget (Baseline) ---
    print("\n--- Step 4: Running Uniform Budget ---")
    pyramidkv_utils.LAYER_BUDGETS.clear()
    
    avg_budget = total_budget // len(budgets)
    uniform_budgets = {i: avg_budget for i in range(len(budgets))}
    pyramidkv_utils.LAYER_BUDGETS = uniform_budgets
    
    del outputs_rankkv
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        outputs_uniform = model(inputs.input_ids, labels=inputs.input_ids, use_cache=True)
        loss_uniform = outputs_uniform.loss.item()
        ppl_uniform = torch.exp(outputs_uniform.loss).item()
        
    print(f"Uniform PPL: {ppl_uniform:.4f}")
    
    # --- Summary ---
    print("\n=== Experiment Results ===")
    print(f"RankKV PPL: {ppl_rankkv:.4f}")
    print(f"Uniform PPL: {ppl_uniform:.4f}")
    if ppl_rankkv < ppl_uniform:
        print("RESULT: RankKV outperformed Uniform Budget!")
    else:
        print("RESULT: Uniform Budget outperformed RankKV (check budget distribution)")

if __name__ == "__main__":
    main()
