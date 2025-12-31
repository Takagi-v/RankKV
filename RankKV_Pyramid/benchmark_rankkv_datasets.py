
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import time
import math
import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
from pyramidkv.monkeypatch import replace_gptneox
import pyramidkv.pyramidkv_utils as pyramidkv_utils
from rankkv.rank_analysis import compute_effective_rank
from rankkv.budget_allocation import allocate_budget_proportional

def get_data(dataset_name, tokenizer, seq_len=2048):
    print(f"Loading dataset: {dataset_name}...")
    if dataset_name == 'wikitext':
        # WikiText-2
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds['text'])
    elif dataset_name == 'pg19':
        # PG-19 (Stream to avoid downloading huge file)
        try:
            ds = load_dataset("pg19", split="test", streaming=True)
            # Take the first book
            text = next(iter(ds))['text']
        except Exception as e:
            print(f"Error loading PG-19: {e}. Falling back to local dummy text.")
            text = "The quick brown fox jumps over the lazy dog. " * 10000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    tokens = tokenizer(text, return_tensors="pt").input_ids
    if tokens.shape[1] < seq_len:
        print(f"Warning: Text length {tokens.shape[1]} < {seq_len}. Repeating text.")
        tokens = tokens.repeat(1, math.ceil(seq_len / tokens.shape[1]))
    
    return tokens[:, :seq_len]

def measure_metrics(model, input_ids, generate_len=50):
    device = input_ids.device
    seq_len = input_ids.shape[1]
    
    # Split into context (for compression) and target (for PPL measurement)
    # Use 75% for context, 25% for PPL evaluation
    split_idx = int(seq_len * 0.75)
    context_ids = input_ids[:, :split_idx]
    target_ids = input_ids[:, split_idx:]
    
    # 1. TTFT (Time To First Token) - Prefill Context
    torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.no_grad():
        # This will compress the context into KV cache
        outputs = model(context_ids, use_cache=True)
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    ttft = (t1 - t0) * 1000 # ms
    past_key_values = outputs.past_key_values
    
    # 2. PPL Calculation (on Target segment using Compressed History)
    # We evaluate loss token by token or in chunks? 
    # For speed, we can process target_ids in one go if model supports it with past_key_values?
    # GPT-NeoX supports passing past_key_values and input_ids together.
    # It will attend to past + current.
    
    with torch.no_grad():
        # Target needs labels for loss
        # Note: We need to be careful. The model expects input_ids to be the *continuation*.
        outputs_ppl = model(target_ids, past_key_values=past_key_values, use_cache=True, labels=target_ids)
        ppl_loss = outputs_ppl.loss.item()
        ppl = math.exp(ppl_loss) if ppl_loss < 20 else 1e9
        
        # Update past_key_values with target (which might be compressed too if it was long? 
        # But here target is relatively short, usually appended)
        # For the generation test, we should probably start from the END of input_ids.
        # But to be consistent with PPL context, let's use the state AFTER target for generation?
        # Or just use the state after context? 
        # Let's use state after context for generation to measure TPOT on "clean" compressed state.
    
    # 3. TPOT (Time Per Output Token)
    # Generate tokens one by one starting from end of CONTEXT
    next_token = context_ids[:, -1:]
    # Reset past_key_values to the state after context (outputs.past_key_values)
    past_key_values = outputs.past_key_values
    
    torch.cuda.synchronize()
    t2 = time.time()
    
    with torch.no_grad():
        for _ in range(generate_len):
            outputs_gen = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs_gen.past_key_values
            next_token = torch.argmax(outputs_gen.logits[:, -1, :], dim=-1, keepdim=True)
            
    torch.cuda.synchronize()
    t3 = time.time()
    
    gen_time = (t3 - t2) * 1000 # ms
    tpot = gen_time / generate_len
    throughput = generate_len / (t3 - t2) # tokens/sec
    
    # Estimate FLOPs (Very rough approx: 2 * N * P)
    # Params
    params = sum(p.numel() for p in model.parameters())
    # Prefill FLOPs
    prefill_flops = 2 * params * context_ids.shape[1]
    # Gen FLOPs
    gen_flops = 2 * params * generate_len
    
    total_flops = prefill_flops + gen_flops
    avg_gflops = (total_flops / 1e9) / (t3 - t0)
    
    return {
        "ppl": ppl,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "throughput_tps": throughput,
        "avg_gflops": avg_gflops
    }

def main():
    model_name = "EleutherAI/pythia-2.8b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    replace_gptneox("pyramidkv")
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Patch layer_idx
    if hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    else:
        layers = model.model.layers if hasattr(model, "model") else []
    for i, layer in enumerate(layers):
        if hasattr(layer, "attention"):
            layer.attention.layer_idx = i

    datasets = ['wikitext', 'pg19']
    seq_len = 2048
    
    for ds_name in datasets:
        print(f"\n{'='*20} Processing {ds_name} {'='*20}")
        try:
            input_ids = get_data(ds_name, tokenizer, seq_len).to(device)
        except Exception as e:
            print(f"Skipping {ds_name} due to error: {e}")
            continue

        print(f"Input shape: {input_ids.shape}")

        # --- Step 1: Rank Analysis ---
        print("Analyzing Ranks...")
        pyramidkv_utils.DISABLE_COMPRESSION = True
        pyramidkv_utils.LAYER_BUDGETS.clear()
        
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True, use_cache=True)
            
        ranks = []
        for layer_attn in outputs.attentions:
            attn_avg = layer_attn[0].mean(dim=0).float()
            rank = compute_effective_rank(attn_avg)
            ranks.append(rank)
        
        # Free memory
        del outputs
        torch.cuda.empty_cache()
        
        # --- Step 2: Define Budgets ---
        total_budget = 2048 # Keep ~2048 tokens total cache (compressed from 2048 input?)
        # Wait, if input is 2048, and we want to compress, total budget should be LESS than 2048 * 32.
        # PyramidKV usually keeps a small buffer.
        # Let's target a compression ratio, e.g., 20% (retain ~400 tokens per layer avg? or Total?)
        # "Total budget" in allocate_budget_proportional usually means SUM of budgets across layers.
        # If we want 128 tokens per layer avg -> Total = 128 * 32 = 4096.
        # Let's try aggressive compression: Avg 128 tokens/layer.
        
        target_avg_budget = 128
        total_budget_sum = target_avg_budget * len(ranks)
        
        rankkv_budgets = allocate_budget_proportional(ranks, total_budget_sum, min_budget=32)
        uniform_budgets = [target_avg_budget] * len(ranks)
        
        print(f"Avg Budget: {target_avg_budget}")
        print(f"RankKV Budgets: {rankkv_budgets}")
        
        # --- Step 3: Run RankKV ---
        print("\nRunning RankKV...")
        pyramidkv_utils.DISABLE_COMPRESSION = False
        pyramidkv_utils.LAYER_BUDGETS = {i: b for i, b in enumerate(rankkv_budgets)}
        
        metrics_rankkv = measure_metrics(model, input_ids)
        print(f"RankKV Results: {metrics_rankkv}")
        
        # --- Step 4: Run Uniform ---
        print("\nRunning Uniform...")
        pyramidkv_utils.LAYER_BUDGETS = {i: b for i, b in enumerate(uniform_budgets)}
        
        metrics_uniform = measure_metrics(model, input_ids)
        print(f"Uniform Results: {metrics_uniform}")
        
        # --- Comparison ---
        print("\n--- Comparison ---")
        print(f"{'Metric':<15} | {'RankKV':<15} | {'Uniform':<15} | {'Diff':<10}")
        print("-" * 60)
        for k in metrics_rankkv:
            v1 = metrics_rankkv[k]
            v2 = metrics_uniform[k]
            diff = (v1 - v2) / v2 * 100
            print(f"{k:<15} | {v1:<15.4f} | {v2:<15.4f} | {diff:+.2f}%")

if __name__ == "__main__":
    main()
