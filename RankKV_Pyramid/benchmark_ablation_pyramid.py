
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import time
import math
import numpy as np
import random
import argparse
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
from pyramidkv.monkeypatch import replace_gptneox
import pyramidkv.pyramidkv_utils as pyramidkv_utils
from rankkv.budget_allocation import allocate_budget_proportional

# Define paths
RANK_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "rank_analysis.json")

def load_ranks_from_json(json_path):
    print(f"Loading ranks from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['mean_ranks']

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
    split_idx = int(seq_len * 0.75)
    context_ids = input_ids[:, :split_idx]
    target_ids = input_ids[:, split_idx:]
    
    # 1. TTFT (Time To First Token) - Prefill Context
    torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.no_grad():
        outputs = model(context_ids, use_cache=True)
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    ttft = (t1 - t0) * 1000 # ms
    past_key_values = outputs.past_key_values
    
    # 2. PPL Calculation
    past_length = split_idx
    position_ids = torch.arange(past_length, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        outputs_ppl = model(
            target_ids, 
            past_key_values=past_key_values, 
            position_ids=position_ids,
            use_cache=True, 
            labels=target_ids
        )
        ppl_loss = outputs_ppl.loss.item()
        ppl = math.exp(ppl_loss) if ppl_loss < 20 else 1e9
        
    # 3. TPOT (Time Per Output Token)
    next_token = context_ids[:, -1:]
    past_key_values = outputs.past_key_values
    
    current_pos = split_idx
    
    torch.cuda.synchronize()
    t2 = time.time()
    
    with torch.no_grad():
        for _ in range(generate_len):
            pos_id = torch.tensor([[current_pos]], dtype=torch.long, device=device)
            outputs_gen = model(next_token, past_key_values=past_key_values, position_ids=pos_id, use_cache=True)
            past_key_values = outputs_gen.past_key_values
            next_token = torch.argmax(outputs_gen.logits[:, -1, :], dim=-1, keepdim=True)
            current_pos += 1
            
    torch.cuda.synchronize()
    t3 = time.time()
    
    gen_time = (t3 - t2) * 1000 # ms
    tpot = gen_time / generate_len
    throughput = generate_len / (t3 - t2) # tokens/sec
    
    # Estimate FLOPs
    params = sum(p.numel() for p in model.parameters())
    total_flops = (2 * params * context_ids.shape[1]) + (2 * params * generate_len)
    avg_gflops = (total_flops / 1e9) / (t3 - t0)
    
    # Memory usage
    peak_mem = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    # Estimate KV Cache Memory
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    if hasattr(pyramidkv_utils, 'DISABLE_COMPRESSION') and pyramidkv_utils.DISABLE_COMPRESSION:
        total_tokens = num_layers * input_ids.shape[1]
    elif hasattr(pyramidkv_utils, 'LAYER_BUDGETS') and pyramidkv_utils.LAYER_BUDGETS:
        total_tokens = sum(pyramidkv_utils.LAYER_BUDGETS.values())
    else:
        # Fallback to max_capacity_prompt approximation if LAYER_BUDGETS not used (standard PyramidKV)
        # But for ablation we always use LAYER_BUDGETS
        total_tokens = 0 
        
    kv_cache_mem_bytes = total_tokens * num_heads * head_dim * 2 * 2 # fp16, K+V
    kv_cache_mem_mb = kv_cache_mem_bytes / (1024 * 1024)
    
    return {
        "ppl": ppl,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "throughput_tps": throughput,
        "avg_gflops": avg_gflops,
        "peak_mem_mb": peak_mem_mb,
        "kv_cache_mem_mb": kv_cache_mem_mb
    }

def main():
    parser = argparse.ArgumentParser(description="Run Ablation Studies for RankKV-Pyramid")
    parser.add_argument("--budget", type=int, default=512, help="Average budget per layer")
    parser.add_argument("--ablation_type", type=str, required=True, choices=['random', 'inverse'], help="Type of ablation experiment")
    args = parser.parse_args()

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

    model.eval()

    # Load ranks
    try:
        ranks = load_ranks_from_json(RANK_JSON_PATH)
        print(f"Loaded {len(ranks)} layer ranks.")
    except Exception as e:
        print(f"Error loading ranks: {e}")
        return

    # Calculate standard RankKV budgets first
    target_avg_budget = args.budget
    total_budget_sum = target_avg_budget * len(ranks)
    standard_budgets = allocate_budget_proportional(ranks, total_budget_sum, min_budget=32)
    
    # Apply Ablation Logic
    final_budgets = standard_budgets[:] # Copy
    
    if args.ablation_type == 'random':
        print("Applying Random Shuffle to Budgets...")
        random.seed(42) # Fixed seed for reproducibility
        random.shuffle(final_budgets)
    elif args.ablation_type == 'inverse':
        print("Applying Inverse Rank Logic...")
        
        # 1. Get pairs of (layer_idx, budget)
        pairs = list(enumerate(standard_budgets))
        # 2. Sort by budget descending (High to Low)
        sorted_by_budget = sorted(pairs, key=lambda x: x[1], reverse=True)
        # 3. Extract just the layer indices in order of "Importance" (High Budget -> Important)
        important_layers = [p[0] for p in sorted_by_budget]
        
        # 4. Create a list of budget values sorted Ascending (Low to High)
        low_to_high_budgets = sorted(standard_budgets)
        
        # 5. Assign Low budgets to Important layers
        new_budget_map = {}
        for layer_idx, low_budget in zip(important_layers, low_to_high_budgets):
            new_budget_map[layer_idx] = low_budget
            
        # 6. Reconstruct the list
        final_budgets = [new_budget_map[i] for i in range(len(standard_budgets))]
        
    print(f"Ablation Budgets ({args.ablation_type}): {final_budgets}")

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

        print(f"\nRunning Ablation: {args.ablation_type.upper()}...")
        pyramidkv_utils.DISABLE_COMPRESSION = False
        pyramidkv_utils.LAYER_BUDGETS = {i: b for i, b in enumerate(final_budgets)}
        
        torch.cuda.empty_cache()
        metrics = measure_metrics(model, input_ids)
        
        # Add ablation info to metrics
        metrics['ablation_type'] = args.ablation_type
        
        print(f"Results ({args.ablation_type}): {metrics}")
        print(f"JSON_RESULT: {json.dumps(metrics)}")

if __name__ == "__main__":
    main()
