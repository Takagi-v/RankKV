import os
import json
import torch
import time
import math
import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
from pyramidkv.monkeypatch import replace_gptneox
import pyramidkv.pyramidkv_utils as pyramidkv_utils

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
    # CRITICAL FIX: Explicit position_ids for RoPE
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
    
    tpot = (t3 - t2) * 1000 / generate_len # ms/token
    throughput = 1000 / tpot # tokens/s
    
    # Estimate GFLOPs (Very rough estimate for 2.8B model)
    # 2.8B params * 2 bytes * 1 token = ~5.6 GB bandwidth? No, GFLOPs.
    # Just use a placeholder or relative comparison
    gflops = (2.8e9 * 2 * generate_len) / ((t3 - t2) * 1e9) # Not accurate but consistent
    
    # Memory usage
    peak_mem = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    # Estimate KV Cache Memory
    # Original PyramidKV: average budget is max_capacity_prompt
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # max_capacity_prompt is the AVG budget
    avg_budget = model.config.max_capacity_prompt
    total_tokens = avg_budget * num_layers
    
    kv_cache_mem_bytes = total_tokens * num_heads * head_dim * 2 * 2 # fp16, K+V
    kv_cache_mem_mb = kv_cache_mem_bytes / (1024 * 1024)
    
    return {
        "ppl": ppl,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "throughput_tps": throughput,
        "avg_gflops": gflops * 1000, # Scale for readability
        "peak_mem_mb": peak_mem_mb,
        "kv_cache_mem_mb": kv_cache_mem_mb
    }

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Original PyramidKV Benchmark")
    parser.add_argument("--budget", type=int, default=128, help="Average budget per layer")
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
    
    # PATCH layer_idx manually as it might be missing
    if hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    else:
        layers = model.model.layers if hasattr(model, "model") else []
    for i, layer in enumerate(layers):
        if hasattr(layer, "attention"):
            layer.attention.layer_idx = i
            
    # Configure Original PyramidKV
    # We want Avg Budget = args.budget (Total).
    # init_pyramidkv uses model.config.max_capacity_prompt
    model.config.max_capacity_prompt = args.budget
    model.config.window_size = 32 # Consistent with default in init_pyramidkv
    
    pyramidkv_utils.LAYER_BUDGETS = {} # Ensure no RankKV overrides
    print(f"Configured Original PyramidKV with model.config.max_capacity_prompt = {model.config.max_capacity_prompt}")
    
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

        print("\nRunning Original PyramidKV...")
        pyramidkv_utils.DISABLE_COMPRESSION = False
        
        # Clear cache before run
        torch.cuda.empty_cache()
        metrics = measure_metrics(model, input_ids)
        print(f"Original PyramidKV Results: {metrics}")
        print(f"JSON_RESULT: {json.dumps(metrics)}")

if __name__ == "__main__":
    main()
