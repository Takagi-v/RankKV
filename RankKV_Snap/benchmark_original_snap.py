
import torch
import time
import json
import os
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
import math

# Add the current directory to sys.path to ensure modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import monkeypatch and utils
from snapkv.monkeypatch import replace_gptneox
from snapkv.snapkv_utils import LAYER_BUDGETS

# Configuration
MODEL_NAME = "EleutherAI/pythia-2.8b"
DATASET_NAMES = ["wikitext", "pg19"] 
SEQ_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_BUDGET = 256  # Fixed budget for Original SnapKV comparison

def get_wikitext2(tokenizer, seq_len):
    print("Loading dataset: wikitext...")
    try:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    except Exception as e:
        print(f"Error loading wikitext: {e}. Falling back to local/dummy.")
        return None
        
    encodings = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    # Just take one sequence of seq_len for demonstration/speed
    if encodings.input_ids.size(1) < seq_len:
        print(f"Warning: Text too short ({encodings.input_ids.size(1)} < {seq_len})")
        return encodings.input_ids.to(DEVICE)
    
    return encodings.input_ids[:, :seq_len].to(DEVICE)

def get_pg19(tokenizer, seq_len):
    print("Loading dataset: pg19...")
    try:
        # PG-19 is large, we try to load a subset or handle error
        testdata = load_dataset('pg19', split='test', streaming=True)
        # Get first example
        text = next(iter(testdata))['text']
        encodings = tokenizer(text, return_tensors='pt')
        return encodings.input_ids[:, :seq_len].to(DEVICE)
    except Exception as e:
        print(f"Error loading PG-19: {e}. Falling back to local dummy text.")
        # Create dummy text
        dummy_text = "The quick brown fox jumps over the lazy dog. " * 500
        encodings = tokenizer(dummy_text, return_tensors='pt')
        return encodings.input_ids[:, :seq_len].to(DEVICE)

def evaluate_performance(model, input_ids, budget):
    # Warmup
    with torch.no_grad():
        _ = model(input_ids[:, :128])
    
    torch.cuda.synchronize()
    
    seq_len = input_ids.shape[1]
    # Split into context (for compression) and target (for PPL measurement)
    # Same logic as benchmark_rankkv_snap.py
    split_idx = int(seq_len * 0.75)
    context_ids = input_ids[:, :split_idx]
    target_ids = input_ids[:, split_idx:]
    
    # 1. Prefill / TTFT
    start_time = time.time()
    with torch.no_grad():
        outputs = model(context_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.time() - start_time
    ttft_ms = prefill_time * 1000
    
    past_key_values = outputs.past_key_values
    
    # 2. PPL Calculation
    # We measure PPL on the target_ids using the compressed past_key_values
    # CRITICAL FIX: Must provide correct position_ids because cache is compressed!
    # The cache length is small (e.g. 256), but the actual position is split_idx (e.g. 1536).
    # If we don't provide position_ids, HF generates [256, 257...] which messes up RoPE.
    
    past_length = split_idx # The number of tokens actually processed (before compression)
    # Target ids positions: [past_length, past_length + 1, ..., seq_len - 1]
    position_ids = torch.arange(past_length, seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
    
    with torch.no_grad():
        outputs_ppl = model(
            target_ids, 
            past_key_values=past_key_values, 
            position_ids=position_ids, 
            use_cache=True, 
            labels=target_ids
        )
        ppl_loss = outputs_ppl.loss.item()
        # Avoid overflow
        ppl = math.exp(ppl_loss) if ppl_loss < 20 else 1e9

    # 3. Generation Speed (TPOT)
    # Generate 50 tokens
    gen_len = 50
    next_token = context_ids[:, -1:] # Start generation from end of context
    
    # We need to track position for generation too
    current_pos = split_idx
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(gen_len):
            # Position for the single new token
            pos_id = torch.tensor([[current_pos]], dtype=torch.long, device=model.device)
            
            outputs = model(next_token, past_key_values=past_key_values, position_ids=pos_id, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            current_pos += 1
            
    torch.cuda.synchronize()
    gen_time = time.time() - start_time
    tpot_ms = (gen_time / gen_len) * 1000
    throughput = gen_len / gen_time
    
    # FLOPs estimation
    # Pythia 2.8B ~ 2.8e9 params
    avg_gflops = (2.8e9 * 2 * gen_len) / gen_time / 1e9

    # Memory usage
    peak_mem = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    # Estimate KV Cache Memory
    # For Original SnapKV, budget is fixed per layer.
    # KV Cache = layers * (Budget * num_heads * head_dim * 2 bytes * 2 (K+V))
    # Pythia 2.8B: 32 layers, 32 heads, 128 dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # Avg budget per layer is passed as argument
    total_tokens = budget * num_layers
    kv_cache_mem_bytes = total_tokens * num_heads * head_dim * 2 * 2 # fp16, K+V
    kv_cache_mem_mb = kv_cache_mem_bytes / (1024 * 1024)

    return {
        "ppl": ppl,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "throughput_tps": throughput,
        "avg_gflops": avg_gflops,
        "peak_mem_mb": peak_mem_mb,
        "kv_cache_mem_mb": kv_cache_mem_mb
    }

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Original SnapKV Benchmark")
    parser.add_argument("--budget", type=int, default=256, help="Average budget per layer")
    args = parser.parse_args()
    
    FIXED_BUDGET = args.budget

    print(f"Using device: {DEVICE}")
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Patch before loading to ensure class methods are replaced
    replace_gptneox("snapkv")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map=DEVICE, 
        trust_remote_code=True
    )
    
    # Manually patch layer_idx onto attention modules as Pythia/GPT-NeoX implementation might not expose it
    print("Patching layer_idx...")
    if hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    else:
        layers = model.model.layers if hasattr(model, "model") else []
        
    for i, layer in enumerate(layers):
        if hasattr(layer, "attention"):
            layer.attention.layer_idx = i
    
    # Apply Original SnapKV (Fixed Budget)
    # Clear RankKV budgets to ensure we use the fixed capacity
    LAYER_BUDGETS.clear()
    
    # Set config for SnapKV
    # Note: max_capacity_prompt includes window_size
    model.config.window_size = 32
    model.config.max_capacity_prompt = FIXED_BUDGET
    model.config.kernel_size = 5
    model.config.pooling = 'avgpool'
    
    # replace_gptneox("snapkv")  # Already called above
    
    print(f"Original SnapKV Configuration: Fixed Budget = {FIXED_BUDGET}, Window Size = 32")

    for dataset_name in DATASET_NAMES:
        print(f"\n==================== Processing {dataset_name} ====================")
        
        if dataset_name == "wikitext":
            input_ids = get_wikitext2(tokenizer, SEQ_LEN)
        elif dataset_name == "pg19":
            input_ids = get_pg19(tokenizer, SEQ_LEN)
            
        if input_ids is None:
            continue
            
        print(f"Input shape: {input_ids.shape}")
        
        # Run Original SnapKV
        print(f"\nRunning Original SnapKV (Fixed Budget {FIXED_BUDGET})...")
        results = evaluate_performance(model, input_ids, FIXED_BUDGET)
        
        # Print JSON-like results for parsing
        print(f"JSON_RESULT: {json.dumps(results)}")

        print(f"Original SnapKV Results: {results}")
        
        # Save results to file for later comparison if needed
        # (Optional, but good for record)

if __name__ == "__main__":
    main()
