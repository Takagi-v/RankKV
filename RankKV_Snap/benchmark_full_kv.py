
import os
import json
import torch
import time
import math
import argparse
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
from snapkv.monkeypatch import replace_gptneox
import snapkv.snapkv_utils as snapkv_utils

def get_data(dataset_name, tokenizer, seq_len=2048):
    print(f"Loading dataset: {dataset_name}...")
    if dataset_name == 'wikitext':
        # WikiText-2
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds['text'])
    elif dataset_name == 'pg19':
        # PG-19
        try:
            ds = load_dataset("pg19", split="test", streaming=True)
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
    
    # Split into context and target
    split_idx = int(seq_len * 0.75)
    context_ids = input_ids[:, :split_idx]
    target_ids = input_ids[:, split_idx:]
    
    # 1. TTFT
    torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.no_grad():
        outputs = model(context_ids, use_cache=True)
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    ttft = (t1 - t0) * 1000 # ms
    past_key_values = outputs.past_key_values
    
    # 2. PPL
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
        
    # 3. TPOT
    next_token = context_ids[:, -1:]
    # Re-use past_key_values from context
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
    
    tpot = (t3 - t2) * 1000 / generate_len # ms
    throughput = generate_len / (t3 - t2)
    
    # Estimate FLOPs
    params = sum(p.numel() for p in model.parameters())
    total_flops = (2 * params * context_ids.shape[1]) + (2 * params * generate_len)
    avg_gflops = (total_flops / 1e9) / (t3 - t0)
    
    # Memory usage
    peak_mem = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    # KV Cache Memory (Full)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # Full KV: all tokens in context + generated
    # Actually, measure_metrics context is split_idx.
    # But usually we talk about memory during generation.
    # Here we estimate memory for the context length?
    # Or maximum memory?
    # Let's align with other benchmarks: total_tokens = num_layers * current_seq_len
    # But other benchmarks calculate 'budget' based memory.
    # For Full KV, budget is seq_len.
    
    # Let's use the sequence length processed so far (split_idx + generate_len eventually)
    # But for comparison, let's use the context length or max length?
    # The 'kv_cache_mem_mb' in other scripts is based on 'total_tokens' allocated.
    # For Full KV, it's the full sequence length.
    total_tokens = seq_len * num_layers # Is this right?
    # Wait, seq_len is 2048.
    # Yes, Full KV stores all tokens.
    
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
    parser = argparse.ArgumentParser()
    # No budget arg needed, but maybe other args?
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "EleutherAI/pythia-2.8b"
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Monkeypatch
    replace_gptneox(None) # method=None or just don't pass anything if we disable compression?
    # replace_gptneox("snapkv") # We need some method to initialize the class, but we will disable it.
    # Actually replace_gptneox takes 'method' arg.
    replace_gptneox("snapkv")
    
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # Assign layer_idx manually as it might be missing
    for i, layer in enumerate(model.gpt_neox.layers):
        layer.attention.layer_idx = i
    
    # Disable Compression
    snapkv_utils.DISABLE_COMPRESSION = True
    snapkv_utils.LAYER_BUDGETS = {}
    
    datasets = ['wikitext', 'pg19']
    seq_len = 2048
    
    results_list = []
    
    for ds_name in datasets:
        print(f"\nProcessing {ds_name}...")
        try:
            input_ids = get_data(ds_name, tokenizer, seq_len).to(device)
            
            torch.cuda.empty_cache()
            metrics = measure_metrics(model, input_ids)
            print(f"Full KV Results ({ds_name}): {metrics}")
            print(f"JSON_RESULT: {json.dumps(metrics)}")
            results_list.append(metrics)
            
        except Exception as e:
            print(f"Error on {ds_name}: {e}")
            results_list.append({"error": str(e)})
            
    # Output JSON for master script
    # We output individual JSON_RESULT lines above

if __name__ == "__main__":
    main()
