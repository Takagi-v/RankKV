import torch
import time
from tqdm import tqdm
from datasets import load_dataset
from .config import kv_state, gen_cfg

def get_real_long_text(min_tokens=None):
    if min_tokens is None:
        min_tokens = gen_cfg.eval_ppl_len
        
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        long_text = ""
        for item in dataset["text"]:
            long_text += item
            if len(long_text) / 4 > min_tokens * 1.5: 
                break
        return long_text
    except Exception as e:
        print(f"Dataset load failed: {e}, using dummy text.")
        return "The history of artificial intelligence " * 500

def run_benchmark(model, tokenizer, text, exp_label: str):
    """
    统一运行：显存、TPOT、PPL
    注意：exp_config 字典被移除，改为直接读取当前的 kv_state，
    调用者需要在外部先配置好 kv_state。
    """
    
    # 重置统计
    kv_state.reset_stats()
    if not kv_state.enable_compression:
        current_budget_info = "Full Cache"
    else:
        base_budget = kv_state.recent_size + kv_state.heavy_size
        current_budget_info = f"{base_budget}"

    print(f"   [Running] {exp_label} - Speed & Memory...")
    
    # 1. Speed Test
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # 使用 gen_cfg 中的参数
    trunc_len = gen_cfg.benchmark_input_len
    if inputs.input_ids.shape[1] > trunc_len:
        inputs.input_ids = inputs.input_ids[:, :trunc_len]
        inputs.attention_mask = inputs.attention_mask[:, :trunc_len]
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            min_new_tokens=gen_cfg.min_new_tokens,
            max_new_tokens=gen_cfg.max_new_tokens, 
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    # 使用 gen_cfg 计算
    generated_tokens = gen_cfg.max_new_tokens 
    tpot = (total_time / generated_tokens) * 1000 
    throughput = generated_tokens / total_time
    effective_kv_gb = sum(kv_state.layer_kv_sizes.values()) / 1024**3
    
    # 2. PPL Test
    print(f"   [Running] {exp_label} - PPL...")
    torch.cuda.empty_cache()
    
    # 使用 gen_cfg 中的参数
    eval_len = gen_cfg.eval_ppl_len
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)[:, :eval_len]
    
    nlls = []
    past_key_values = None
    prev_token = input_ids[:, 0:1]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    with torch.no_grad():
        for i in tqdm(range(1, input_ids.size(1)), leave=False):
            target_token = input_ids[:, i:i+1]
            outputs = model(prev_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            loss = loss_fct(logits, target_token.view(-1))
            nlls.append(loss)
            prev_token = target_token
            
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    
    return {
        "Experiment": exp_label,
        "Budget": current_budget_info,
        "KV Size (GB)": effective_kv_gb,
        "TPOT (ms)": tpot,
        "Throughput (T/s)": throughput,
        "PPL": ppl
    }