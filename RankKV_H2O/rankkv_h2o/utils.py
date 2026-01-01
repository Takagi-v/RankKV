import torch
import time
from tqdm import tqdm
from datasets import load_dataset
from . import config

def get_real_long_text(min_tokens=2048):
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

def run_benchmark(model, tokenizer, text, exp_config):
    """统一运行：显存、TPOT、PPL"""
    # 更新全局配置
    config.ENABLE_KV_COMPRESSION = exp_config["compress"]
    config.H2O_RECENT_SIZE = exp_config.get("r", 64)
    config.H2O_HEAVY_SIZE = exp_config.get("h", 64)
    # LAYER_BUDGETS 由外部调用者在 strategy 中设置，这里只需重置统计
    config.LAYER_KV_SIZES = {}

    print(f"   [Running] {exp_config['name']} - Speed & Memory...")
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] > 100:
        inputs.input_ids = inputs.input_ids[:, :100]
        inputs.attention_mask = inputs.attention_mask[:, :100]
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            min_new_tokens=1024,
            max_new_tokens=1024, 
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    tpot = (total_time / 1024) * 1000 
    throughput = 1024 / total_time
    effective_kv_gb = sum(config.LAYER_KV_SIZES.values()) / 1024**3
    
    # PPL 测试
    print(f"   [Running] {exp_config['name']} - PPL...")
    torch.cuda.empty_cache()
    
    eval_len = 2048
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
        "Experiment": exp_config["name"],
        "Budget": f"{config.H2O_RECENT_SIZE + config.H2O_HEAVY_SIZE}",
        "KV Size (GB)": effective_kv_gb,
        "TPOT (ms)": tpot,
        "Throughput (T/s)": throughput,
        "PPL": ppl
    }