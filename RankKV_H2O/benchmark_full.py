import torch
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from rankkv_h2o import patch, utils
from rankkv_h2o.config import system_cfg, gen_cfg, kv_state

if __name__ == "__main__":
    # 1. Setup Environment
    print(f"Device: {system_cfg.DEVICE}")
    print(f"Model: {system_cfg.MODEL_ID}")
    
    # gen_cfg.min_new_tokens = 1024 
    
    tokenizer = AutoTokenizer.from_pretrained(system_cfg.MODEL_ID, cache_dir=system_cfg.CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        system_cfg.MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=system_cfg.CACHE_DIR
    )
    
    patch.enable_h2o_monkey_patch(model)
    
    long_text = utils.get_real_long_text()

    # 2. Configure State for Full KV
    print("\n>>> Starting Full KV Benchmark (Upper Bound)...\n")
    
    kv_state.enable_compression = False
    kv_state.reset_stats()
    
    results = []
    
    # 3. Run Benchmark
    res = utils.run_benchmark(model, tokenizer, long_text, exp_label="Dense (Full KV)")
    results.append(res)
    
    print(f"   -> Done. PPL: {res['PPL']:.2f}")

    # 4. Save Results
    if not os.path.exists(system_cfg.OUTPUT_DIR):
        os.makedirs(system_cfg.OUTPUT_DIR)
        print(f"Created output directory: {system_cfg.OUTPUT_DIR}")

    save_filename = "benchmark_full_results.csv"
    save_path = os.path.join(system_cfg.OUTPUT_DIR, save_filename)
    
    df = pd.DataFrame(results)
    print("\n================ FINAL FULL KV RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")