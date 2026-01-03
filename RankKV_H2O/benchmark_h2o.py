import torch
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from rankkv_h2o import patch, utils
from rankkv_h2o.config import system_cfg, gen_cfg, kv_state

if __name__ == "__main__":
    print(f"Device: {system_cfg.DEVICE}")
    
    # 1. Setup
    tokenizer = AutoTokenizer.from_pretrained(system_cfg.MODEL_ID, cache_dir=system_cfg.CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        system_cfg.MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=system_cfg.CACHE_DIR
    )
    
    patch.enable_h2o_monkey_patch(model)
    long_text = utils.get_real_long_text()

    # 2. Define Experiments
    # Format: (Name, Enable_Compress, Recent_Size, Heavy_Size)
    experiments_config = [
        ("H2O (32+32)",   True,  32,  32),
        ("H2O (64+64)",   True,  64,  64),
        ("H2O (128+128)", True,  128, 128),
        ("H2O (256+256)", True,  256, 256),
    ]

    results = []
    print("\n>>> Starting H2O Benchmarks...\n")
    
    # 3. Loop
    for name, do_compress, r, h in experiments_config:
        
        # --- State Management ---
        kv_state.enable_compression = do_compress
        
        # 设置 H2O 参数
        kv_state.set_budget(recent=r, heavy=h, layer_map=None)
        
        # --- Run ---
        res = utils.run_benchmark(model, tokenizer, long_text, exp_label=name)
        results.append(res)
        print(f"   -> Done. PPL: {res['PPL']:.2f}")

    # 4. Save
    if not os.path.exists(system_cfg.OUTPUT_DIR):
        os.makedirs(system_cfg.OUTPUT_DIR)

    save_filename = "benchmark_H2O_results.csv"
    save_path = os.path.join(system_cfg.OUTPUT_DIR, save_filename)
    
    df = pd.DataFrame(results)
    print("\n================ FINAL H2O KV RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")