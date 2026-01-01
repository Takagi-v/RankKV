import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from rankkv_h2o import config, patch, utils

if __name__ == "__main__":
    print(f"Device: {config.DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, cache_dir=config.CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=config.CACHE_DIR
    )
    
    # 注入 H2O 逻辑
    patch.enable_h2o_monkey_patch(model)
    
    long_text = utils.get_real_long_text()

    # 定义实验列表
    all_experiments = [
        {"name": "Baseline", "compress": False, "r": 0, "h": 0},
        {"name": "H2O (64+64)", "compress": True, "r": 64, "h": 64},
        {"name": "Local (128)", "compress": True, "r": 128, "h": 0},
        {"name": "H2O (128+128)", "compress": True, "r": 128, "h": 128},
    ]

    results = []
    print("\n>>> Starting H2O Benchmarks...\n")
    
    for exp in all_experiments:
        # 重置 Layer Budget 为空，确保使用标准 H2O 逻辑
        config.LAYER_BUDGETS = {} 
        res = utils.run_benchmark(model, tokenizer, long_text, exp)
        results.append(res)
        print(f"   -> Done. PPL: {res['PPL']:.2f}")

    import os

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Created output directory: {config.OUTPUT_DIR}")

    save_filename = "benchmark_H2O_results.csv"
    save_path = os.path.join(config.OUTPUT_DIR, save_filename)
    
    df = pd.DataFrame(results)
    print("\n================ FINAL H2O KV RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")