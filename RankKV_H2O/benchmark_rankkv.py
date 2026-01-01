import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from rankkv_h2o import config, patch, utils, strategy

if __name__ == "__main__":
    print(f"Device: {config.DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, cache_dir=config.CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=config.CACHE_DIR
    )
    patch.enable_h2o_monkey_patch(model)
    
    long_text = utils.get_real_long_text()
    
    budget_levels = [64, 128, 256, 512]
    results = []
    
    for budget in budget_levels:
        r = budget // 2
        h = budget - r
        
        # --- 1. H2O Baseline (不使用 Rank) ---
        config.LAYER_BUDGETS = {} # 清空
        exp_h2o = {
            "name": f"H2O (B={budget})", "compress": True, "r": r, "h": h
        }
        res_h2o = utils.run_benchmark(model, tokenizer, long_text, exp_h2o)
        results.append(res_h2o)

        # --- 2. RankKV (使用 Rank) ---
        target_avg_budget = r + h
        
        # A. Profiling (获取 Ranks)
        ranks = strategy.profile_model_ranks(model, tokenizer, long_text, model.device)
        
        # B. Allocation (分配 Budgets)
        config.LAYER_BUDGETS = strategy.allocate_budgets(
            ranks, 
            total_avg_budget=target_avg_budget, 
            num_layers=len(model.gpt_neox.layers),
            min_budget=32,
            alpha=0.3,
            inverse=False 
        )
        
        exp_rankkv = {
            "name": f"RankKV (B={budget})", "compress": True, "r": r, "h": h
        }
        res_rankkv = utils.run_benchmark(model, tokenizer, long_text, exp_rankkv)
        results.append(res_rankkv)

    import os

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Created output directory: {config.OUTPUT_DIR}")

    save_filename = "benchmark_rankkv_results.csv"
    save_path = os.path.join(config.OUTPUT_DIR, save_filename)
    
    df = pd.DataFrame(results)
    print("\n================ FINAL RANK KV RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")