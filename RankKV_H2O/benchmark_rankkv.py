import torch
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from rankkv_h2o import patch, utils, strategy
from rankkv_h2o.config import system_cfg, gen_cfg, kv_state

if __name__ == "__main__":
    print(f"Device: {system_cfg.DEVICE}")
    print(f"Model: {system_cfg.MODEL_ID}")
    
    # ================ 1. Centralized Configuration =================
    gen_cfg.min_new_tokens = 512
    gen_cfg.max_new_tokens = 512
    gen_cfg.eval_ppl_len = 2048
    
    BUDGET_LEVELS = [64, 128, 256, 512]
    RANK_ALPHA = 0.3
    MIN_LAYER_BUDGET = 32
    
    # ================ 2. Model Loading =================
    tokenizer = AutoTokenizer.from_pretrained(system_cfg.MODEL_ID, cache_dir=system_cfg.CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        system_cfg.MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=system_cfg.CACHE_DIR
    )
    patch.enable_h2o_monkey_patch(model)
    
    long_text = utils.get_real_long_text()
    
    # ================ 3. Rank Profiling =================
    ranks = {}
    
    if kv_state.rank_mode == "dynamic":
        print(f">>> [Mode: DYNAMIC] Profiling model ranks now...")
        ranks = strategy.profile_model_ranks(model, tokenizer, long_text, model.device)
        strategy.save_ranks(ranks, system_cfg.static_rank_path)
        
    elif kv_state.rank_mode == "static":
        print(f">>> [Mode: STATIC] Loading ranks from {system_cfg.static_rank_path}...")
        if os.path.exists(system_cfg.static_rank_path):
            ranks = strategy.load_ranks(system_cfg.static_rank_path)
        else:
            print(f"!!! Warning: Static file not found. Fallback to DYNAMIC profiling.")
            ranks = strategy.profile_model_ranks(model, tokenizer, long_text, model.device)
            strategy.save_ranks(ranks, system_cfg.static_rank_path)

    # ================ 4. Experiment Loop =================
    results = []
    
    for budget in BUDGET_LEVELS:
        r = budget // 2
        h = budget - r
        target_avg_budget = r + h
        
        # 计算分配
        layer_budgets = strategy.allocate_budgets(
            ranks, 
            total_avg_budget=target_avg_budget, 
            num_layers=len(model.gpt_neox.layers),
            min_budget=MIN_LAYER_BUDGET,
            alpha=RANK_ALPHA
        )
        
        # 应用状态
        kv_state.enable_compression = True
        kv_state.set_budget(recent=r, heavy=h, layer_map=layer_budgets)
        
        res_rankkv = utils.run_benchmark(model, tokenizer, long_text, exp_label=f"RankKV (B={budget})")
        results.append(res_rankkv)

        print(f"   -> Done. PPL: {res_rankkv['PPL']:.2f}")

    # ================ 5. Save Results =================
    if not os.path.exists(system_cfg.OUTPUT_DIR):
        os.makedirs(system_cfg.OUTPUT_DIR)

    save_path = os.path.join(system_cfg.OUTPUT_DIR, "benchmark_rankkv_results.csv")
    df = pd.DataFrame(results)
    
    print("\n================ FINAL RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")