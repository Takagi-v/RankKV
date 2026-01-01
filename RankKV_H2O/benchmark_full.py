import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from rankkv_h2o import config, patch, utils

if __name__ == "__main__":
    # 1. Setup Environment
    print(f"Device: {config.DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, cache_dir=config.CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=config.CACHE_DIR
    )
    
    # Inject logic to enable memory tracking (even for full attention)
    patch.enable_h2o_monkey_patch(model)
    
    # Load test data
    long_text = utils.get_real_long_text()
    print(f"Test Text Length: {len(tokenizer(long_text)['input_ids'][0])} tokens")

    # 2. Define Experiment (Full / Dense)
    # compress=False ensures no tokens are evicted
    exp_dense = {
        "name": "Dense (Full KV)", 
        "compress": False, 
        "r": 0, 
        "h": 0
    }

    results = []
    print("\n>>> Starting Full KV Benchmark (Upper Bound)...\n")
    
    # 3. Run Benchmark
    # Reset any previous budgets (safety measure)
    config.LAYER_BUDGETS = {} 
    
    res = utils.run_benchmark(model, tokenizer, long_text, exp_dense)
    results.append(res)
    
    print(f"   -> Done. PPL: {res['PPL']:.2f}")

    # 4. Save Results
    import os

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Created output directory: {config.OUTPUT_DIR}")

    save_filename = "benchmark_full_results.csv"
    save_path = os.path.join(config.OUTPUT_DIR, save_filename)
    
    df = pd.DataFrame(results)
    print("\n================ FINAL FULL KV RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")