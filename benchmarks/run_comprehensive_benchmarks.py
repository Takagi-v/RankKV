import os
import subprocess
import json
import re
import argparse
from datetime import datetime

# Define budgets to test
BUDGETS = [256, 512, 1024]
# BUDGETS = [128, 256, 512, 1024] # Can be expanded

# Define benchmarks
BENCHMARKS = [
    {
        "name": "Original SnapKV",
        "script": "RankKV_Snap/benchmark_original_snap.py",
        "type": "original",
        "method": "snapkv"
    },
    {
        "name": "RankKV + SnapKV",
        "script": "RankKV_Snap/benchmark_rankkv_snap.py",
        "type": "rankkv",
        "method": "snapkv"
    },
    {
        "name": "Original PyramidKV",
        "script": "RankKV_Pyramid/benchmark_original_pyramid.py",
        "type": "original",
        "method": "pyramidkv"
    },
    {
        "name": "RankKV + PyramidKV",
        "script": "RankKV_Pyramid/benchmark_rankkv_vs_standard.py",
        "type": "rankkv",
        "method": "pyramidkv"
    }
]

# Output file
OUTPUT_FILE = "results/comprehensive_benchmark_results.md"
os.makedirs("results", exist_ok=True)

def run_benchmark(script_path, budget):
    """Runs a benchmark script and parses its JSON output."""
    cmd = ["python", script_path, "--budget", str(budget)]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run process and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        output = result.stdout
        
        # Extract all JSON results using regex
        matches = re.findall(r"JSON_RESULT:\s*(\{.*?\})", output) # Non-greedy match
        
        results = []
        if matches:
            for json_str in matches:
                try:
                    data = json.loads(json_str)
                    results.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON chunk from {script_path}")
            return results
        else:
            print(f"No JSON result found in {script_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        print("Stderr:", e.stderr)
        return None

def main():
    # Initialize Markdown file
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        f.write(f"# Comprehensive Benchmark Results\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Device:** {('cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') or True else 'cpu')}\n") # Simplified check
        f.write(f"**Model:** EleutherAI/pythia-2.8b\n\n")

    # Run Full KV Baseline Once
    print("\nRunning Full KV Baseline (once)...")
    full_kv_script = "../RankKV_Snap/benchmark_full_kv.py"
    full_kv_results = []
    
    cmd = ["python", full_kv_script]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        matches = re.findall(r"JSON_RESULT:\s*(\{.*?\})", result.stdout)
        for match in matches:
             full_kv_results.append(json.loads(match))
        print(f"Got {len(full_kv_results)} Full KV results")
    except subprocess.CalledProcessError as e:
        print(f"Error running Full KV: {e}")
        print("Stderr:", e.stderr)

    all_results = {}

    # Skip compressed benchmarks for now as requested
    print("\nSkipping compressed benchmarks (Full KV only requested)...")
    
    with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
        f.write(f"## Full KV Baseline Results\n\n")
        f.write("| Method | Type | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | GFLOPs | Peak Mem (MB) | KV Cache (MB) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
    
    # Write Full KV Results
    datasets = ['wikitext', 'pg19']
    for i, metrics in enumerate(full_kv_results):
         dataset = datasets[i] if i < len(datasets) else f"unknown_{i}"
         if isinstance(metrics, dict) and 'ppl' in metrics:
             ppl = metrics.get('ppl', 0)
             ttft = metrics.get('ttft_ms', 0)
             tpot = metrics.get('tpot_ms', 0)
             throughput = metrics.get('throughput_tps', 0)
             gflops = metrics.get('avg_gflops', 0)
             peak_mem = metrics.get('peak_mem_mb', 0)
             kv_mem = metrics.get('kv_cache_mem_mb', 0)
             
             row = f"| full_kv | baseline | {dataset} | {ppl:.2f} | {ttft:.2f} | {tpot:.2f} | {throughput:.2f} | {gflops:.2f} | {peak_mem:.2f} | {kv_mem:.2f} |"
             with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
                 f.write(row + "\n")
         else:
              with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
                  f.write(f"| full_kv | baseline | {dataset} | INVALID | - | - | - | - | - | - |\n")

    # return # Exit after Full KV

    # Original Loop (disabled)
    for budget in BUDGETS:
        print(f"\n{'='*40}\nTesting Average Budget: {budget}\n{'='*40}")
        
        with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
            f.write(f"## Average Budget: {budget}\n\n")
            f.write("| Method | Type | Dataset | PPL | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | GFLOPs | Peak Mem (MB) | KV Cache (MB) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        # Write Full KV Results first
        datasets = ['wikitext', 'pg19']
        for i, metrics in enumerate(full_kv_results):
             dataset = datasets[i] if i < len(datasets) else f"unknown_{i}"
             if isinstance(metrics, dict) and 'ppl' in metrics:
                 ppl = metrics.get('ppl', 0)
                 ttft = metrics.get('ttft_ms', 0)
                 tpot = metrics.get('tpot_ms', 0)
                 throughput = metrics.get('throughput_tps', 0)
                 gflops = metrics.get('avg_gflops', 0)
                 peak_mem = metrics.get('peak_mem_mb', 0)
                 kv_mem = metrics.get('kv_cache_mem_mb', 0)
                 
                 row = f"| full_kv | baseline | {dataset} | {ppl:.2f} | {ttft:.2f} | {tpot:.2f} | {throughput:.2f} | {gflops:.2f} | {peak_mem:.2f} | {kv_mem:.2f} |"
                 with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
                     f.write(row + "\n")
             else:
                  with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
                      f.write(f"| full_kv | baseline | {dataset} | INVALID | - | - | - | - | - | - |\n")

        for bench in BENCHMARKS:
            print(f"Starting {bench['name']}...")
            results_list = run_benchmark(bench['script'], budget)
            
            if results_list:
                print(f"Got {len(results_list)} results for {bench['name']}")
                
                # Assume order is wikitext, pg19
                datasets = ['wikitext', 'pg19']
                
                for i, metrics in enumerate(results_list):
                    if i < len(datasets):
                        dataset = datasets[i]
                    else:
                        dataset = f"unknown_{i}"
                        
                    if isinstance(metrics, dict) and 'ppl' in metrics:
                        # Ensure all keys exist, default to -1 or 0 if missing
                        ppl = metrics.get('ppl', 0)
                        ttft = metrics.get('ttft_ms', 0)
                        tpot = metrics.get('tpot_ms', 0)
                        throughput = metrics.get('throughput_tps', 0)
                        gflops = metrics.get('avg_gflops', 0)
                        peak_mem = metrics.get('peak_mem_mb', 0)
                        kv_mem = metrics.get('kv_cache_mem_mb', 0)
                        
                        row = f"| {bench['method']} | {bench['type']} | {dataset} | {ppl:.2f} | {ttft:.2f} | {tpot:.2f} | {throughput:.2f} | {gflops:.2f} | {peak_mem:.2f} | {kv_mem:.2f} |"
                        with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
                            f.write(row + "\n")
                    else:
                         with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
                             f.write(f"| {bench['method']} | {bench['type']} | {dataset} | INVALID_FORMAT | - | - | - | - | - | - |\n")

            else:
                with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
                    f.write(f"| {bench['method']} | {bench['type']} | ERROR | - | - | - | - | - | - | - |\n")

    print(f"\nAll benchmarks completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
