
import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rankkv.pipeline import RankKVAnalyzer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_data(dataset_name="wikitext", num_calibration=32, num_test=50, seq_len=2048):
    print(f"Loading {dataset_name}...")
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t) > seq_len * 4]
    elif dataset_name == "pg19":
        ds = load_dataset("pg19", split="test", streaming=True)
        texts = []
        for x in ds:
            if len(x['text']) > seq_len * 4:
                texts.append(x['text'])
                if len(texts) >= num_calibration + num_test:
                    break
    
    calibration_texts = texts[:num_calibration]
    test_texts = texts[num_calibration:num_calibration+num_test]
    return calibration_texts, test_texts

def compute_rank_profile(model, tokenizer, texts, device, max_length=1024):
    analyzer = RankKVAnalyzer(model)
    all_ranks = []
    
    for text in tqdm(texts, desc="Computing Ranks"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
        with torch.no_grad():
            res = analyzer.analyze(inputs, show_progress=False)
        all_ranks.append(res.layer_ranks)
        
    return np.array(all_ranks)

def main():
    model_name = "EleutherAI/pythia-2.8b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # 1. Prepare Data
    calib_texts, test_texts = load_data("wikitext", 32, 50)
    
    # 2. Compute Static Profile from Calibration Set
    print("\n--- Phase 1: Generating Static Profile (Calibration) ---")
    calib_ranks = compute_rank_profile(model, tokenizer, calib_texts, device)
    static_profile = np.mean(calib_ranks, axis=0) # [NumLayers]
    
    # Save static profile
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/static_verification"))
    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, "static_profile_wikitext.json")
    with open(profile_path, "w") as f:
        json.dump(static_profile.tolist(), f)
    print(f"Static profile saved to {profile_path}")
    
    # 3. Validation on Test Set
    print("\n--- Phase 2: Validating Consistency on Test Set ---")
    test_ranks = compute_rank_profile(model, tokenizer, test_texts, device)
    
    correlations = []
    similarities = []
    
    static_tensor = torch.tensor(static_profile, dtype=torch.float32)
    
    for i in range(len(test_texts)):
        sample_rank = test_ranks[i]
        sample_tensor = torch.tensor(sample_rank, dtype=torch.float32)
        
        # Pearson Correlation
        corr, _ = pearsonr(static_profile, sample_rank)
        correlations.append(corr)
        
        # Cosine Similarity
        cos_sim = F.cosine_similarity(static_tensor.unsqueeze(0), sample_tensor.unsqueeze(0)).item()
        similarities.append(cos_sim)
        
    avg_corr = np.mean(correlations)
    avg_sim = np.mean(similarities)
    
    print(f"\nResults over {len(test_texts)} test samples:")
    print(f"Average Pearson Correlation with Static Profile: {avg_corr:.4f}")
    print(f"Average Cosine Similarity with Static Profile:   {avg_sim:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Static Profile
    plt.plot(static_profile, label='Static Profile (Mean of 32)', color='black', linewidth=2.5)
    
    # Plot some random test samples
    for i in range(5):
        plt.plot(test_ranks[i], color='gray', alpha=0.3, linestyle='--')
    plt.plot([], [], color='gray', alpha=0.3, linestyle='--', label='Individual Test Samples')
        
    plt.xlabel('Layer Index')
    plt.ylabel('Effective Rank')
    plt.title(f'Rank Consistency: Static vs Individual (Sim={avg_sim:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, "consistency_plot.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
