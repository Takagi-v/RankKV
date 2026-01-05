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
    """
    Load calibration and test data from specified dataset.
    
    Args:
        dataset_name: "wikitext" or "pg19"
        num_calibration: number of calibration samples
        num_test: number of test samples
        seq_len: minimum sequence length threshold
    
    Returns:
        calibration_texts, test_texts
    """
    print(f"Loading {dataset_name}...")
    
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # 降低长度阈值：从 seq_len * 4 改为 seq_len，以获取更多样本
        min_length = seq_len
        texts = [t for t in ds["text"] if len(t) > min_length]
        print(f"Found {len(texts)} texts with length > {min_length} characters")
        
    elif dataset_name == "pg19":
        ds = load_dataset("pg19", split="test", streaming=True)
        min_length = seq_len
        texts = []
        for x in ds:
            if len(x['text']) > min_length:
                texts.append(x['text'])
                if len(texts) >= num_calibration + num_test:
                    break
        print(f"Loaded {len(texts)} texts from pg19")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 检查是否有足够的数据
    total_needed = num_calibration + num_test
    if len(texts) < total_needed:
        print(f"Warning: Only found {len(texts)} texts, need {total_needed}")
        if len(texts) == 0:
            raise ValueError("No texts found! Try lowering the minimum length threshold.")
        # 按比例分配可用数据
        ratio = num_calibration / total_needed
        num_calibration = max(1, int(len(texts) * ratio))
        num_test = len(texts) - num_calibration
        print(f"Adjusted: calibration={num_calibration}, test={num_test}")
    
    calibration_texts = texts[:num_calibration]
    test_texts = texts[num_calibration:num_calibration + num_test]
    
    print(f"Final split: {len(calibration_texts)} calibration, {len(test_texts)} test texts")
    
    return calibration_texts, test_texts


def compute_rank_profile(model, tokenizer, texts, device, max_length=1024):
    """
    Compute rank profile for each text using RankKVAnalyzer.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: list of text strings
        device: torch device
        max_length: maximum sequence length for tokenization
    
    Returns:
        numpy array of shape [num_texts, num_layers]
    """
    if len(texts) == 0:
        print("Warning: No texts provided for rank computation!")
        return np.array([])
    
    analyzer = RankKVAnalyzer(model)
    all_ranks = []
    
    for text in tqdm(texts, desc="Computing Ranks"):
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        ).input_ids.to(device)
        
        with torch.no_grad():
            res = analyzer.analyze(inputs, show_progress=False)
        all_ranks.append(res.layer_ranks)
    
    return np.array(all_ranks)


def main():
    model_name = "EleutherAI/pythia-2.8b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 1. Prepare Data
    try:
        calib_texts, test_texts = load_data("wikitext", num_calibration=32, num_test=50)
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Compute Static Profile from Calibration Set
    print("\n--- Phase 1: Generating Static Profile (Calibration) ---")
    calib_ranks = compute_rank_profile(model, tokenizer, calib_texts, device)
    
    if calib_ranks.size == 0:
        print("Error: No calibration ranks computed!")
        return
    
    static_profile = np.mean(calib_ranks, axis=0)  # [NumLayers]
    
    # Save static profile
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../results/static_verification")
    )
    os.makedirs(output_dir, exist_ok=True)
    
    profile_path = os.path.join(output_dir, "static_profile_wikitext.json")
    with open(profile_path, "w") as f:
        json.dump(static_profile.tolist(), f)
    print(f"Static profile saved to {profile_path}")
    
    # 3. Validation on Test Set
    print("\n--- Phase 2: Validating Consistency on Test Set ---")
    
    if len(test_texts) == 0:
        print("Warning: No test texts available! Skipping validation.")
        return
    
    test_ranks = compute_rank_profile(model, tokenizer, test_texts, device)
    
    if test_ranks.size == 0:
        print("Warning: No test ranks computed! Skipping validation.")
        return
    
    correlations = []
    similarities = []
    
    static_tensor = torch.tensor(static_profile, dtype=torch.float32)
    
    for i in range(len(test_ranks)):
        sample_rank = test_ranks[i]
        sample_tensor = torch.tensor(sample_rank, dtype=torch.float32)
        
        # Pearson Correlation
        corr, _ = pearsonr(static_profile, sample_rank)
        correlations.append(corr)
        
        # Cosine Similarity
        cos_sim = F.cosine_similarity(
            static_tensor.unsqueeze(0), 
            sample_tensor.unsqueeze(0)
        ).item()
        similarities.append(cos_sim)
    
    avg_corr = np.mean(correlations)
    avg_sim = np.mean(similarities)
    
    print(f"\nResults over {len(test_ranks)} test samples:")
    print(f"Average Pearson Correlation with Static Profile: {avg_corr:.4f}")
    print(f"Average Cosine Similarity with Static Profile:   {avg_sim:.4f}")
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Static Profile
    plt.plot(
        static_profile, 
        label='Static Profile (Mean of Calibration)', 
        color='black', 
        linewidth=2.5
    )
    
    # Plot some random test samples (up to 5)
    num_samples_to_plot = min(5, len(test_ranks))
    for i in range(num_samples_to_plot):
        plt.plot(test_ranks[i], color='gray', alpha=0.3, linestyle='--')
    
    if num_samples_to_plot > 0:
        plt.plot([], [], color='gray', alpha=0.3, linestyle='--', label='Individual Test Samples')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Effective Rank')
    plt.title(f'Rank Consistency: Static vs Individual (Sim={avg_sim:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path_png = os.path.join(output_dir, "consistency_plot.png")
    save_path_pdf = os.path.join(output_dir, "consistency_plot.pdf")
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"Plot saved to {save_path_png} and {save_path_pdf}")
    
    # 5. Save detailed results
    results = {
        "num_calibration_samples": len(calib_texts),
        "num_test_samples": len(test_ranks),
        "avg_pearson_correlation": float(avg_corr),
        "avg_cosine_similarity": float(avg_sim),
        "correlations": [float(c) for c in correlations],
        "similarities": [float(s) for s in similarities],
    }
    
    results_path = os.path.join(output_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {results_path}")


if __name__ == "__main__":
    main()