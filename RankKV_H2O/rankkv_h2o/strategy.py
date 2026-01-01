import torch
import torch.nn.functional as F

def compute_effective_rank(matrix):
    """计算单个Attention矩阵的有效秩"""
    try:
        U, S, V = torch.svd(matrix.float())
    except:
        return 1.0 # 兜底
    
    S_normalized = S / (S.sum() + 1e-10)
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    effective_rank = torch.exp(entropy)
    return effective_rank.item()

def profile_model_ranks(model, tokenizer, calibration_text, device):
    """预运行一次以获取每层的有效秩"""
    print(">>> [RankKV] Profiling Effective Ranks...")
    inputs = tokenizer(calibration_text, return_tensors="pt").to(device)
    if inputs.input_ids.shape[1] > 256:
        inputs.input_ids = inputs.input_ids[:, :256]
        
    rank_data = {}
    
    def get_attn_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_matrix = output[1]
                if attn_matrix is not None:
                    avg_attn = attn_matrix[0].mean(dim=0) 
                    rank = compute_effective_rank(avg_attn)
                    rank_data[layer_idx] = rank
        return hook

    handles = []
    for i, layer in enumerate(model.gpt_neox.layers):
        handles.append(layer.attention.register_forward_hook(get_attn_hook(i)))
    
    with torch.no_grad():
        model(inputs.input_ids, output_attentions=True)
        
    for h in handles: h.remove()
    
    print(f">>> [RankKV] Ranks: {rank_data}")
    return rank_data

def allocate_budgets(ranks, total_avg_budget, num_layers, min_budget=64, alpha=0.5, inverse=False):
    """根据 Rank 分配 Budget"""
    rank_list = [ranks.get(i, 100.0) for i in range(num_layers)]
    ranks_tensor = torch.tensor(rank_list, dtype=torch.float32)
    
    if inverse == True:
        ranks_tensor = 1.0 / (ranks_tensor + 1e-6)
    
    smoothed_ranks = torch.pow(ranks_tensor, alpha)
    weights = smoothed_ranks / smoothed_ranks.sum()
    
    total_token_slots = total_avg_budget * num_layers
    budgets = (weights * total_token_slots).int()
    
    final_budgets = {}
    budgets = torch.maximum(budgets, torch.tensor(min_budget))
    
    current_sum = budgets.sum()
    if current_sum > total_token_slots:
        scale_factor = total_token_slots / current_sum
        budgets = (budgets.float() * scale_factor).int()
        budgets = torch.maximum(budgets, torch.tensor(min_budget))

    for i in range(num_layers):
        final_budgets[i] = budgets[i].item()
    
    print(f">>> [RankKV] Budget Allocation Stats:")
    print(f"    Min: {min(final_budgets.values())}, Max: {max(final_budgets.values())}")
    return final_budgets