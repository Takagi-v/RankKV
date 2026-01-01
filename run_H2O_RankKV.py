import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from datasets import load_dataset
from tqdm import tqdm
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
import math
import time
import pandas as pd
import gc

# ================= 配置区域 =================
MODEL_ID = "EleutherAI/pythia-2.8b"
CACHE_DIR = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全局变量
ENABLE_KV_COMPRESSION = False
H2O_RECENT_SIZE = 64
H2O_HEAVY_SIZE = 64
LAYER_KV_SIZES = {}
LAYER_BUDGETS = {} 
# ===========================================

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
    """
    预运行一次以获取每层的有效秩
    """
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
    """
    根据 Rank 分配 Budget (改进版：增加平滑和保底)
    
    Args:
        alpha: 平滑系数。
               alpha=1.0 为线性分配（你之前的策略，贫富差距大）。
               alpha=0.0 为完全均匀分配（等同于 H2O）。
               推荐 alpha=0.5 (平方根) 或 0.3，能保留 Rank 差异但不会太极端。
        min_budget: 每层至少保留的 token 数，建议设为 64 或 32，防止饿死。
        strategy: "direct" (Rank高Budget高) | "inverse" (Rank低Budget高)
    """
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
    print(f"    Sample (L0, L15, L31): {final_budgets[0]}, {final_budgets[15]}, {final_budgets[31]}")
    
    return final_budgets

def h2o_gpt_neox_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    head_mask: torch.FloatTensor = None,
    layer_past=None,
    output_attentions: bool = False,
    cache_position: torch.LongTensor = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    **kwargs
):
    bsz, q_len, _ = hidden_states.size()
    num_attention_heads = self.config.num_attention_heads
    hidden_size = self.config.hidden_size
    head_size = self.head_size 

    qkv = self.query_key_value(hidden_states)
    qkv = qkv.view(bsz, q_len, num_attention_heads, 3 * head_size)
    qkv = qkv.permute(0, 2, 1, 3) 
    query, key, value = qkv.chunk(3, dim=-1)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value, seq_len=key.shape[-2])
    else:
        cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if layer_past is not None:
        key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs={"cache_position": cache_position})

    seq_len = key.shape[2]

    # --- RankKV Budget ---
    if self.layer_idx in LAYER_BUDGETS:
        layer_budget = LAYER_BUDGETS[self.layer_idx]
    else:
        layer_budget = H2O_RECENT_SIZE + H2O_HEAVY_SIZE
    
    current_limit = layer_budget
    
    # 保证 recent 不会超过 limit
    fixed_recent = min(H2O_RECENT_SIZE, current_limit)
    dynamic_heavy = max(0, current_limit - fixed_recent)

    if ENABLE_KV_COMPRESSION and q_len == 1 and seq_len > current_limit:
        
        if hasattr(layer_past, "h2o_scores"):
            prev_len = layer_past.h2o_scores.shape[-1]
            if seq_len > prev_len:
                # 补 0 对齐
                diff = seq_len - prev_len
                layer_past.h2o_scores = torch.cat([
                    layer_past.h2o_scores, 
                    torch.zeros((bsz, num_attention_heads, diff), device=key.device)
                ], dim=-1)
            elif seq_len < prev_len:
                # 异常截断
                layer_past.h2o_scores = layer_past.h2o_scores[..., :seq_len]

        kept_indices = None
        
        if dynamic_heavy == 0:
            kept_indices = torch.arange(seq_len - fixed_recent, seq_len, device=key.device).expand(bsz, -1)
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_size)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

            if not hasattr(layer_past, "h2o_scores"):
                layer_past.h2o_scores = torch.zeros((bsz, num_attention_heads, seq_len), dtype=torch.float32, device=key.device)
                layer_past.h2o_scores[:, :, :seq_len] += attn_weights.squeeze(-2)
            else:
                layer_past.h2o_scores += attn_weights.squeeze(-2)

            # 选择 Heavy Hitters
            total_scores = layer_past.h2o_scores.sum(dim=1).clone()
            
            # 保护 Recent 区域
            total_scores[:, -fixed_recent:] = -float('inf')
            
            _, heavy_indices = torch.topk(total_scores, k=dynamic_heavy, dim=-1)
            recent_indices = torch.arange(seq_len - fixed_recent, seq_len, device=key.device).expand(bsz, -1)
            
            kept_indices = torch.cat([heavy_indices, recent_indices], dim=-1)
            kept_indices, _ = kept_indices.sort(dim=-1)

        # 执行裁剪
        if kept_indices is not None:
            def gather_kv(tensor, idx):
                idx_expanded = idx.unsqueeze(1).unsqueeze(-1).expand(-1, tensor.size(1), -1, tensor.size(3))
                return torch.gather(tensor, 2, idx_expanded)
            
            pruned_key = gather_kv(key, kept_indices)
            pruned_value = gather_kv(value, kept_indices)
            
            # 同步裁剪 h2o_scores
            if hasattr(layer_past, "h2o_scores"):
                if kept_indices.device != layer_past.h2o_scores.device:
                    kept_indices = kept_indices.to(layer_past.h2o_scores.device)
                    
                idx_scores = kept_indices.unsqueeze(1).expand(-1, num_attention_heads, -1)
                layer_past.h2o_scores = torch.gather(layer_past.h2o_scores, 2, idx_scores)

            if isinstance(layer_past, (DynamicCache, Cache)):
                if hasattr(layer_past, "key_cache") and len(layer_past.key_cache) > self.layer_idx:
                    layer_past.key_cache[self.layer_idx] = pruned_key
                    layer_past.value_cache[self.layer_idx] = pruned_value
            
            key = pruned_key
            value = pruned_value

    kv_size_bytes = key.element_size() * key.nelement() + value.element_size() * value.nelement()
    global LAYER_KV_SIZES
    LAYER_KV_SIZES[self.layer_idx] = kv_size_bytes

    is_causal_masking = (q_len > 1)
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=is_causal_masking 
    )
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, hidden_size)
    attn_output = self.dense(attn_output)
    return (attn_output, (key, value) if output_attentions else None)

def enable_h2o_monkey_patch(model):
    print(f"\n>>> [System] Injecting H2O Logic...")
    for layer in model.gpt_neox.layers:
        import types
        layer.attention.forward = types.MethodType(h2o_gpt_neox_attention_forward, layer.attention)

def get_real_long_text(min_tokens=2048):
    """
    获取足够长的文本用于测试
    Args:
        min_tokens: 至少需要的 token 数量 (默认 4096)
    """
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        long_text = ""
        for item in dataset["text"]:
            long_text += item
            if len(long_text) / 4 > min_tokens * 1.5: 
                break
                
        return long_text
    except Exception as e:
        print(f"Dataset load failed: {e}, using dummy text.")
        return "The history of artificial intelligence " * 500  

def run_benchmark(model, tokenizer, text, exp_config):
    """
    统一运行：显存、TPOT、PPL
    """
    global ENABLE_KV_COMPRESSION, H2O_RECENT_SIZE, H2O_HEAVY_SIZE, LAYER_KV_SIZES, LAYER_BUDGETS
    
    # 设置参数
    ENABLE_KV_COMPRESSION = exp_config["compress"]
    H2O_RECENT_SIZE = exp_config.get("r", 64)
    H2O_HEAVY_SIZE = exp_config.get("h", 64)
    INVERSE = exp_config.get("inverse",False)
    LAYER_BUDGETS = {}

    # 初始化RankKV
    if exp_config.get("method") == "RankKV":
        target_avg_budget = H2O_RECENT_SIZE + H2O_HEAVY_SIZE
        num_layers = len(model.gpt_neox.layers)
        
        # A. 分析 Ranks
        ranks = profile_model_ranks(model, tokenizer, text, model.device)
        
        # B. 分配 Budget
        LAYER_BUDGETS = allocate_budgets(
            ranks, 
            total_avg_budget=target_avg_budget, 
            num_layers=num_layers,
            min_budget=32,
            alpha=0.3,
            inverse=INVERSE  
        )
        
        print(f">>> [RankKV] Initialized. Average Budget: {target_avg_budget}")
    
    # 速度 & 显存测试
    print(f"   [Running] {exp_config['name']} - Speed & Memory...")
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] > 100:
        inputs.input_ids = inputs.input_ids[:, :100]
        inputs.attention_mask = inputs.attention_mask[:, :100]
        
    LAYER_KV_SIZES = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 计时开始
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            min_new_tokens=1024,
            max_new_tokens=1024, 
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    tpot = (total_time / 1024) * 1000 
    throughput = 1024 / total_time
    effective_kv_gb = sum(LAYER_KV_SIZES.values()) / 1024**3
    
    # PPL 测试
    print(f"   [Running] {exp_config['name']} - PPL...")
    torch.cuda.empty_cache()
    
    eval_len = 2048
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)[:, :eval_len]
    
    nlls = []
    past_key_values = None
    prev_token = input_ids[:, 0:1]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    with torch.no_grad():
        for i in tqdm(range(1, input_ids.size(1)), leave=False):
            target_token = input_ids[:, i:i+1]
            outputs = model(prev_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            loss = loss_fct(logits, target_token.view(-1))
            nlls.append(loss)
            prev_token = target_token
            
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    
    return {
        "Experiment": exp_config["name"],
        "Budget (Tokens)": H2O_RECENT_SIZE + H2O_HEAVY_SIZE if ENABLE_KV_COMPRESSION else "Full",
        "KV Size (GB)": effective_kv_gb,
        "TPOT (ms)": tpot,
        "Throughput (T/s)": throughput,
        "PPL": ppl
    }

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=CACHE_DIR)
    enable_h2o_monkey_patch(model)
    long_text = get_real_long_text()
    input_ids = tokenizer(long_text, return_tensors="pt").input_ids
    print(f"Current Text Token Length: {input_ids.shape[1]}")
    
    budget_levels = [64, 128, 256, 512]
    results = []
    for budget in budget_levels:
        r = budget // 2
        h = budget - r
        
        # --- 1. Run H2O Baseline ---
        exp_h2o = {
            "name": f"H2O (B={budget})",
            "compress": True,
            "method": "H2O",
            "r": r, 
            "h": h,
            "inverse": False
        }
        print(f"--- Running {exp_h2o['name']} ---")
        res_h2o = run_benchmark(model, tokenizer, long_text, exp_h2o)
        results.append(res_h2o)

        # --- 2. Run RankKV (Inverse Strategy) ---
        exp_rankkv = {
            "name": f"RankKV (B={budget})",
            "compress": True,
            "method": "RankKV",
            "r": r, 
            "h": h,
            "inverse": False
        }
        print(f"--- Running {exp_rankkv['name']} ---")
        
        res_rankkv = run_benchmark(model, tokenizer, long_text, exp_rankkv)
        results.append(res_rankkv)

    # Run Dense (Upper Bound)
    exp_dense = {"name": "Dense (Full)", "compress": False, "r": 0, "h": 0}
    results.append(run_benchmark(model, tokenizer, long_text, exp_dense))

    # 输出结果
    df = pd.DataFrame(results)
    print("\n================ ROBUSTNESS CURVE RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv("direct_results_half_r.csv", index=False)