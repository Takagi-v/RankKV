import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
import math
from rankkv_h2o.config import kv_state

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

    # Update Cache (Standard HF Logic)
    if layer_past is not None:
        key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs={"cache_position": cache_position})

    seq_len = key.shape[2]

    # --- RankKV Budget Logic ---
    if self.layer_idx in kv_state.layer_budgets:
        layer_budget = kv_state.layer_budgets[self.layer_idx]
    else:
        layer_budget = kv_state.recent_size + kv_state.heavy_size
    
    current_limit = layer_budget

    # Core Compress Logic
    if kv_state.enable_compression and q_len == 1 and seq_len > current_limit:
        
        fixed_recent = min(kv_state.recent_size, current_limit)
        dynamic_heavy = max(0, current_limit - fixed_recent)
        
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
                # Resize check in case cache grew
                if layer_past.h2o_scores.shape[-1] < seq_len:
                    diff = seq_len - layer_past.h2o_scores.shape[-1]
                    layer_past.h2o_scores = torch.cat([
                        layer_past.h2o_scores, 
                        torch.zeros((bsz, num_attention_heads, diff), device=key.device)
                    ], dim=-1)
                
                layer_past.h2o_scores[..., :seq_len] += attn_weights.squeeze(-2)
            
            total_scores = layer_past.h2o_scores[..., :seq_len].sum(dim=1).clone()
            total_scores[:, -fixed_recent:] = -float('inf')
            
            _, heavy_indices = torch.topk(total_scores, k=dynamic_heavy, dim=-1)
            recent_indices = torch.arange(seq_len - fixed_recent, seq_len, device=key.device).expand(bsz, -1)
            kept_indices = torch.cat([heavy_indices, recent_indices], dim=-1)
            kept_indices, _ = kept_indices.sort(dim=-1)
            kept_indices = kept_indices.clamp(min=0, max=seq_len - 1)

        if kept_indices is not None:
             # Helper to gather
            def gather_kv(tensor, idx):
                idx_expanded = idx.unsqueeze(1).unsqueeze(-1).expand(-1, tensor.size(1), -1, tensor.size(3))
                return torch.gather(tensor, 2, idx_expanded)
            
            pruned_key = gather_kv(key, kept_indices)
            pruned_value = gather_kv(value, kept_indices)
            
            # Update H2O Scores for next step
            if hasattr(layer_past, "h2o_scores"):
                if kept_indices.device != layer_past.h2o_scores.device:
                    kept_indices = kept_indices.to(layer_past.h2o_scores.device)

                # Safe check
                current_score_len = layer_past.h2o_scores.shape[-1]
                kept_indices = kept_indices.clamp(max=current_score_len - 1)
                
                idx_scores = kept_indices.unsqueeze(1).expand(-1, num_attention_heads, -1)
                layer_past.h2o_scores = torch.gather(layer_past.h2o_scores, 2, idx_scores)

            # Update DynamicCache explicitly (Crucial for next steps)
            if isinstance(layer_past, (DynamicCache, Cache)):
                if hasattr(layer_past, "key_cache") and len(layer_past.key_cache) > self.layer_idx:
                    layer_past.key_cache[self.layer_idx] = pruned_key
                    layer_past.value_cache[self.layer_idx] = pruned_value
            
            key = pruned_key
            value = pruned_value

    # Statistics
    kv_size_bytes = key.element_size() * key.nelement() + value.element_size() * value.nelement()
    kv_state.layer_kv_sizes[self.layer_idx] = kv_size_bytes

    # Attention Calculation
    is_causal_masking = (q_len > 1)
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=is_causal_masking 
    )
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, hidden_size)
    attn_output = self.dense(attn_output)

    return (attn_output, (key, value))