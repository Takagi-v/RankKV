
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from .snapkv_utils import init_snapkv

def gptneox_attn_forward_SnapKV(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    # Initialize SnapKV
    init_snapkv(self)

    has_layer_past = layer_past is not None

    # Compute QKV
    qkv = self.query_key_value(hidden_states)
    
    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)    
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]
    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    max_pos_needed = position_ids.max().item() + 1
    current_seq_len = key.shape[-2]
    if has_layer_past:
        current_seq_len += layer_past[0].shape[-2]
        
    seq_len = max(max_pos_needed, current_seq_len)
    
    cos, sin = self.rotary_emb(value, seq_len=seq_len)
    query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)

    # SnapKV Logic
    if not has_layer_past:
        # Initial prompt processing
        num_key_value_groups = 1 # Assumption for Pythia/GPT-NeoX
        
        # update_kv returns the compressed keys/values
        key_compress, value_compress = self.kv_cluster.update_kv(
            key, query, value, attention_mask, num_key_value_groups
        )
        
        if self.layer_idx == 0 and torch.distributed.is_initialized() == False:
            print(f"SnapKV: Layer 0 compressed KV from {key.shape[-2]} to {key_compress.shape[-2]}")
        elif self.layer_idx == 0:
            pass 
        
        # We store the compressed version in cache
        present = (key_compress, value_compress) if use_cache else None
        
        # We use the FULL key/value for the current attention calculation
        # This is consistent with standard practice (prefill sees all)
        
    else:
        # Decoding step
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None
        
    # Attention calculation
    # Fix attention_mask if shapes mismatch due to compression
    if attention_mask is not None and key.shape[-2] != attention_mask.shape[-1]:
        attention_mask = torch.zeros(
            (attention_mask.shape[0], 1, 1, key.shape[-2]), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )

    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
    
    if hasattr(self, "_merge_heads"):
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
    else:
        bsz, num_heads, seq_len, head_dim = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)
    
    attn_output = self.dense(attn_output)
    attn_output = self.attention_dropout(attn_output)
    
    if output_attentions:
        return attn_output, present, attn_weights
    else:
        return attn_output, present
