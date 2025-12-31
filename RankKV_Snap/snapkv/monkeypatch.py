
import transformers
from .gptneox_model import gptneox_attn_forward_SnapKV

def replace_gptneox(method):
    if method == "snapkv":
        print("Using SnapKV for GPT-NeoX!")
        try:
            transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention.forward = gptneox_attn_forward_SnapKV
        except AttributeError:
            pass
            
        try:
            transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXSdpaAttention.forward = gptneox_attn_forward_SnapKV
            print("Patched GPTNeoXSdpaAttention")
        except AttributeError:
            pass

        try:
            transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXFlashAttention2.forward = gptneox_attn_forward_SnapKV
            print("Patched GPTNeoXFlashAttention2")
        except AttributeError:
            pass
