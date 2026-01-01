import types
from .modeling import h2o_gpt_neox_attention_forward

def enable_h2o_monkey_patch(model):
    print(f"\n>>> [System] Injecting H2O/RankKV Logic...")
    for layer in model.gpt_neox.layers:
        layer.attention.forward = types.MethodType(h2o_gpt_neox_attention_forward, layer.attention)