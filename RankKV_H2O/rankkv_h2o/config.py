import torch
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

# ================= 1. Static System Configurations =================
@dataclass
class SystemConfig:
    MODEL_ID: str = "EleutherAI/pythia-2.8b"
    CACHE_DIR: str = "../model"
    OUTPUT_DIR: str = "../results"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def static_rank_path(self):
        return os.path.join(self.OUTPUT_DIR, "model_ranks_cache.json")

# ================= 2. Generation & Benchmark Settings =================
@dataclass
class GenerationConfig:
    calibration_len: int = 512     # Rank的截断长度
    benchmark_input_len: int = 512 # TPOT的输入截断长度
    eval_ppl_len: int = 2048       # PPL的评估长度
    
    min_new_tokens: int = 1024
    max_new_tokens: int = 1024
    
    dummy_text_fallback: bool = True

# ================= 3. Runtime Compression State (Singleton) =================
@dataclass
class CompressionState:
    """
    这是一个运行时状态对象。
    Benchmark 脚本会修改这里的值，Modeling 会读取这里的值。
    """
    enable_compression: bool = False
    rank_mode: str = "static"  # "dynamic" or "static"
    
    # Global Default Budgets (H2O Baseline Logic)
    recent_size: int = 64
    heavy_size: int = 64
    layer_budgets: Dict[int, int] = field(default_factory=dict)
    layer_kv_sizes: Dict[int, int] = field(default_factory=dict)

    def reset_stats(self):
        self.layer_kv_sizes = {}

    def set_budget(self, recent: int, heavy: int, layer_map: Optional[Dict[int, int]] = None):
        self.recent_size = recent
        self.heavy_size = heavy
        self.layer_budgets = layer_map if layer_map else {}

# ================= Global Instances =================
system_cfg = SystemConfig()
gen_cfg = GenerationConfig()
kv_state = CompressionState()