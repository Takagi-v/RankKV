import torch

# ================= PATH configs =================
MODEL_ID = "EleutherAI/pythia-2.8b"
CACHE_DIR = "../model"
OUTPUT_DIR = "../results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================  Global State  =================
ENABLE_KV_COMPRESSION = False
H2O_RECENT_SIZE = 64
H2O_HEAVY_SIZE = 64
LAYER_KV_SIZES = {} 
LAYER_BUDGETS = {}      # record Budget distribution