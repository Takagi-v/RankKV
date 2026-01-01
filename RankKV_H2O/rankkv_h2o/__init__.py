# rankkv_h2o/__init__.py

from . import config
from .patch import enable_h2o_monkey_patch
from .modeling import h2o_gpt_neox_attention_forward
from .utils import run_benchmark, get_real_long_text
from .strategy import (
    compute_effective_rank,
    profile_model_ranks,
    allocate_budgets
)

__all__ = [
    "config",
    "enable_h2o_monkey_patch",
    "run_benchmark",
    "get_real_long_text",
    "profile_model_ranks",
    "allocate_budgets",
]

"""
RankKV with H2O Integration Package
===================================

This package implements the RankKV compression strategy integrated with H2O 
eviction logic for GPT-NeoX models.

Modules:
    - config: Global configuration and state variables (e.g., budget settings).
    - patch: Utilities to inject custom attention logic into HuggingFace models.
    - modeling: Modified attention forward pass supporting KV cache compression.
    - strategy: Core RankKV logic for calculating effective ranks and allocating budgets.
    - utils: Benchmark utilities and data loading helpers.

Usage:
    >>> from rankkv_h2o import enable_h2o_monkey_patch, run_benchmark
    >>> model = ... # Load model
    >>> enable_h2o_monkey_patch(model) # Inject logic
    >>> run_benchmark(model, tokenizer, text, exp_config)
"""