"""
utils/seed.py — deterministic seeding for reproducible experiments.

Usage:
    from utils.seed import set_seed
    set_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices).

    PyTorch is imported lazily so this function works even before the full
    ML environment is installed (e.g. during initial setup / smoke tests).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch  # lazy — torch may not be installed during early setup

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # torch < 1.8

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        pass  # NumPy / random seeding already applied above
