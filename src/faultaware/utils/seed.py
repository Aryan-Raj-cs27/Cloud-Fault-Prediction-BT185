"""Global seeding for reproducibility (CLAUDE.md §6.3)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every source of randomness used in the project.

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, ask PyTorch/cuDNN for deterministic kernels.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_rng(seed: int) -> np.random.Generator:
    """Return an isolated NumPy generator (preferred over the global state).

    Args:
        seed: Seed for the generator.

    Returns:
        A ``numpy.random.Generator``.
    """
    return np.random.default_rng(seed)
