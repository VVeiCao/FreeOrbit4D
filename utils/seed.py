"""Seed helpers for reproducible pipeline runs."""

from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np


def resolve_seed(config: Any, stage_path: Optional[str] = None, default: int = 23) -> int:
    """Resolve a stage seed, falling back to the pipeline-wide seed.

    If ``stage_path`` points to ``None`` or a missing value, ``common.seed`` is
    used. This keeps one seed in normal scene configs while still allowing a
    stage-specific override when needed.
    """
    value = None
    if stage_path:
        value = config.get(stage_path, None)
    if value is None:
        value = config.get("common.seed", default)
    return int(value)


def set_global_seed(seed: int, deterministic: bool = False) -> int:
    """Set Python, NumPy, and PyTorch random seeds.

    ``PYTHONHASHSEED`` affects child Python processes. For the current process,
    Python reads it at startup, but setting it here still ensures subprocesses
    spawned later inherit the same value.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    return seed
