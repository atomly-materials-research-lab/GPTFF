from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def create_data_loader_generators(seed: int) -> dict[str, torch.Generator]:
    return {
        "train": torch.Generator().manual_seed(seed),
        "validation": torch.Generator().manual_seed(seed + 1),
        "test": torch.Generator().manual_seed(seed + 2),
    }


def seed_data_loader_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
