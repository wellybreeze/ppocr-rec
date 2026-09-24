from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger("ppocr_rec")


def select_device(device: str | int | list | None = None, batch: int = 0) -> torch.device:
    if device is None or device == "" or device == "None":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = str(device).strip()
    if device in {"-1", "cpu"}:
        return torch.device("cpu")
    if device.startswith("cuda") or device.isdigit():
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available, using CPU")
            return torch.device("cpu")
        idx = device.split(":")[-1] if ":" in device else device
        return torch.device(f"cuda:{idx}" if idx.isdigit() else "cuda:0")
    return torch.device(device)


def init_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def de_parallel(model: torch.nn.Module) -> torch.nn.Module:
    return unwrap_model(model)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.ema = copy_model(unwrap_model(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay
        msd = unwrap_model(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].detach() * (1.0 - d))


def copy_model(model: torch.nn.Module) -> torch.nn.Module:
    import copy

    return copy.deepcopy(model)


def increment_path(path: str | Path, exist_ok: bool = False) -> Path:
    path = Path(path)
    if exist_ok or not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    i = 2
    while path.with_name(f"{path.name}{i}").exists():
        i += 1
    p = path.with_name(f"{path.name}{i}")
    p.mkdir(parents=True, exist_ok=True)
    return p
