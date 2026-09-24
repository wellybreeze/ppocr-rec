from __future__ import annotations

from collections import defaultdict
from typing import Callable

DEFAULT_CALLBACKS = {
    "on_pretrain_routine_start": [],
    "on_pretrain_routine_end": [],
    "on_train_start": [],
    "on_train_epoch_start": [],
    "on_train_batch_start": [],
    "on_train_batch_end": [],
    "on_train_epoch_end": [],
    "on_train_end": [],
    "on_val_start": [],
    "on_val_batch_start": [],
    "on_val_batch_end": [],
    "on_val_end": [],
    "on_predict_start": [],
    "on_predict_end": [],
    "on_export_start": [],
    "on_export_end": [],
}


def get_default_callbacks() -> dict[str, list]:
    return {k: list(v) for k, v in DEFAULT_CALLBACKS.items()}


def add_integration_callbacks(instance) -> None:
    return


class CallbackRunner:
    def __init__(self, callbacks: dict | None = None):
        self.callbacks = callbacks or get_default_callbacks()

    def run_callbacks(self, event: str) -> None:
        for fn in self.callbacks.get(event, []):
            fn(self)

    def add_callback(self, event: str, func: Callable) -> None:
        self.callbacks.setdefault(event, []).append(func)
