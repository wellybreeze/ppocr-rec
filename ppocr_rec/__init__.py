from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CFG_DIR = PACKAGE_ROOT / "cfg"
DEFAULT_CFG_PATH = CFG_DIR / "default.yaml"
DICT_DIR = CFG_DIR / "dicts"
MODELS_DIR = CFG_DIR / "models"

__all__ = ["OCR", "PACKAGE_ROOT", "CFG_DIR", "DEFAULT_CFG_PATH"]


def __getattr__(name: str):
    if name == "OCR":
        from ppocr_rec.models.rec.model import OCR

        return OCR
    raise AttributeError(name)
