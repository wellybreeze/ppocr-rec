from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ppocr_rec.cfg import get_cfg, get_save_dir, resolve_model_yaml
from ppocr_rec.engine.exported import is_exported_path, load_exported
from ppocr_rec.nn.tasks import RecognitionModel, yaml_model_load
from ppocr_rec.utils.callbacks import get_default_callbacks
from ppocr_rec.utils.torch_utils import unwrap_model


class Model(torch.nn.Module):
    """Facade: train / val / predict / export via task_map."""

    def __init__(
        self,
        model: str | Path = "crnn.yaml",
        task: str | None = "rec",
        verbose: bool = False,
        character_dict: str | Path | None = None,
    ):
        super().__init__()
        self.callbacks = get_default_callbacks()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.ckpt: dict[str, Any] = {}
        self.cfg = None
        self.ckpt_path = None
        self.overrides: dict[str, Any] = {}
        self.metrics = None
        self.task = task or "rec"
        self.model_name = str(model)
        self.verbose = verbose
        model_s = str(model).strip()
        if model_s.endswith((".yaml", ".yml")) or (not Path(model_s).suffix):
            self._new(model_s, task=self.task, verbose=verbose)
        else:
            self._load(model_s, task=self.task, character_dict=character_dict)

    def _new(self, cfg: str, task=None, verbose=False):
        cfg_path = resolve_model_yaml(cfg)
        self.cfg = str(cfg_path)
        self.overrides["model"] = self.cfg
        self.overrides["task"] = task or "rec"
        self.model = self._smart_load("model")(cfg_path, verbose=verbose)
        self.task = "rec"

    def _yaml_defaults(self) -> dict[str, Any]:
        """Pull train/val/predict defaults (e.g. imgsz) from the model yaml."""
        out: dict[str, Any] = {}
        cfg = getattr(self, "cfg", None)
        if not cfg:
            return out
        try:
            y = yaml_model_load(cfg)
        except Exception:
            return out
        if y.get("imgsz") is not None:
            out["imgsz"] = y["imgsz"]
        return out

    def _load(self, weights: str, task=None, character_dict=None):
        if is_exported_path(weights):
            self.model = load_exported(weights, character_dict=character_dict)
            self.ckpt = {}
            self.ckpt_path = weights
            self.cfg = None
            self.overrides["imgsz"] = list(self.model.imgsz)
            self.overrides["model"] = str(weights)
            self.task = "rec"
            return
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.ckpt = ckpt if isinstance(ckpt, dict) else {"model": ckpt}
        self.ckpt_path = weights
        yaml_file = None
        if isinstance(self.ckpt, dict):
            yaml_file = self.ckpt.get("cfg") or self.ckpt.get("args", {}).get("model")
        charset = self.ckpt.get("charset") if isinstance(self.ckpt, dict) else None
        args = self.ckpt.get("args", {}) if isinstance(self.ckpt, dict) else {}
        if yaml_file:
            self.cfg = str(resolve_model_yaml(yaml_file)) if not Path(yaml_file).exists() else yaml_file
        else:
            self.cfg = str(resolve_model_yaml("crnn.yaml"))
        model_cls = self._smart_load("model")
        self.model = model_cls(self.cfg, charset=charset, verbose=False)
        state = self.ckpt.get("model", self.ckpt)
        self.model.load(state if not isinstance(state, RecognitionModel) else state.state_dict(), strict=False)
        self.overrides.update({k: args[k] for k in ("imgsz", "data", "task") if k in args})
        self.task = "rec"

    def _check_is_pytorch_model(self):
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("Model is not a PyTorch module")

    def _smart_load(self, key: str):
        return self.task_map[self.task][key]

    @property
    def task_map(self) -> dict:
        raise NotImplementedError

    def predict(self, source=None, **kwargs):
        self._check_is_pytorch_model()
        overrides = {**self._yaml_defaults(), **self.overrides, **kwargs, "mode": "predict"}
        predictor_cls = self._smart_load("predictor")
        self.predictor = predictor_cls(overrides=overrides)
        self.predictor.setup_model(model=self.model)
        return self.predictor(source)

    def __call__(self, source=None, **kwargs):
        return self.predict(source=source, **kwargs)

    def val(self, **kwargs):
        self._check_is_pytorch_model()
        overrides = {**self._yaml_defaults(), **self.overrides, **kwargs, "mode": "val"}
        validator = self._smart_load("validator")(overrides=overrides)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def train(self, **kwargs):
        self._check_is_pytorch_model()
        overrides = {**self._yaml_defaults(), **self.overrides, **kwargs, "mode": "train"}
        if self.cfg:
            overrides.setdefault("model", self.cfg)
        resume = overrides.get("resume")
        if resume in {True, "true", "True", "last"} and self.ckpt_path and str(self.ckpt_path).endswith(".pt"):
            overrides["resume"] = self.ckpt_path
        trainer_cls = self._smart_load("trainer")
        self.trainer = trainer_cls(overrides=overrides)
        self.trainer.train()
        if getattr(self.trainer, "best", None) and Path(self.trainer.best).exists():
            self._load(str(self.trainer.best))
        return getattr(self.trainer, "metrics", None)

    def export(self, **kwargs):
        self._check_is_pytorch_model()
        from ppocr_rec.engine.exported import ExportedRecModel
        from ppocr_rec.engine.exporter import Exporter

        if isinstance(self.model, ExportedRecModel):
            raise TypeError("Already an exported model; load a .pt or yaml to export.")
        overrides = {**self._yaml_defaults(), **self.overrides, **kwargs, "mode": "export"}
        return Exporter(overrides=overrides)(model=self.model)

    def load(self, weights: str):
        self.model.load(weights, strict=False)
        return self
