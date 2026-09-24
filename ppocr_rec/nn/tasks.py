from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ppocr_rec.cfg import resolve_model_yaml, yaml_load
from ppocr_rec.nn.modules.backbones import build_backbone
from ppocr_rec.nn.modules.heads import build_head
from ppocr_rec.nn.modules.necks import build_neck
from ppocr_rec.utils.loss import build_loss


def yaml_model_load(path: str | Path) -> dict:
    path = resolve_model_yaml(path)
    cfg = yaml_load(path)
    cfg["_yaml_path"] = str(path)
    return cfg


class RecognitionModel(nn.Module):
    """PaddleOCR-style Transform → Backbone → Neck → Head."""

    def __init__(self, cfg: str | Path | dict, charset=None, ch: int = 3, verbose: bool = True):
        super().__init__()
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)
        self.yaml = cfg
        arch = cfg.get("Architecture", cfg)
        self.algorithm = arch.get("algorithm") or cfg.get("algorithm", "CRNN")
        in_channels = arch.get("in_channels", ch)

        self.use_transform = bool(arch.get("Transform"))
        self.transform = None
        if self.use_transform:
            raise NotImplementedError("Transform (TPS/STN) is not implemented in v1")

        bb_cfg = arch.get("Backbone")
        self.use_backbone = bool(bb_cfg)
        if self.use_backbone:
            self.backbone = build_backbone(bb_cfg, in_channels=in_channels)
            in_channels = self.backbone.out_channels
        else:
            self.backbone = None

        nk_cfg = arch.get("Neck")
        self.use_neck = bool(nk_cfg)
        if self.use_neck:
            self.neck = build_neck(nk_cfg, in_channels=in_channels)
            in_channels = self.neck.out_channels
        else:
            self.neck = None

        hd_cfg = deepcopy(arch.get("Head") or {})
        if charset is None:
            from ppocr_rec.data.charset import Charset

            charset = Charset(None, use_space_char=False)
        self._inject_out_channels(hd_cfg, charset)
        self.head = build_head(hd_cfg, in_channels=in_channels)
        self.criterion = None
        self.loss_name = cfg.get("loss", "CTCLoss")
        if arch.get("Head", {}).get("name") == "MultiHead":
            self.loss_name = "MultiLoss"
        self.charset = charset

    @staticmethod
    def _inject_out_channels(head_cfg: dict, charset) -> None:
        name = head_cfg.get("name", "CTCHead")
        if name == "MultiHead":
            head_cfg["out_channels_list"] = {
                "CTCLabelDecode": charset.ctc_num,
                "NRTRLabelDecode": charset.nrtr_num,
                "SARLabelDecode": charset.sar_num,
            }
        else:
            head_cfg["out_channels"] = charset.ctc_num

    def init_criterion(self):
        extra = {}
        if self.loss_name == "MultiLoss":
            arch = self.yaml.get("Architecture", self.yaml)
            gtc = None
            for item in arch.get("Head", {}).get("head_list", []) or []:
                gtc = next(iter(item.keys()), None)
            if gtc == "SARHead":
                extra["loss_config_list"] = [{"CTCLoss": None}, {"SARLoss": None}]
                extra["ignore_index"] = getattr(self.charset, "sar_padding_idx", 92)
            else:
                extra["loss_config_list"] = [{"CTCLoss": None}, {"NRTRLoss": None}]
        self.criterion = build_loss(self.loss_name, **extra)
        return self.criterion

    def predict(self, x: torch.Tensor):
        if self.use_backbone:
            x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        x = self.head(x)
        return x

    def loss(self, batch: dict, preds=None):
        if self.criterion is None:
            self.init_criterion()
        if preds is None:
            if getattr(self.head, "needs_targets", False):
                preds = self.forward_train(batch)
            else:
                preds = self.predict(batch["img"])
        return self.criterion(preds, batch)

    def forward_train(self, batch: dict):
        x = batch["img"]
        if self.use_backbone:
            x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        targets = None
        if getattr(self.head, "needs_targets", False):
            targets = [
                batch.get("label"),
                batch.get("label_gtc", batch.get("label_sar")),
                batch.get("length"),
                batch.get("valid_ratio"),
            ]
        return self.head(x, targets=targets)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def load(self, weights: str | Path | dict, strict: bool = False):
        if not isinstance(weights, dict):
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        else:
            ckpt = weights
        state = ckpt.get("model", ckpt)
        if isinstance(state, nn.Module):
            state = state.state_dict()
        own = self.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            k2 = k[7:] if k.startswith("module.") else k
            if k2 in own and own[k2].shape == v.shape:
                filtered[k2] = v
            else:
                skipped.append(k2)
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        return {"loaded": len(filtered), "skipped": skipped, "missing": list(missing)}
