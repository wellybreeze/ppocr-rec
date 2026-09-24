from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class CTCLoss(torch.nn.Module):
    def __init__(self, blank: int = 0, use_focal_loss: bool = False):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, dict):
            predicts = predicts["ctc"]
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        log_probs = F.log_softmax(predicts, dim=2).transpose(0, 1)  # T,B,C
        t, b, _ = log_probs.shape
        preds_lengths = torch.full((b,), t, dtype=torch.long, device=log_probs.device)
        labels = batch["label"].to(dtype=torch.long)
        label_lengths = batch["length"].to(dtype=torch.long)
        loss = self.loss_func(log_probs, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = (1.0 - torch.exp(-loss)).square()
            loss = loss * weight
        loss = loss.mean()
        return {"loss": loss}


class NRTRLoss(torch.nn.Module):
    def __init__(self, smoothing: bool = True, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, pred, batch):
        max_len = int(batch["length"].max().item())
        tgt = batch["label_gtc"][:, 1 : 2 + max_len]
        pred = pred.reshape(-1, pred.shape[2])
        tgt = tgt.reshape(-1)
        if self.smoothing:
            eps = 0.1
            n_class = pred.shape[1]
            one_hot = F.one_hot(tgt.clamp(min=0), n_class).float()
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / max(n_class - 1, 1)
            log_prb = F.log_softmax(pred, dim=1)
            non_pad = tgt != 0
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss[non_pad].mean() if non_pad.any() else pred.new_zeros(())
        else:
            loss = self.ce(pred, tgt)
        return {"loss": loss}


class SARLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = 92):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, predicts, batch):
        predict = predicts[:, :-1, :]
        label = batch["label_sar"][:, 1:]
        loss = self.loss_func(predict.reshape(-1, predict.shape[-1]), label.reshape(-1).long())
        return {"loss": loss}


class MultiLoss(torch.nn.Module):
    def __init__(self, loss_config_list=None, weight_1: float = 1.0, weight_2: float = 1.0, **kwargs):
        super().__init__()
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.ctc = CTCLoss()
        self.nrtr = NRTRLoss()
        self.sar = SARLoss(ignore_index=kwargs.get("ignore_index", 92))
        names = []
        for item in loss_config_list or [{"CTCLoss": None}, {"NRTRLoss": None}]:
            names.append(next(iter(item.keys())))
        self.names = names

    def forward(self, predicts, batch):
        total = 0.0
        out = {}
        if "CTCLoss" in self.names:
            loss = self.ctc(predicts["ctc"], batch)["loss"] * self.weight_1
            out["CTCLoss"] = loss
            total = total + loss
        if "NRTRLoss" in self.names:
            loss = self.nrtr(predicts["gtc"], batch)["loss"] * self.weight_2
            out["NRTRLoss"] = loss
            total = total + loss
        if "SARLoss" in self.names:
            loss = self.sar(predicts["sar"], batch)["loss"] * self.weight_2
            out["SARLoss"] = loss
            total = total + loss
        out["loss"] = total
        return out


def build_loss(name: str, **kwargs):
    mapping = {"CTCLoss": CTCLoss, "NRTRLoss": NRTRLoss, "SARLoss": SARLoss, "MultiLoss": MultiLoss}
    if name not in mapping:
        raise KeyError(f"Unknown loss {name}")
    return mapping[name](**kwargs)
