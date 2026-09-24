"""PyTorch wrappers so ONNX is exported in one shot (normalize + CTC)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecOnnxWrapper(nn.Module):
    """Optional in-graph (x-mean)/std and CTC greedy decode around a rec model."""

    def __init__(
        self,
        body: nn.Module,
        *,
        normalize: bool,
        mean: list[float],
        std: list[float],
        postprocess: bool,
        max_text_length: int,
        ignored_tokens: list[int],
        remove_duplicate: bool,
    ):
        super().__init__()
        self.body = body
        self.do_norm = bool(normalize)
        self.do_ctc = bool(postprocess)
        self.max_text_length = int(max_text_length)
        self.remove_duplicate = bool(remove_duplicate)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("ignored", torch.tensor(list(ignored_tokens or [0]), dtype=torch.long))

    def forward(self, x: torch.Tensor):
        if self.do_norm:
            x = (x - self.mean) / self.std
        preds = self.body(x)
        if isinstance(preds, dict):
            preds = preds.get("ctc", next(iter(preds.values())))
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        if not self.do_ctc:
            return preds
        return self._ctc(preds)

    def _ctc(self, preds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = preds.argmax(dim=2)
        prob = preds.max(dim=2).values
        mask = torch.ones_like(idx, dtype=torch.bool)
        if self.remove_duplicate:
            mask = mask & torch.cat((mask[:, :1], idx[:, 1:] != idx[:, :-1]), dim=1)
        if self.ignored.numel():
            mask = mask & ~idx.unsqueeze(-1).eq(self.ignored.view(1, 1, -1)).any(-1)
        mask_f = mask.to(dtype=prob.dtype)
        conf = (prob * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        pos = torch.cumsum(torch.ones_like(idx), dim=1) - 1
        keys = torch.where(mask, pos, pos + 1_000_000)
        pad = self.max_text_length
        order = F.pad(keys, (0, pad), value=2_000_000).topk(pad, dim=1, largest=False, sorted=True).indices
        gathered = F.pad(idx, (0, pad), value=-1).gather(1, order)
        gmask = F.pad(mask, (0, pad), value=False).gather(1, order)
        text_ids = torch.where(gmask, gathered, torch.full_like(gathered, -1)).to(torch.int32)
        return text_ids, conf


def set_onnx_meta(model, key: str, value: str) -> None:
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    prop = model.metadata_props.add()
    prop.key = key
    prop.value = value


def stamp_onnx(path: str, charset, extra: dict[str, str], simplify: bool) -> None:
    import onnx

    model = onnx.load(path)
    if charset is not None and getattr(charset, "raw_chars", None):
        set_onnx_meta(model, "character", "\n".join(charset.raw_chars))
    for k, v in extra.items():
        set_onnx_meta(model, k, str(v))
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify

            model, ok = onnx_simplify(model)
            if not ok:
                from ppocr_rec.utils.torch_utils import LOGGER

                LOGGER.warning("onnxsim could not validate, keeping unsimplified graph")
        except ImportError:
            from ppocr_rec.utils.torch_utils import LOGGER

            LOGGER.warning("onnxsim not installed, skip simplify (pip install onnxsim)")
    onnx.save(model, path)
