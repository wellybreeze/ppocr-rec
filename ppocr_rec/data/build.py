from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from ppocr_rec import DICT_DIR
from ppocr_rec.data.charset import Charset
from ppocr_rec.data.dataset import RecDataset, rec_collate


def load_rec_data_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data["path"])
    if not root.is_absolute():
        root = (Path(path).parent / root).resolve()
    data["path"] = root
    dict_path = data.get("dict")
    if dict_path:
        dp = Path(dict_path)
        if not dp.is_absolute() and not dp.exists():
            cand = DICT_DIR / dp.name
            if cand.exists():
                dp = cand
            else:
                dp = root / dp
        data["dict"] = str(dp)
    return data


def build_charset(data_cfg: dict, args) -> Charset:
    dict_path = data_cfg.get("dict") or getattr(args, "character_dict", None)
    return Charset(
        dict_path=dict_path,
        use_space_char=data_cfg.get("use_space_char", getattr(args, "use_space_char", True)),
        max_text_length=getattr(args, "max_text_length", 25),
    )


class MultiScaleBatchSampler(Sampler):
    """Same-scale batches, matching PaddleOCR MultiScaleSampler (ds_width=false).

    `scales` uses Paddle layout `[[W, H], ...]`, e.g. `[[320, 32], [320, 48], [320, 64]]`.
    Each yielded batch item is `(index, H, W)` for RecDataset.
    """

    def __init__(self, n: int, batch_size: int, scales: list, drop_last: bool = True, shuffle: bool = True):
        self.n = n
        self.batch_size = max(int(batch_size), 1)
        self.hw = [(int(s[1]), int(s[0])) for s in scales]  # (H, W)
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = list(range(self.n))
        if self.shuffle:
            random.shuffle(ids)
        i = 0
        while i < self.n:
            h, w = random.choice(self.hw)
            chunk = ids[i : i + self.batch_size]
            if len(chunk) < self.batch_size:
                if self.drop_last:
                    break
            yield [(idx, h, w) for idx in chunk]
            i += self.batch_size

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return math.ceil(self.n / self.batch_size)


# Paddle RecAug / RecConAug / SVTRRecAug defaults. gray/bgr2rgb are extras (off).
REC_AUG_DEFAULTS = {
    "tia_prob": 0.4,
    "crop_prob": 0.4,
    "reverse_prob": 0.4,
    "noise_prob": 0.4,
    "jitter_prob": 0.4,
    "blur_prob": 0.4,
    "hsv_aug_prob": 0.4,
    "gray_prob": 0.0,
    "bgr2rgb_prob": 0.0,
    "con_aug_prob": 0.5,
    "geometry_p": 0.5,
    "deterioration_p": 0.25,
    "colorjitter_p": 0.25,
}


def infer_aug_cfg(model_yaml: dict) -> dict:
    """Derive RecAug / RecConAug / multi-scale from model yaml + algorithm."""
    aug = dict(model_yaml.get("aug") or {})
    arch = model_yaml.get("Architecture", model_yaml)
    algo = arch.get("algorithm") or model_yaml.get("algorithm", "CRNN")
    head = arch.get("Head") or {}
    names = [next(iter(x.keys())) for x in head.get("head_list", [])] if head.get("name") == "MultiHead" else []
    is_v3 = "SARHead" in names
    out = {
        "type": aug.get("type") or ("svtr" if algo == "SVTR" else "rec"),
        "con_aug": aug.get("con_aug"),
        "con_aug_num": int(aug.get("con_aug_num", 2)),
        "multi_scale": aug.get("multi_scale"),
        "padding": aug.get("padding", True),
        "rec_aug": True,
        **REC_AUG_DEFAULTS,
    }
    if out["con_aug"] is None:
        out["con_aug"] = head.get("name") == "MultiHead" or algo in {"SVTR_LCNet", "SVTR_HGNet"}
        if algo == "SVTR":
            out["con_aug"] = False
    if out["multi_scale"] is None and algo in {"SVTR_LCNet", "SVTR_HGNet"} and not is_v3:
        out["multi_scale"] = [[320, 32], [320, 48], [320, 64]]
    if algo == "SVTR" and "padding" not in (model_yaml.get("aug") or {}):
        out["padding"] = False
    return out


def overlay_aug_args(aug: dict, args) -> dict:
    """Apply default.yaml / CLI aug knobs. Empty `con_aug` / `multi_scale` keep recipe."""
    out = dict(aug)
    rec = getattr(args, "rec_aug", None)
    if rec is not None:
        out["rec_aug"] = bool(rec)
    if getattr(args, "con_aug", None) is not None:
        out["con_aug"] = bool(args.con_aug)
    if getattr(args, "con_aug_num", None) not in (None, ""):
        out["con_aug_num"] = int(args.con_aug_num)
    ms = getattr(args, "multi_scale", None)
    if ms:
        out["multi_scale"] = ms
    for k, default in REC_AUG_DEFAULTS.items():
        v = getattr(args, k, None)
        out[k] = default if v in (None, "") else float(v)
    return out


def build_dataloader(
    dataset: RecDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    drop_last: bool = False,
    scales=None,
):
    kwargs = dict(
        num_workers=workers,
        collate_fn=rec_collate,
        pin_memory=torch.cuda.is_available(),
    )
    if scales and shuffle:
        sampler = MultiScaleBatchSampler(len(dataset), batch_size, scales, drop_last=drop_last, shuffle=True)
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
