from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ppocr_rec.data.augment import apply_rec_aug, rec_con_aug, resize_norm_img
from ppocr_rec.data.charset import Charset


class RecDataset(Dataset):
    """PaddleOCR-style rec dataset: `relpath<TAB>text` lines."""

    def __init__(
        self,
        img_root: str | Path,
        label_file: str | Path,
        charset: Charset,
        img_h: int = 48,
        img_w: int = 320,
        delimiter: str = "\t",
        augment: bool = False,
        encode_gtc: str | None = None,
        aug_type: str = "rec",
        con_aug: bool = False,
        con_aug_num: int = 2,
        con_aug_prob: float = 0.5,
        padding: bool = True,
        fraction: float = 1.0,
        seed: int = 0,
        aug_kwargs: dict | None = None,
    ):
        self.img_root = Path(img_root)
        self.charset = charset
        self.img_h = img_h
        self.img_w = img_w
        self.delimiter = delimiter
        self.augment = augment
        self.encode_gtc = encode_gtc
        self.aug_type = aug_type
        self.aug_kwargs = dict(aug_kwargs or {})
        self.do_rec_aug = bool(augment) and bool(self.aug_kwargs.get("rec_aug", True))
        self.con_aug = bool(con_aug) and bool(augment)
        self.con_aug_num = con_aug_num
        self.con_aug_prob = float(self.aug_kwargs.get("con_aug_prob", con_aug_prob))
        self.padding = padding
        self.samples: list[tuple[str, str]] = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                parts = line.split(delimiter, 1)
                if len(parts) != 2:
                    continue
                self.samples.append((parts[0], parts[1]))
        frac = float(fraction)
        if 0 < frac < 1 and self.samples:
            n = max(1, int(round(len(self.samples) * frac)))
            rng = random.Random(int(seed))
            idx = list(range(len(self.samples)))
            rng.shuffle(idx)
            self.samples = [self.samples[i] for i in idx[:n]]

    def __len__(self):
        return len(self.samples)

    def _read(self, idx: int) -> tuple[np.ndarray, str, Path]:
        rel, text = self.samples[idx]
        path = self.img_root / rel
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            text = ""
        return img, text, path

    def __getitem__(self, idx):
        img_h, img_w = self.img_h, self.img_w
        if isinstance(idx, tuple):
            idx, img_h, img_w = int(idx[0]), int(idx[1]), int(idx[2])
        img, text, path = self._read(idx)
        if self.con_aug:
            extras = []
            for _ in range(self.con_aug_num):
                j = random.randint(0, max(len(self.samples) - 1, 0))
                extra_img, extra_text, _ = self._read(j)
                extras.append((extra_img, extra_text))
            img, text = rec_con_aug(img, text, extras, img_h, img_w, self.charset.max_text_length, self.con_aug_prob)
        if self.do_rec_aug:
            img = apply_rec_aug(img, self.aug_type, self.aug_kwargs)
        image, valid_ratio = resize_norm_img(img, img_h, img_w, padding=self.padding)
        ctc_ids = self.charset.encode_ctc(text)
        if ctc_ids is None:
            ctc_ids = []
        label = self.charset.pad_ctc(ctc_ids)
        length = np.array(max(len(ctc_ids), 1), dtype=np.int64)
        item = {
            "img": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "length": torch.from_numpy(length),
            "valid_ratio": torch.tensor(valid_ratio, dtype=torch.float32),
            "text": text,
            "path": str(path),
        }
        if self.encode_gtc == "nrtr":
            gtc = self.charset.encode_nrtr(text)
            if gtc is None:
                gtc = np.zeros((self.charset.max_text_length,), dtype=np.int64)
            item["label_gtc"] = torch.from_numpy(gtc)
        if self.encode_gtc == "sar":
            sar = self.charset.encode_sar(text)
            if sar is None:
                sar = np.full((self.charset.max_text_length,), self.charset.sar_padding_idx, dtype=np.int64)
            item["label_sar"] = torch.from_numpy(sar)
        return item


def rec_collate(batch: list[dict]) -> dict:
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, 0)
        else:
            out[k] = vals
    return out
