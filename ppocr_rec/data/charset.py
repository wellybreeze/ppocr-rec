from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Charset:
    """Character table with CTC / NRTR / SAR encodings aligned to PaddleOCR."""

    def __init__(
        self,
        dict_path: str | Path | None = None,
        use_space_char: bool = True,
        max_text_length: int = 25,
        chars: list[str] | None = None,
    ):
        self.max_text_length = max_text_length
        self.use_space_char = use_space_char
        from_table = chars is not None or dict_path is not None
        if chars is not None:
            chars = list(chars)
        elif dict_path is None:
            chars = list("0123456789abcdefghijklmnopqrstuvwxyz")
        else:
            chars = []
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    ch = line.strip("\n").strip("\r")
                    if ch != "":
                        chars.append(ch)
        if from_table and use_space_char and " " not in chars:
            chars.append(" ")
        self.raw_chars = chars

        self.ctc_chars = ["blank"] + chars
        self.ctc_dict = {c: i for i, c in enumerate(self.ctc_chars)}

        self.nrtr_chars = ["blank", "<unk>", "<s>", "</s>"] + chars
        self.nrtr_dict = {c: i for i, c in enumerate(self.nrtr_chars)}

        self.sar_chars = chars + ["<UKN>", "<BOS/EOS>", "<PAD>"]
        self.sar_unknown_idx = len(chars)
        self.sar_start_idx = len(chars) + 1
        self.sar_end_idx = self.sar_start_idx
        self.sar_padding_idx = len(chars) + 2
        self.sar_dict = {c: i for i, c in enumerate(chars)}

    @property
    def ctc_num(self) -> int:
        return len(self.ctc_chars)

    @property
    def nrtr_num(self) -> int:
        return len(self.nrtr_chars)

    @property
    def sar_num(self) -> int:
        return len(self.sar_chars)

    def encode_ctc(self, text: str) -> list[int] | None:
        ids = []
        for ch in text:
            if ch not in self.ctc_dict:
                return None
            ids.append(self.ctc_dict[ch])
        if not ids or len(ids) > self.max_text_length:
            return None
        return ids

    def pad_ctc(self, ids: list[int]) -> np.ndarray:
        out = ids + [0] * (self.max_text_length - len(ids))
        return np.array(out, dtype=np.int64)

    def encode_nrtr(self, text: str) -> np.ndarray | None:
        ids = []
        for ch in text:
            ids.append(self.nrtr_dict.get(ch, 1))  # <unk>
        if not ids or len(ids) >= self.max_text_length - 1:
            return None
        ids = [2] + ids + [3]  # <s> ... </s>
        ids = ids + [0] * (self.max_text_length - len(ids))
        return np.array(ids, dtype=np.int64)

    def encode_sar(self, text: str) -> np.ndarray | None:
        ids = []
        for ch in text:
            ids.append(self.sar_dict.get(ch, self.sar_unknown_idx))
        if not ids or len(ids) >= self.max_text_length - 1:
            return None
        target = [self.sar_start_idx] + ids + [self.sar_end_idx]
        padded = [self.sar_padding_idx] * self.max_text_length
        padded[: len(target)] = target
        return np.array(padded, dtype=np.int64)


def ctc_decode_ids(ids, probs, charset: Charset, remove_duplicate: bool = True) -> tuple[str, float]:
    text_chars = []
    confs = []
    prev = -1
    for i, idx in enumerate(ids):
        idx = int(idx)
        if idx == 0:
            prev = idx
            continue
        if remove_duplicate and idx == prev:
            prev = idx
            continue
        if idx < len(charset.ctc_chars):
            ch = charset.ctc_chars[idx]
            if ch != "blank":
                text_chars.append(ch)
                if probs is not None:
                    confs.append(float(probs[i]))
        prev = idx
    conf = float(np.mean(confs)) if confs else 0.0
    return "".join(text_chars), conf
