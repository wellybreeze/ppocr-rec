from __future__ import annotations

import numpy as np
import torch


class CTCLabelDecode:
    def __init__(self, charset):
        self.charset = charset

    def __call__(self, preds, labels=None):
        if isinstance(preds, dict):
            preds = preds.get("ctc", preds)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        texts = []
        from ppocr_rec.data.charset import ctc_decode_ids

        for i in range(preds_idx.shape[0]):
            texts.append(ctc_decode_ids(preds_idx[i], preds_prob[i], self.charset))
        if labels is None:
            return texts
        gt = []
        for row in labels:
            if isinstance(row, torch.Tensor):
                row = row.detach().cpu().numpy()
            chars = []
            for idx in row:
                idx = int(idx)
                if idx <= 0:
                    continue
                if idx < len(self.charset.ctc_chars):
                    chars.append(self.charset.ctc_chars[idx])
            gt.append(("".join(chars), 1.0))
        return texts, gt
