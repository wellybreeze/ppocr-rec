from __future__ import annotations

import numpy as np
import torch

from ppocr_rec.engine.predictor import BasePredictor
from ppocr_rec.engine.results import RecResult
from ppocr_rec.utils.decode import CTCLabelDecode


class RecPredictor(BasePredictor):
    def setup_model(self, model):
        super().setup_model(model)
        charset = getattr(model, "charset", None)
        dict_path = getattr(self.args, "character_dict", None)
        if dict_path:
            from ppocr_rec.data.charset import Charset

            charset = Charset(
                dict_path,
                use_space_char=bool(getattr(self.args, "use_space_char", True)),
                max_text_length=int(getattr(self.args, "max_text_length", 25)),
            )
            if hasattr(model, "charset"):
                model.charset = charset
        self.decoder = CTCLabelDecode(charset)

    def postprocess(self, preds, path):
        if isinstance(preds, dict) and "text_ids" in preds:
            decoded = self._decode_exported_ids(preds)
        else:
            decoded = self.decoder(preds)
        if not isinstance(path, (list, tuple)):
            path = [path] * len(decoded)
        results = []
        for (text, conf), p in zip(decoded, path):
            results.append(RecResult(text=text, conf=float(conf), path=p))
        return results

    def _decode_exported_ids(self, preds: dict) -> list[tuple[str, float]]:
        ids = preds["text_ids"]
        confs = preds["text_confs"]
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        if isinstance(confs, torch.Tensor):
            confs = confs.detach().cpu().numpy()
        charset = self.decoder.charset
        out = []
        for i in range(ids.shape[0]):
            chars = []
            for tid in ids[i]:
                tid = int(tid)
                if tid < 0:
                    continue
                if tid < len(charset.ctc_chars):
                    ch = charset.ctc_chars[tid]
                    if ch != "blank":
                        chars.append(ch)
            conf = float(np.reshape(confs[i], (-1,))[0]) if confs is not None else 0.0
            out.append(("".join(chars), conf))
        return out
