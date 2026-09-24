from ppocr_rec.nn.modules.heads.ctc_head import CTCHead
from ppocr_rec.nn.modules.heads.multi_head import MultiHead
from ppocr_rec.nn.modules.heads.nrtr_head import Transformer
from ppocr_rec.nn.modules.heads.sar_head import SARHead

HEADS = {"CTCHead": CTCHead, "MultiHead": MultiHead, "SARHead": SARHead, "Transformer": Transformer}


def register_head(name, cls):
    HEADS[name] = cls


def build_head(cfg: dict, in_channels: int):
    from copy import deepcopy

    cfg = deepcopy(cfg)
    name = cfg.pop("name")
    cfg["in_channels"] = in_channels
    if name not in HEADS:
        raise KeyError(f"Unknown head {name}. Registered: {list(HEADS)}")
    return HEADS[name](**cfg)
