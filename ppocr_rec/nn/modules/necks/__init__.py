from ppocr_rec.nn.modules.necks.rnn import Im2Seq, SequenceEncoder

NECKS = {"SequenceEncoder": SequenceEncoder, "Im2Seq": Im2Seq}


def build_neck(cfg: dict, in_channels: int):
    cfg = dict(cfg)
    name = cfg.pop("name")
    cfg["in_channels"] = in_channels
    return NECKS[name](**cfg)
