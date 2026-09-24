from __future__ import annotations

import pytest
import torch

from ppocr_rec.data.charset import Charset
from ppocr_rec.nn.tasks import RecognitionModel


CTC_MODELS = [
    ("crnn.yaml", 32, 100),
    ("svtr.yaml", 32, 100),
]

MULTI_NRTR = [
    ("ppocrv4_mobile.yaml", 48, 320),
    ("ppocrv5_mobile.yaml", 48, 320),
    ("ppocrv4_server.yaml", 48, 320),
    ("ppocrv6_tiny.yaml", 48, 320),
    ("ppocrv6_small.yaml", 48, 320),
]

MULTI_SAR = [
    ("ppocrv3_mobile.yaml", 48, 320),
]

HEAVY = [
    ("ppocrv5_server.yaml", 48, 320),
    ("ppocrv6_medium.yaml", 48, 320),
]


@pytest.fixture
def tiny_cs(tmp_path):
    p = tmp_path / "dict.txt"
    p.write_text("a\nb\nc\n0\n1\n2\n", encoding="utf-8")
    return Charset(p, use_space_char=False, max_text_length=25)


def _batch(cs: Charset, b=2, h=48, w=320, gtc=None):
    img = torch.randn(b, 3, h, w)
    texts = ["ab", "a0"]
    labels = torch.stack([torch.from_numpy(cs.pad_ctc(cs.encode_ctc(t))) for t in texts])
    batch = {
        "img": img,
        "label": labels,
        "length": torch.tensor([2, 2], dtype=torch.long),
        "valid_ratio": torch.ones(b),
    }
    if gtc == "nrtr":
        batch["label_gtc"] = torch.stack([torch.from_numpy(cs.encode_nrtr(t)) for t in texts])
    if gtc == "sar":
        batch["label_sar"] = torch.stack([torch.from_numpy(cs.encode_sar(t)) for t in texts])
    return batch


@pytest.mark.parametrize("yaml_name,h,w", CTC_MODELS)
def test_ctc_eval_shape(tiny_cs, yaml_name, h, w):
    model = RecognitionModel(yaml_name, charset=tiny_cs)
    model.eval()
    y = model(torch.randn(2, 3, h, w))
    assert y.ndim == 3
    assert y.shape[0] == 2
    assert y.shape[2] == tiny_cs.ctc_num


@pytest.mark.parametrize("yaml_name,h,w", MULTI_NRTR + HEAVY)
def test_multihead_nrtr_eval_and_train(tiny_cs, yaml_name, h, w):
    model = RecognitionModel(yaml_name, charset=tiny_cs)
    model.eval()
    y = model(torch.randn(1, 3, h, w))
    assert y.ndim == 3 and y.shape[0] == 1 and y.shape[2] == tiny_cs.ctc_num
    model.train()
    out = model(_batch(tiny_cs, b=2, h=h, w=w, gtc="nrtr"))
    assert "loss" in out and torch.isfinite(out["loss"])


@pytest.mark.parametrize("yaml_name,h,w", MULTI_SAR)
def test_multihead_sar_eval_and_train(tiny_cs, yaml_name, h, w):
    model = RecognitionModel(yaml_name, charset=tiny_cs)
    model.eval()
    y = model(torch.randn(1, 3, h, w))
    assert y.ndim == 3 and y.shape[2] == tiny_cs.ctc_num
    model.train()
    out = model(_batch(tiny_cs, b=2, h=h, w=w, gtc="sar"))
    assert "loss" in out and torch.isfinite(out["loss"])


def test_pretrained_skips_mismatched_classifier(tiny_cs):
    model = RecognitionModel("crnn.yaml", charset=tiny_cs)
    sd = model.state_dict()
    # Simulate a different dict size on the CTC classifier.
    key = [k for k in sd if k.endswith("fc.weight")][-1]
    fake = {k: v.clone() for k, v in sd.items()}
    w = fake[key]
    fake[key] = torch.randn(w.shape[0] + 3, w.shape[1])
    info = model.load(fake, strict=False)
    assert key in info["skipped"] or any(key.endswith(s) or s.endswith(key) for s in info["skipped"])
