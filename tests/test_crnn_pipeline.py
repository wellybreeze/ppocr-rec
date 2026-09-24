from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def tiny_dict(tmp_path: Path) -> Path:
    p = tmp_path / "dict.txt"
    p.write_text("a\nb\nc\n0\n1\n2\n", encoding="utf-8")
    return p


def test_charset_ctc_roundtrip(tiny_dict):
    from ppocr_rec.data.charset import Charset, ctc_decode_ids

    cs = Charset(tiny_dict, use_space_char=False, max_text_length=10)
    ids = cs.encode_ctc("ab01")
    assert ids == [cs.ctc_dict["a"], cs.ctc_dict["b"], cs.ctc_dict["0"], cs.ctc_dict["1"]]
    padded = cs.pad_ctc(ids)
    text, _ = ctc_decode_ids(padded, None, cs)
    assert text == "ab01"
    assert cs.ctc_chars[0] == "blank"


def test_ctc_decode_blank_and_dup(tiny_dict):
    from ppocr_rec.data.charset import Charset, ctc_decode_ids

    cs = Charset(tiny_dict, use_space_char=False)
    a = cs.ctc_dict["a"]
    b = cs.ctc_dict["b"]
    ids = [0, a, a, 0, b, 0]
    text, _ = ctc_decode_ids(ids, [1] * 6, cs)
    assert text == "ab"


def test_crnn_forward_shape(tiny_dict):
    from ppocr_rec.data.charset import Charset
    from ppocr_rec.nn.tasks import RecognitionModel

    cs = Charset(tiny_dict, use_space_char=False)
    model = RecognitionModel("crnn.yaml", charset=cs)
    x = torch.randn(2, 3, 32, 100)
    y = model(x)
    assert y.ndim == 3
    assert y.shape[0] == 2
    assert y.shape[2] == cs.ctc_num


def test_crnn_train_val_predict_export(tmp_path: Path, tiny_dict):
    from ppocr_rec import OCR

    root = tmp_path / "data"
    (root / "imgs").mkdir(parents=True)
    lines = []
    for i, text in enumerate(["a0", "b1", "ab", "c2", "a1", "b2", "ac", "bc"]):
        img = np.full((32, 80, 3), 200, dtype=np.uint8)
        cv2.putText(img, text, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        rel = f"imgs/{i}.png"
        cv2.imwrite(str(root / rel), img)
        lines.append(f"{rel}\t{text}")
    (root / "train_list.txt").write_text("\n".join(lines), encoding="utf-8")
    (root / "val_list.txt").write_text("\n".join(lines[:4]), encoding="utf-8")
    data_yaml = tmp_path / "rec.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train_list.txt",
                "val": "val_list.txt",
                "dict": str(tiny_dict),
                "use_space_char": False,
                "delimiter": "\t",
            }
        ),
        encoding="utf-8",
    )
    model = OCR("crnn.yaml")
    model.train(
        data=str(data_yaml),
        epochs=1,
        batch=4,
        imgsz=[32, 100],
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="t1",
        exist_ok=True,
        amp=False,
        pretrained=False,
        verbose=False,
    )
    metrics = model.val(data=str(data_yaml), batch=4, workers=0, device="cpu", imgsz=[32, 100], verbose=False)
    assert "acc" in metrics
    preds = model.predict(str(root / "imgs" / "0.png"), imgsz=[32, 100], device="cpu")
    assert preds and preds[0].text is not None
    out = model.export(format="torchscript", imgsz=[32, 100], device="cpu", project=str(tmp_path / "runs"), name="exp", exist_ok=True)
    assert Path(out).exists()
    assert (Path(out).parent / "dict.txt").exists()
    assert (Path(out).parent / "metadata.yaml").exists()
    exported = OCR(out)
    preds2 = exported.predict(str(root / "imgs" / "0.png"), device="cpu")
    assert preds2 and preds2[0].text is not None


def test_yaml_imgsz_follows_model():
    from ppocr_rec import OCR

    assert OCR("crnn.yaml")._yaml_defaults()["imgsz"] == [32, 100]
    assert OCR("ppocrv5_mobile.yaml")._yaml_defaults()["imgsz"] == [48, 320]


def test_cli_parses_imgsz_list():
    from ppocr_rec.cfg import parse_key_value

    _, mode, ov = parse_key_value(["train", "imgsz=[32, 100]", "amp=false", "epochs=2"])
    assert mode == "train"
    assert ov["imgsz"] == [32, 100]
    assert ov["amp"] is False
    assert ov["epochs"] == 2


def test_export_onnx_roundtrip(tmp_path: Path, tiny_dict):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ppocr_rec import OCR
    from ppocr_rec.data.charset import Charset
    from ppocr_rec.nn.tasks import RecognitionModel

    cs = Charset(tiny_dict, use_space_char=False)
    model = OCR("crnn.yaml")
    model.model = RecognitionModel("crnn.yaml", charset=cs)
    img = np.full((32, 80, 3), 200, dtype=np.uint8)
    cv2.putText(img, "a0", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    im_path = tmp_path / "a0.png"
    cv2.imwrite(str(im_path), img)
    out = model.export(
        format="onnx",
        imgsz=[32, 100],
        device="cpu",
        dynamic=True,
        project=str(tmp_path / "runs"),
        name="onnx",
        exist_ok=True,
    )
    loaded = OCR(out)
    preds = loaded.predict(str(im_path), device="cpu")
    assert preds and isinstance(preds[0].text, str)
