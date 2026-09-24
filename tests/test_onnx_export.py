from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml


pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture
def tiny_dict(tmp_path: Path) -> Path:
    p = tmp_path / "dict.txt"
    p.write_text("a\nb\nc\n0\n1\n2\n", encoding="utf-8")
    return p


def _tiny_crnn(tiny_dict: Path):
    from ppocr_rec import OCR
    from ppocr_rec.data.charset import Charset
    from ppocr_rec.nn.tasks import RecognitionModel

    cs = Charset(tiny_dict, use_space_char=False, max_text_length=8)
    model = OCR("crnn.yaml")
    model.model = RecognitionModel("crnn.yaml", charset=cs)
    model.model.eval()
    return model, cs


def python_ctc_postprocess(preds: np.ndarray, ignored_tokens=(0,), remove_duplicate=True):
    preds_idx, preds_prob = preds.argmax(axis=2), preds.max(axis=2)
    ids, confs = [], []
    for i in range(len(preds_idx)):
        tok = preds_idx[i]
        sel = np.ones(len(tok), dtype=bool)
        if remove_duplicate:
            sel[1:] = tok[1:] != tok[:-1]
        for t in ignored_tokens:
            sel &= tok != t
        conf = preds_prob[i][sel].tolist() or [0]
        ids.append(tok[sel])
        confs.append(np.mean(conf))
    return ids, confs


def test_onnx_end2end_outputs_and_matches_python(tmp_path: Path, tiny_dict: Path):
    import onnx
    import onnxruntime as ort

    model, cs = _tiny_crnn(tiny_dict)
    out = model.export(
        format="onnx",
        imgsz=[32, 100],
        device="cpu",
        dynamic=True,
        postprocess=True,
        normalize=True,
        simplify=False,
        max_text_length=32,
        project=str(tmp_path / "runs"),
        name="e2e",
        exist_ok=True,
    )
    proto = onnx.load(out)
    names = [o.name for o in proto.graph.output]
    assert names == ["text_ids", "text_confs"]
    meta = {p.key: p.value for p in proto.metadata_props}
    assert "character" in meta
    assert "a" in meta["character"]

    x01 = np.random.rand(2, 3, 32, 80).astype(np.float32)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    x_norm = (x01 - mean) / std
    with torch.no_grad():
        py_preds = model.model(torch.from_numpy(x_norm)).cpu().numpy()
    py_ids, py_confs = python_ctc_postprocess(py_preds)

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    onnx_ids, onnx_confs = sess.run(None, {sess.get_inputs()[0].name: x01})
    for i in range(len(py_ids)):
        valid = onnx_ids[i][onnx_ids[i] >= 0]
        assert np.array_equal(valid, np.array(py_ids[i])), (valid, py_ids[i])
        assert abs(float(onnx_confs[i].reshape(-1)[0]) - float(py_confs[i])) < 1e-4


def test_onnx_end2end_predict_and_metadata_dict(tmp_path: Path, tiny_dict: Path):
    from ppocr_rec import OCR

    model, _ = _tiny_crnn(tiny_dict)
    img = np.full((32, 80, 3), 200, dtype=np.uint8)
    cv2.putText(img, "a0", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    im_path = tmp_path / "a0.png"
    cv2.imwrite(str(im_path), img)
    out = model.export(
        format="onnx",
        imgsz=[32, 100],
        device="cpu",
        postprocess=True,
        normalize=True,
        simplify=False,
        max_text_length=32,
        project=str(tmp_path / "runs"),
        name="pred",
        exist_ok=True,
    )
    (Path(out).parent / "dict.txt").unlink()
    loaded = OCR(out)
    preds = loaded.predict(str(im_path), device="cpu")
    assert preds and isinstance(preds[0].text, str)
    assert 0.0 <= preds[0].conf <= 1.0


def test_onnx_export_ctc_flags_and_output_path(tmp_path: Path, tiny_dict: Path):
    import ast

    import onnx
    import onnxruntime as ort

    model, _ = _tiny_crnn(tiny_dict)
    dest = tmp_path / "out" / "rec.onnx"
    img = np.full((32, 80, 3), 200, dtype=np.uint8)
    cv2.putText(img, "a0", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    im_path = tmp_path / "a0.png"
    cv2.imwrite(str(im_path), img)
    out = model.export(
        format="onnx",
        imgsz=[32, 100],
        device="cpu",
        postprocess=True,
        normalize=True,
        simplify=False,
        max_text_length=32,
        remove_duplicate=False,
        ignored_tokens=[0],
        output=str(dest),
        test_image=str(im_path),
        project=str(tmp_path / "runs"),
        name="flags",
        exist_ok=True,
    )
    assert Path(out) == dest
    assert dest.exists()
    assert (dest.parent / "metadata.yaml").exists()
    proto = onnx.load(out)
    cfg = ast.literal_eval({p.key: p.value for p in proto.metadata_props}["postprocess_config"])
    assert cfg["remove_duplicate"] is False
    assert cfg["ignored_tokens"] == [0]

    x01 = np.random.rand(1, 3, 32, 80).astype(np.float32)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    with torch.no_grad():
        py_preds = model.model(torch.from_numpy((x01 - mean) / std)).cpu().numpy()
    py_ids, _ = python_ctc_postprocess(py_preds, ignored_tokens=[0], remove_duplicate=False)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    onnx_ids, _ = sess.run(None, {sess.get_inputs()[0].name: x01})
    valid = onnx_ids[0][onnx_ids[0] >= 0]
    assert np.array_equal(valid, np.array(py_ids[0]))


def test_onnx_raw_logits_without_postprocess(tmp_path: Path, tiny_dict: Path):
    import onnx

    model, _ = _tiny_crnn(tiny_dict)
    out = model.export(
        format="onnx",
        imgsz=[32, 100],
        device="cpu",
        postprocess=False,
        normalize=False,
        simplify=False,
        project=str(tmp_path / "runs"),
        name="raw",
        exist_ok=True,
    )
    proto = onnx.load(out)
    assert [o.name for o in proto.graph.output] == ["output"]
