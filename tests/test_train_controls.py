from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from ppocr_rec.data.charset import Charset
from ppocr_rec.data.dataset import RecDataset


@pytest.fixture
def tiny_dict(tmp_path: Path) -> Path:
    p = tmp_path / "dict.txt"
    p.write_text("a\nb\nc\n0\n1\n2\n", encoding="utf-8")
    return p


def _make_rec_data(tmp_path: Path, tiny_dict: Path, n: int = 8) -> Path:
    root = tmp_path / "data"
    (root / "imgs").mkdir(parents=True)
    lines = []
    texts = ["a0", "b1", "ab", "c2", "a1", "b2", "ac", "bc"]
    for i in range(n):
        text = texts[i % len(texts)]
        img = np.full((32, 80, 3), 200, dtype=np.uint8)
        cv2.putText(img, text, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        rel = f"imgs/{i}.png"
        cv2.imwrite(str(root / rel), img)
        lines.append(f"{rel}\t{text}")
    (root / "train_list.txt").write_text("\n".join(lines), encoding="utf-8")
    (root / "val_list.txt").write_text("\n".join(lines[: max(n // 2, 1)]), encoding="utf-8")
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
    return data_yaml


def test_early_stopping_patience_zero_never_stops():
    from ppocr_rec.engine.trainer import EarlyStopping

    stop = EarlyStopping(patience=0)
    for epoch in range(20):
        assert stop(epoch, 0.1) is False


def test_early_stopping_stops_after_patience():
    from ppocr_rec.engine.trainer import EarlyStopping

    stop = EarlyStopping(patience=2)
    assert stop(0, 0.5) is False
    assert stop(1, 0.4) is False
    assert stop(2, 0.3) is True


def test_early_stopping_reset_on_improvement():
    from ppocr_rec.engine.trainer import EarlyStopping

    stop = EarlyStopping(patience=2)
    assert stop(0, 0.2) is False
    assert stop(1, 0.1) is False
    assert stop(2, 0.5) is False
    assert stop(3, 0.4) is False
    assert stop(4, 0.3) is True


def test_dataset_fraction_keeps_reproducible_subset(tmp_path: Path, tiny_dict: Path):
    data_yaml = _make_rec_data(tmp_path, tiny_dict, n=8)
    root = tmp_path / "data"
    cs = Charset(tiny_dict, use_space_char=False)
    a = RecDataset(root, root / "train_list.txt", cs, img_h=32, img_w=100, fraction=0.5, seed=0)
    b = RecDataset(root, root / "train_list.txt", cs, img_h=32, img_w=100, fraction=0.5, seed=0)
    c = RecDataset(root, root / "train_list.txt", cs, img_h=32, img_w=100, fraction=1.0, seed=0)
    assert len(a) == 4
    assert len(c) == 8
    assert a.samples == b.samples


def _train_kwargs(tmp_path: Path, data_yaml: Path, **extra):
    kw = dict(
        data=str(data_yaml),
        epochs=2,
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
        patience=0,
        save_period=-1,
        fraction=1.0,
    )
    kw.update(extra)
    return kw


def test_save_period_writes_epoch_checkpoints(tmp_path: Path, tiny_dict: Path):
    from ppocr_rec import OCR

    data_yaml = _make_rec_data(tmp_path, tiny_dict)
    OCR("crnn.yaml").train(**_train_kwargs(tmp_path, data_yaml, epochs=2, save_period=1))
    wdir = tmp_path / "runs" / "t1" / "weights"
    assert (wdir / "last.pt").exists()
    assert (wdir / "best.pt").exists()
    assert (wdir / "epoch1.pt").exists()
    assert (wdir / "epoch2.pt").exists()


def test_resume_continues_from_last_epoch(tmp_path: Path, tiny_dict: Path):
    from ppocr_rec import OCR

    data_yaml = _make_rec_data(tmp_path, tiny_dict)
    last = tmp_path / "runs" / "t1" / "weights" / "last.pt"
    OCR("crnn.yaml").train(**_train_kwargs(tmp_path, data_yaml, epochs=1))
    ckpt1 = torch.load(last, map_location="cpu", weights_only=False)
    assert ckpt1["epoch"] == 0
    model = OCR("crnn.yaml")
    model.train(**_train_kwargs(tmp_path, data_yaml, epochs=3, resume=str(last)))
    ckpt2 = torch.load(last, map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] == 2
    assert ckpt2.get("best_fitness") is not None
    assert model.trainer.start_epoch == 1


def test_patience_stops_when_acc_plateaus(tmp_path: Path, tiny_dict: Path, monkeypatch):
    from ppocr_rec import OCR
    from ppocr_rec.models.rec.val import RecValidator

    accs = iter([0.4, 0.2, 0.1, 0.05])

    def fake_call(self, trainer=None, model=None):
        self.metrics = {"acc": next(accs), "norm_edit_dis": 0.0}
        return self.metrics

    monkeypatch.setattr(RecValidator, "__call__", fake_call)
    data_yaml = _make_rec_data(tmp_path, tiny_dict)
    model = OCR("crnn.yaml")
    model.train(**_train_kwargs(tmp_path, data_yaml, epochs=10, patience=2))
    assert model.trainer.epoch < 9
    assert model.trainer.epoch <= 2
