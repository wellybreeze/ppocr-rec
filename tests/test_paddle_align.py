from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_paddle_torch_rec.py"
PRETRAINED = ROOT / "pretrained"


def _load_cmp():
    spec = importlib.util.spec_from_file_location("compare_paddle_torch_rec", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _paddle_ok():
    paddle_root = ROOT / "PaddleOCR"
    if str(paddle_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(paddle_root))
    try:
        import paddle  # noqa: F401
        from ppocr.modeling.architectures.base_model import BaseModel  # noqa: F401
    except Exception:
        return False
    return True


ALIGN_JOBS = [
    "ppocrv3_mobile",
    "ppocrv4_mobile",
    "ppocrv4_server",
    "ppocrv5_mobile",
    "ppocrv5_server",
    "ppocrv6_tiny",
    "ppocrv6_small",
    "ppocrv6_medium",
]


@pytest.mark.skipif(not SCRIPT.exists(), reason="compare script missing")
@pytest.mark.parametrize("name", ALIGN_JOBS)
def test_paddle_torch_softmax_aligns(name):
    if not _paddle_ok():
        pytest.skip("paddle/PaddleOCR not available")
    cmp = _load_cmp()
    job = next(j for j in cmp.JOBS if j["name"] == name)
    if not job["pt"].exists() or not job["pdparams"].exists():
        pytest.skip(f"missing weights for {name}")
    rng = np.random.RandomState(0)
    x = rng.randn(1, 3, 48, 320).astype(np.float32)
    res = cmp.compare_one(job, x)
    assert res["softmax"]["cosine"] > 0.9999
    assert res["softmax"]["max_abs"] < 1e-4
    assert res["softmax"]["argmax_agree"] == 1.0
    assert res["logits"]["cosine"] > 0.9999
