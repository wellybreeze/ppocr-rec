from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_converter():
    path = Path(__file__).resolve().parents[1] / "tools" / "convert_paddle_rec.py"
    spec = importlib.util.spec_from_file_location("convert_paddle_rec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_remap_bn_and_distill_keys():
    cvt = _load_converter()
    assert cvt.remap_paddle_key("backbone.bn._mean") == "backbone.bn.running_mean"
    assert cvt.remap_paddle_key("backbone.bn._variance") == "backbone.bn.running_var"
    assert cvt.remap_paddle_key("Student.head.ctc_head.fc.weight") == "head.ctc_head.fc.weight"
    assert cvt.remap_paddle_key("module.backbone.conv.weight") == "backbone.conv.weight"
    assert cvt.remap_paddle_key("backbone.conv1._conv.weight") == "backbone.conv1.conv.weight"
    assert cvt.remap_paddle_key("backbone.block_list.0._depthwise_conv._batch_norm._mean") == (
        "backbone.block_list.0.dw.bn.running_mean"
    )
    assert cvt.remap_paddle_key("head.gtc_head.embedding.embedding.weight") == "head.gtc_head.embedding.weight"
    assert cvt.remap_paddle_key("head.ctc_encoder.encoder.svtr_block.0.linear1.weight") == (
        "head.ctc_encoder.encoder.svtr_block.0.mlp.fc1.weight"
    )
    assert cvt.remap_paddle_key("head.gtc_head.decoder.0.mlp.fc1.weight") == "head.gtc_head.decoder.0.linear1.weight"


def test_square_linear_weight_is_transposed():
    """Paddle Linear is [in, out]; square (d, d) still needs a transpose."""
    cvt = _load_converter()
    w = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    torch_sd = {"head.mixer.proj.weight": torch.zeros(3, 3)}
    paddle_sd = {"head.mixer.proj.weight": w.clone()}
    loaded, skipped, missing, unused = cvt.convert_state(paddle_sd, torch_sd)
    assert not skipped and not missing
    assert torch.equal(loaded["head.mixer.proj.weight"], w.T)


def test_linear_transpose_and_shape_skip():
    cvt = _load_converter()
    torch_sd = {
        "head.fc.weight": torch.zeros(10, 4),
        "head.fc.bias": torch.zeros(10),
        "backbone.conv.weight": torch.zeros(8, 3, 3, 3),
    }
    paddle_sd = {
        "head.fc.weight": torch.randn(4, 10),  # Paddle Linear [in, out]
        "head.fc.bias": torch.randn(10),
        "backbone.conv.weight": torch.randn(8, 3, 3, 3),
        "head.fc.weight_other": torch.randn(1),
    }
    loaded, skipped, missing, unused = cvt.convert_state(paddle_sd, torch_sd)
    assert loaded["head.fc.weight"].shape == (10, 4)
    assert "backbone.conv.weight" in loaded
    assert not skipped
    assert "head.fc.weight_other" in unused or unused


def test_embedding_weight_is_not_transposed():
    cvt = _load_converter()
    w = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torch_sd = {"head.embedding.weight": torch.zeros(4, 3)}
    paddle_sd = {"head.embedding.weight": w.clone()}
    loaded, skipped, *_ = cvt.convert_state(paddle_sd, torch_sd)
    assert not skipped
    assert torch.equal(loaded["head.embedding.weight"], w)


def test_skip_classifier_when_dict_size_differs():
    cvt = _load_converter()
    torch_sd = {"head.ctc_head.fc.weight": torch.zeros(7, 16)}
    paddle_sd = {"head.ctc_head.fc.weight": torch.randn(6625, 16)}
    loaded, skipped, missing, unused = cvt.convert_state(paddle_sd, torch_sd)
    assert "head.ctc_head.fc.weight" not in loaded
    assert skipped
