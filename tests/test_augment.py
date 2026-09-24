from __future__ import annotations

import numpy as np

from ppocr_rec.data.augment import rec_aug, rec_con_aug, resize_norm_img
from ppocr_rec.data.build import MultiScaleBatchSampler, infer_aug_cfg
from ppocr_rec.data.tia import tia_distort, tia_perspective, tia_stretch


def test_tia_keeps_hw():
    img = np.full((32, 80, 3), 120, dtype=np.uint8)
    for fn in (tia_distort, tia_stretch, tia_perspective):
        out = fn(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_rec_aug_uint8():
    img = np.full((32, 96, 3), 180, dtype=np.uint8)
    out = rec_aug(img, tia_prob=1.0)
    assert out.ndim == 3 and out.dtype == np.uint8


def test_rec_aug_probs_off_is_identity_except_tia():
    img = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    out = rec_aug(
        img.copy(),
        tia_prob=0,
        crop_prob=0,
        reverse_prob=0,
        noise_prob=0,
        jitter_prob=0,
        blur_prob=0,
        hsv_aug_prob=0,
        gray_prob=0,
        bgr2rgb_prob=0,
    )
    assert np.array_equal(out, img)


def test_gray_and_bgr2rgb_probs():
    img = np.zeros((16, 24, 3), dtype=np.uint8)
    img[:, :, 0] = 10
    img[:, :, 1] = 80
    img[:, :, 2] = 200
    gray = rec_aug(
        img.copy(),
        tia_prob=0,
        crop_prob=0,
        reverse_prob=0,
        noise_prob=0,
        jitter_prob=0,
        blur_prob=0,
        hsv_aug_prob=0,
        gray_prob=1.0,
        bgr2rgb_prob=0,
    )
    assert gray.shape == img.shape
    assert np.allclose(gray[:, :, 0], gray[:, :, 1]) and np.allclose(gray[:, :, 1], gray[:, :, 2])
    swapped = rec_aug(
        img.copy(),
        tia_prob=0,
        crop_prob=0,
        reverse_prob=0,
        noise_prob=0,
        jitter_prob=0,
        blur_prob=0,
        hsv_aug_prob=0,
        gray_prob=0,
        bgr2rgb_prob=1.0,
    )
    assert np.array_equal(swapped[:, :, 0], img[:, :, 2])
    assert np.array_equal(swapped[:, :, 2], img[:, :, 0])


def test_overlay_aug_args_cli_overrides_recipe():
    from types import SimpleNamespace

    from ppocr_rec.data.build import overlay_aug_args

    aug = infer_aug_cfg({"Architecture": {"algorithm": "CRNN"}})
    out = overlay_aug_args(
        aug,
        SimpleNamespace(rec_aug=False, hsv_aug_prob=0.0, gray_prob=0.2, con_aug=True, con_aug_num=1),
    )
    assert out["rec_aug"] is False
    assert out["hsv_aug_prob"] == 0.0
    assert out["gray_prob"] == 0.2
    assert out["tia_prob"] == 0.4
    assert out["con_aug"] is True
    assert out["con_aug_num"] == 1


def test_get_cfg_aug_matches_paddle_defaults():
    from ppocr_rec.cfg import get_cfg

    args = get_cfg()
    assert args.tia_prob == 0.4
    assert args.crop_prob == args.reverse_prob == args.noise_prob == 0.4
    assert args.jitter_prob == args.blur_prob == args.hsv_aug_prob == 0.4
    assert args.gray_prob == 0.0 and args.bgr2rgb_prob == 0.0
    assert args.con_aug_prob == 0.5
    assert args.geometry_p == 0.5 and args.deterioration_p == 0.25 and args.colorjitter_p == 0.25
    assert args.rec_aug is True
    assert args.con_aug is None


def test_rec_con_aug_concat_text():
    a = np.full((32, 40, 3), 10, dtype=np.uint8)
    b = np.full((32, 40, 3), 200, dtype=np.uint8)
    img, text = rec_con_aug(a, "ab", [(b, "cd")], img_h=32, img_w=320, max_text_length=25, prob=1.0)
    assert text == "abcd"
    assert img.shape[0] == 32
    assert img.shape[1] > 40


def test_infer_aug_v5_has_multiscale():
    from ppocr_rec.nn.tasks import yaml_model_load

    cfg = yaml_model_load("ppocrv5_mobile.yaml")
    aug = infer_aug_cfg(cfg)
    assert aug["type"] == "rec"
    assert aug["con_aug"] is True
    assert aug["multi_scale"] == [[320, 32], [320, 48], [320, 64]]


def test_infer_aug_v3_no_multiscale():
    from ppocr_rec.nn.tasks import yaml_model_load

    cfg = yaml_model_load("ppocrv3_mobile.yaml")
    aug = infer_aug_cfg(cfg)
    assert aug["con_aug"] is True
    assert aug["multi_scale"] is None


def test_infer_aug_svtr_uses_svtr_pipeline():
    from ppocr_rec.nn.tasks import yaml_model_load

    cfg = yaml_model_load("svtr.yaml")
    aug = infer_aug_cfg(cfg)
    assert aug["type"] == "svtr"
    assert aug["padding"] is False


def test_multiscale_batch_same_hw():
    sampler = MultiScaleBatchSampler(20, batch_size=4, scales=[[320, 32], [320, 48]], drop_last=True)
    batches = list(sampler)
    assert batches
    for batch in batches:
        hw = {(h, w) for _, h, w in batch}
        assert len(hw) == 1
        assert len(batch) == 4


def test_ppocrv2_and_svtr_ch_forward():
    import torch

    from ppocr_rec.data.charset import Charset
    from ppocr_rec.nn.tasks import RecognitionModel

    cs = Charset(None, use_space_char=False)
    m = RecognitionModel("ppocrv2.yaml", charset=cs)
    m.eval()
    y = m(torch.randn(1, 3, 32, 320))
    assert y.shape[0] == 1 and y.ndim == 3
    m2 = RecognitionModel("svtr_ch.yaml", charset=cs)
    m2.eval()
    y2 = m2(torch.randn(1, 3, 32, 320))
    assert y2.shape[0] == 1 and y2.ndim == 3
