from __future__ import annotations

import math
import random

import cv2
import numpy as np

from ppocr_rec.data.tia import tia_distort, tia_perspective, tia_stretch


def resize_norm_img(
    img: np.ndarray,
    img_h: int,
    img_w: int,
    padding: bool = True,
    graph_normalize: bool = False,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    """Keep height, scale width, pad right. CHW float32.

    Default: `/255` then `(x-mean)/std` (range ~[-1, 1]), pad 0.
    `graph_normalize=True`: only `/255`, pad with `mean` so an in-graph
    `(x-mean)/std` (see ONNX export) leaves pad at 0.
    """
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not padding:
        resized = cv2.resize(img, (img_w, img_h))
        resized_w = img_w
    else:
        ratio = w / float(h)
        resized_w = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
        resized_w = max(resized_w, 1)
        resized = cv2.resize(img, (resized_w, img_h))
    resized = resized.astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255.0
    c = resized.shape[0]
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)[:c]
    std_arr = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)[:c]
    valid_ratio = min(1.0, float(resized_w / img_w))
    if graph_normalize:
        if padding:
            padded = np.broadcast_to(mean_arr, (c, img_h, img_w)).copy()
            padded[:, :, :resized_w] = resized
        else:
            padded = resized
        return padded, valid_ratio
    resized = (resized - mean_arr) / std_arr
    padded = np.zeros((c, img_h, img_w), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded, valid_ratio


def _flag():
    return 1 if random.random() > 0.5000001 else -1


def hsv_aug(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * _flag()
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + delta), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def jitter(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 10 or w <= 10:
        return img
    thres = min(w, h)
    s = int(random.random() * thres * 0.01)
    src = img.copy()
    for i in range(s):
        img[i:, i:, :] = src[: h - i, : w - i, :]
    return img


def add_gaussian_noise(image: np.ndarray, mean=0, var=0.1) -> np.ndarray:
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = np.clip(image.astype(np.float32) + 0.5 * noise, 0, 255)
    return out.astype(np.uint8)


def get_crop(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    top_crop = min(int(random.randint(1, 8)), h - 1)
    if random.randint(0, 1):
        return image[top_crop:h, :, :].copy()
    return image[0 : h - top_crop, :, :].copy()


def base_data_augmentation(
    img: np.ndarray,
    crop_prob=0.4,
    reverse_prob=0.4,
    noise_prob=0.4,
    jitter_prob=0.4,
    blur_prob=0.4,
    hsv_aug_prob=0.4,
) -> np.ndarray:
    """PaddleOCR `BaseDataAugmentation`."""
    h, w = img.shape[:2]
    if random.random() <= crop_prob and h >= 20 and w >= 20:
        img = get_crop(img)
    if random.random() <= blur_prob:
        fil = cv2.getGaussianKernel(ksize=5, sigma=1, ktype=cv2.CV_32F)
        img = cv2.sepFilter2D(img, -1, fil, fil)
    if random.random() <= hsv_aug_prob:
        img = hsv_aug(img)
    if random.random() <= jitter_prob:
        img = jitter(img)
    if random.random() <= noise_prob:
        img = add_gaussian_noise(img)
    if random.random() <= reverse_prob:
        img = 255 - img
    return img


def extra_color_aug(img: np.ndarray, gray_prob: float = 0.0, bgr2rgb_prob: float = 0.0) -> np.ndarray:
    """Optional extras (not in Paddle RecAug). Default 0 = no-op."""
    if gray_prob and random.random() <= gray_prob:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if bgr2rgb_prob and random.random() <= bgr2rgb_prob:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rec_aug(
    img: np.ndarray,
    tia_prob: float = 0.4,
    crop_prob: float = 0.4,
    reverse_prob: float = 0.4,
    noise_prob: float = 0.4,
    jitter_prob: float = 0.4,
    blur_prob: float = 0.4,
    hsv_aug_prob: float = 0.4,
    gray_prob: float = 0.0,
    bgr2rgb_prob: float = 0.0,
) -> np.ndarray:
    """PaddleOCR `RecAug`: TIA then BaseDataAugmentation."""
    h, w = img.shape[:2]
    if random.random() <= tia_prob and h >= 20 and w >= 20:
        img = tia_distort(img, random.randint(3, 6))
        img = tia_stretch(img, random.randint(3, 6))
        img = tia_perspective(img)
    img = base_data_augmentation(
        img,
        crop_prob=crop_prob,
        reverse_prob=reverse_prob,
        noise_prob=noise_prob,
        jitter_prob=jitter_prob,
        blur_prob=blur_prob,
        hsv_aug_prob=hsv_aug_prob,
    )
    return extra_color_aug(img, gray_prob, bgr2rgb_prob)


def rec_con_aug(img: np.ndarray, text: str, extras: list[tuple[np.ndarray, str]], img_h: int, img_w: int, max_text_length: int, prob: float = 0.5):
    """PaddleOCR `RecConAug`: horizontally concat extra word images."""
    if random.random() > prob or not extras:
        return img, text
    max_wh = img_w / float(img_h)
    for ext_img, ext_text in extras:
        if len(text) + len(ext_text) > max_text_length:
            break
        concat_ratio = img.shape[1] / float(img.shape[0]) + ext_img.shape[1] / float(ext_img.shape[0])
        if concat_ratio > max_wh:
            break
        ori_w = max(round(img.shape[1] / img.shape[0] * img_h), 1)
        ext_w = max(round(ext_img.shape[1] / ext_img.shape[0] * img_h), 1)
        left = cv2.resize(img, (ori_w, img_h))
        right = cv2.resize(ext_img, (ext_w, img_h))
        img = np.concatenate([left, right], axis=1)
        text = text + ext_text
    return img, text


_REC_KEYS = (
    "tia_prob",
    "crop_prob",
    "reverse_prob",
    "noise_prob",
    "jitter_prob",
    "blur_prob",
    "hsv_aug_prob",
    "gray_prob",
    "bgr2rgb_prob",
)
_SVTR_KEYS = ("geometry_p", "deterioration_p", "colorjitter_p", "gray_prob", "bgr2rgb_prob")


def apply_rec_aug(img: np.ndarray, aug_type: str = "rec", kwargs: dict | None = None) -> np.ndarray:
    """Dispatch RecAug / SVTRRecAug with optional probability overrides."""
    kwargs = kwargs or {}
    if aug_type == "svtr":
        from ppocr_rec.data.svtr_aug import svtr_rec_aug

        return svtr_rec_aug(img, **{k: kwargs[k] for k in _SVTR_KEYS if k in kwargs})
    return rec_aug(img, **{k: kwargs[k] for k in _REC_KEYS if k in kwargs})
