"""SVTR / ABINet-style geometry and deterioration augs (OpenCV, no Paddle)."""

from __future__ import annotations

import math
import numbers
import random

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter


def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude


def sample_sym(magnitude, size=None):
    return (np.random.beta(4, 4, size=size) - 0.5) * 2 * magnitude


def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)


def get_interpolation():
    return random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])


class CVRandomRotation:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, img):
        angle = sample_sym(self.degrees)
        src_h, src_w = img.shape[:2]
        m = cv2.getRotationMatrix2D((src_w / 2, src_h / 2), angle, 1.0)
        abs_cos, abs_sin = abs(m[0, 0]), abs(m[0, 1])
        dst_w = int(src_h * abs_sin + src_w * abs_cos)
        dst_h = int(src_h * abs_cos + src_w * abs_sin)
        m[0, 2] += (dst_w - src_w) / 2
        m[1, 2] += (dst_h - src_h) / 2
        return cv2.warpAffine(img, m, (dst_w, dst_h), flags=get_interpolation(), borderMode=cv2.BORDER_REPLICATE)


class CVRandomAffine:
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = [shear] if isinstance(shear, numbers.Number) else shear

    def __call__(self, img):
        src_h, src_w = img.shape[:2]
        angle = sample_sym(self.degrees)
        if self.translate is not None:
            translations = (np.round(sample_sym(self.translate[0] * src_h)), np.round(sample_sym(self.translate[1] * src_h)))
        else:
            translations = (0, 0)
        scale = sample_uniform(self.scale[0], self.scale[1]) if self.scale is not None else 1.0
        if self.shear is not None:
            shear = [sample_sym(self.shear[0]), sample_sym(self.shear[1]) if len(self.shear) > 1 else 0.0]
        else:
            shear = [0.0, 0.0]
        cx, cy = src_w / 2, src_h / 2
        rot = math.radians(angle)
        sx, sy = math.radians(shear[0]), math.radians(shear[1])
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)
        m = np.array([d, -b, 0, -c, a, 0], dtype=np.float32) / max(scale, 1e-6)
        m[2] += m[0] * (-cx) + m[1] * (-cy) + cx
        m[5] += m[3] * (-cx) + m[4] * (-cy) + cy
        m = m.reshape(2, 3)
        dst_w = max(int(src_w * abs(scale) + abs(translations[0])), 2)
        dst_h = max(int(src_h * abs(scale) + abs(translations[1])), 2)
        m[0, 2] += translations[0]
        m[1, 2] += translations[1]
        return cv2.warpAffine(img, m, (dst_w, dst_h), flags=get_interpolation(), borderMode=cv2.BORDER_REPLICATE)


class CVRandomPerspective:
    def __init__(self, distortion=0.5):
        self.distortion = distortion

    def __call__(self, img):
        height, width = img.shape[:2]
        offset_h = sample_asym(self.distortion * height / 2, size=4).astype(np.int32)
        offset_w = sample_asym(self.distortion * width / 2, size=4).astype(np.int32)
        start = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        end = np.array(
            [
                [offset_w[0], offset_h[0]],
                [width - 1 - offset_w[1], offset_h[1]],
                [width - 1 - offset_w[2], height - 1 - offset_h[2]],
                [offset_w[3], height - 1 - offset_h[3]],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(start, end)
        max_x, max_y = int(end[:, 0].max()), int(end[:, 1].max())
        min_x, min_y = int(max(end[:, 0].min(), 0)), int(max(end[:, 1].min(), 0))
        img = cv2.warpPerspective(img, matrix, (max(max_x, 2), max(max_y, 2)), flags=get_interpolation(), borderMode=cv2.BORDER_REPLICATE)
        return img[min_y:, min_x:]


class SVTRGeometry:
    def __init__(self, aug_type=0, degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.0), shear=(45, 15), distortion=0.5, p=0.5):
        self.aug_type = aug_type
        self.p = p
        self.transforms = [
            CVRandomRotation(degrees=degrees),
            CVRandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear),
            CVRandomPerspective(distortion=distortion),
        ]

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        if self.aug_type:
            ops = list(self.transforms)
            random.shuffle(ops)
            for op in ops[: random.randint(1, 3)]:
                img = op(img)
            return img
        return self.transforms[random.randint(0, 2)](img)


class SVTRDeterioration:
    def __init__(self, var=20, degrees=6, factor=4, p=0.25):
        self.p = p
        self.var = var
        self.degrees = degrees
        self.factor = factor

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        ops = [self._noise, self._motion, self._rescale]
        random.shuffle(ops)
        for op in ops:
            img = op(img)
        return img

    def _noise(self, img):
        var = max(int(sample_asym(self.var)), 1)
        noise = np.random.normal(0, var**0.5, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def _motion(self, img):
        degree = max(int(sample_asym(self.degrees)), 1)
        angle = sample_uniform(-90, 90)
        kernel = np.zeros((degree, degree))
        kernel[degree // 2, :] = 1
        matrix = cv2.getRotationMatrix2D((degree // 2, degree // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, matrix, (degree, degree))
        kernel = kernel / max(kernel.sum(), 1e-6)
        return np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)

    def _rescale(self, img):
        factor = round(sample_uniform(0, self.factor))
        if factor == 0:
            return img
        src_h, src_w = img.shape[:2]
        scale_img = cv2.resize(img, (512, 128), interpolation=get_interpolation())
        for _ in range(factor):
            if min(scale_img.shape[:2]) < 4:
                break
            scale_img = cv2.pyrDown(scale_img)
        return cv2.resize(scale_img, (src_w, src_h), interpolation=get_interpolation())


class SVTRColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25):
        self.p = p
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = np.array(self.jitter(Image.fromarray(rgb)))
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def svtr_rec_aug(
    img: np.ndarray,
    aug_type: int = 0,
    geometry_p: float = 0.5,
    deterioration_p: float = 0.25,
    colorjitter_p: float = 0.25,
    gray_prob: float = 0.0,
    bgr2rgb_prob: float = 0.0,
) -> np.ndarray:
    from ppocr_rec.data.augment import extra_color_aug

    img = SVTRGeometry(aug_type=aug_type, p=geometry_p)(img)
    img = SVTRDeterioration(p=deterioration_p)(img)
    img = SVTRColorJitter(p=colorjitter_p)(img)
    return extra_color_aug(img, gray_prob, bgr2rgb_prob)
