from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from ppocr_rec.cfg import get_cfg, get_save_dir
from ppocr_rec.data.augment import resize_norm_img
from ppocr_rec.engine.results import RecResult
from ppocr_rec.utils.callbacks import get_default_callbacks
from ppocr_rec.utils.torch_utils import LOGGER, select_device, unwrap_model


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(source) -> list[Path]:
    if source is None:
        raise ValueError("predict source is required")
    if isinstance(source, (list, tuple)):
        paths = []
        for s in source:
            paths.extend(list_images(s))
        return paths
    if isinstance(source, np.ndarray):
        return [source]
    p = Path(source)
    if p.is_dir():
        return sorted(x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS)
    if p.is_file():
        return [p]
    raise FileNotFoundError(source)


class BasePredictor:
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        self.device = select_device(self.args.device)
        self.model = None
        self.callbacks = _callbacks or get_default_callbacks()

    def setup_model(self, model):
        self.model = unwrap_model(model).to(self.device)
        self.model.eval()

    def __call__(self, source=None, stream=False):
        source = source or self.args.source
        items = list_images(source)
        loaded = [self.load_image(item) for item in items]
        results = [None] * len(loaded)
        imgsz = self.args.imgsz
        if isinstance(imgsz, int):
            img_h, img_w = imgsz, imgsz
        else:
            img_h, img_w = int(imgsz[0]), int(imgsz[1])
        widths = []
        for im, _ in loaded:
            h, w = im.shape[:2]
            ratio = w / float(max(h, 1))
            rw = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
            widths.append(max(rw, 1))
        order = sorted(range(len(loaded)), key=lambda i: widths[i])
        batch_size = max(int(getattr(self.args, "batch", 1) or 1), 1)
        force_full_w = bool(getattr(self.model, "has_pos_embed", False))
        if not force_full_w and self.model is not None:
            try:
                force_full_w = any("pos_embed" in n for n, _ in self.model.named_parameters())
            except Exception:
                force_full_w = False
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                grouped_w = int(math.ceil(max(widths[i] for i in chunk) / 8) * 8) or 8
                max_w = img_w if force_full_w else min(img_w, grouped_w)
                tensors = []
                paths = []
                for i in chunk:
                    arr, _ = resize_norm_img(
                        loaded[i][0],
                        img_h,
                        max_w,
                        graph_normalize=bool(getattr(self.model, "graph_normalize", False)),
                        mean=getattr(self.model, "mean", (0.5, 0.5, 0.5)),
                        std=getattr(self.model, "std", (0.5, 0.5, 0.5)),
                    )
                    tensors.append(torch.from_numpy(arr))
                    paths.append(loaded[i][1])
                batch = torch.stack(tensors, 0).to(self.device)
                preds = self.model(batch)
                rec = self.postprocess(preds, paths)
                if not isinstance(rec, list):
                    rec = [rec]
                for i, r in zip(chunk, rec):
                    results[i] = r
                    if self.args.save_txt:
                        self.save_txt(r)
        return results

    def load_image(self, item):
        if isinstance(item, np.ndarray):
            return item, None
        img = cv2.imread(str(item), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(item)
        return img, str(item)

    def preprocess(self, im: np.ndarray) -> torch.Tensor:
        imgsz = self.args.imgsz
        if isinstance(imgsz, int):
            h, w = imgsz, imgsz
        else:
            h, w = int(imgsz[0]), int(imgsz[1])
        arr, _ = resize_norm_img(
            im,
            h,
            w,
            graph_normalize=bool(getattr(self.model, "graph_normalize", False)),
            mean=getattr(self.model, "mean", (0.5, 0.5, 0.5)),
            std=getattr(self.model, "std", (0.5, 0.5, 0.5)),
        )
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def postprocess(self, preds, path):
        raise NotImplementedError

    def save_txt(self, result: RecResult):
        if result.path:
            out = self.save_dir / "labels" / (Path(result.path).stem + ".txt")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"{result.text}\t{result.conf:.4f}\n", encoding="utf-8")
