from __future__ import annotations

import torch

from ppocr_rec.engine.validator import BaseValidator
from ppocr_rec.utils.decode import CTCLabelDecode
from ppocr_rec.utils.metrics import RecMetrics
from ppocr_rec.data.build import build_charset, build_dataloader, load_rec_data_yaml
from ppocr_rec.data.dataset import RecDataset
from ppocr_rec.nn.tasks import RecognitionModel, yaml_model_load
from pathlib import Path


class RecValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None, overrides=None):
        super().__init__(dataloader, save_dir, args, _callbacks, overrides=overrides)
        self.charset = None
        self.decoder = None
        self.metric = RecMetrics()

    def __call__(self, trainer=None, model=None):
        if trainer is None and model is not None and self.dataloader is None:
            self._setup_from_args(model)
        return super().__call__(trainer=trainer, model=model)

    def _setup_from_args(self, model):
        data_cfg = load_rec_data_yaml(self.args.data)
        self.charset = build_charset(data_cfg, self.args)
        imgsz = self.args.imgsz
        img_h, img_w = (int(imgsz[0]), int(imgsz[1])) if not isinstance(imgsz, int) else (imgsz, imgsz)
        split = data_cfg.get(self.args.split or "val") or data_cfg["train"]
        ds = RecDataset(
            img_root=data_cfg["path"],
            label_file=Path(data_cfg["path"]) / split,
            charset=self.charset,
            img_h=img_h,
            img_w=img_w,
            delimiter=data_cfg.get("delimiter", "\t"),
            augment=False,
        )
        self.dataloader = build_dataloader(ds, int(self.args.batch), False, int(self.args.workers))

    def init_metrics(self, model):
        if self.charset is None:
            self.charset = getattr(model, "charset", None)
        self.decoder = CTCLabelDecode(self.charset)
        self.metric = RecMetrics()

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        return batch

    def postprocess(self, preds):
        return self.decoder(preds)

    def update_metrics(self, preds, batch):
        gts = [(t, 1.0) for t in batch["text"]]
        self.metric.update(preds, gts)

    def get_stats(self):
        return self.metric.results()
