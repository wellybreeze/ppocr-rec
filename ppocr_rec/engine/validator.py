from __future__ import annotations

import torch
from tqdm import tqdm

from ppocr_rec.cfg import get_cfg
from ppocr_rec.utils.callbacks import get_default_callbacks
from ppocr_rec.utils.torch_utils import LOGGER, select_device, unwrap_model


class BaseValidator:
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None, overrides=None):
        if overrides is not None or args is None:
            args = get_cfg(args, overrides)
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.args = args
        self.device = None
        self.model = None
        self.metrics = {}
        self.callbacks = _callbacks or get_default_callbacks()

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            self.device = trainer.device
            model = trainer.model
            self.args = trainer.args
            if self.dataloader is None:
                self.dataloader = trainer.val_loader
        else:
            self.args = self.args or get_cfg(overrides={"mode": "val"})
            self.device = select_device(getattr(self.args, "device", None))
        self.model = unwrap_model(model).to(self.device)
        self.model.eval()
        self.init_metrics(self.model)
        with torch.no_grad():
            for batch in tqdm(self.dataloader, disable=not getattr(self.args, "verbose", True)):
                batch = self.preprocess(batch)
                preds = self.model(batch["img"] if isinstance(batch, dict) else batch)
                preds = self.postprocess(preds)
                self.update_metrics(preds, batch)
        self.metrics = self.get_stats()
        LOGGER.info("val metrics: %s", self.metrics)
        return self.metrics

    def preprocess(self, batch):
        return batch

    def postprocess(self, preds):
        return preds

    def init_metrics(self, model):
        pass

    def update_metrics(self, preds, batch):
        pass

    def get_stats(self):
        return self.metrics
