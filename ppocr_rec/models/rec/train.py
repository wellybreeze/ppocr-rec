from __future__ import annotations

from pathlib import Path

import torch

from ppocr_rec.data.build import build_charset, build_dataloader, infer_aug_cfg, load_rec_data_yaml, overlay_aug_args
from ppocr_rec.data.dataset import RecDataset
from ppocr_rec.engine.trainer import BaseTrainer
from ppocr_rec.nn.tasks import RecognitionModel
from ppocr_rec.utils.torch_utils import LOGGER


class RecTrainer(BaseTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "rec"
        super().__init__(cfg, overrides, _callbacks)
        self.charset = None
        self.data_cfg = None

    def _setup(self):
        self.data_cfg = load_rec_data_yaml(self.args.data)
        self.charset = build_charset(self.data_cfg, self.args)
        imgsz = self.args.imgsz
        if isinstance(imgsz, (list, tuple)):
            self.args.img_h, self.args.img_w = int(imgsz[0]), int(imgsz[1])
        super()._setup()

    def get_model(self, cfg=None, weights=None, verbose=True):
        cfg = cfg or self.args.model
        model = RecognitionModel(cfg, charset=self.charset, verbose=verbose)
        if weights and weights not in {False, "false", "False", None, ""}:
            info = model.load(weights, strict=False)
            LOGGER.info("Loaded pretrained weights: %s", info)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        data = self.data_cfg
        split = data["train"] if mode == "train" else data.get("val") or data["train"]
        label_file = Path(data["path"]) / split
        gtc = None
        arch = {}
        from ppocr_rec.nn.tasks import yaml_model_load

        try:
            y = yaml_model_load(self.args.model)
            arch = y.get("Architecture", y)
        except Exception:
            y, arch = {}, {}
        head = (arch.get("Head") or {}).get("name")
        if head == "MultiHead":
            names = [next(iter(x.keys())) for x in arch["Head"].get("head_list", [])]
            gtc = "sar" if "SARHead" in names else "nrtr"
        aug = overlay_aug_args(
            infer_aug_cfg(y if y else {"Architecture": arch, "algorithm": arch.get("algorithm")}),
            self.args,
        )
        return RecDataset(
            img_root=data["path"],
            label_file=label_file,
            charset=self.charset,
            img_h=int(self.args.img_h),
            img_w=int(self.args.img_w),
            delimiter=data.get("delimiter", "\t"),
            augment=mode == "train",
            encode_gtc=gtc,
            aug_type=aug["type"],
            con_aug=bool(aug["con_aug"]) and mode == "train",
            con_aug_num=int(aug["con_aug_num"]),
            padding=bool(aug["padding"]),
            fraction=float(self.args.fraction or 1.0) if mode == "train" else 1.0,
            seed=int(self.args.seed or 0),
            aug_kwargs=aug,
        )

    def get_dataloader(self, dataset, batch_size, mode="train"):
        scales = None
        if mode == "train":
            try:
                from ppocr_rec.nn.tasks import yaml_model_load

                y = yaml_model_load(self.args.model)
                scales = overlay_aug_args(infer_aug_cfg(y), self.args).get("multi_scale")
            except Exception:
                scales = None
        return build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            workers=int(self.args.workers),
            drop_last=mode == "train",
            scales=scales,
        )

    def get_validator(self):
        from ppocr_rec.models.rec.val import RecValidator

        v = RecValidator(dataloader=self.val_loader, args=self.args)
        v.charset = self.charset
        return v

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["label"] = batch["label"].to(self.device, non_blocking=True)
        batch["length"] = batch["length"].to(self.device, non_blocking=True)
        if "label_gtc" in batch:
            batch["label_gtc"] = batch["label_gtc"].to(self.device, non_blocking=True)
        if "label_sar" in batch:
            batch["label_sar"] = batch["label_sar"].to(self.device, non_blocking=True)
        return batch
