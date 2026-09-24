from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast

    def GradScaler(*, enabled=True):
        device = "cuda" if torch.cuda.is_available() and enabled else "cpu"
        return _GradScaler(device, enabled=enabled and device == "cuda")

    def autocast(*, enabled=True):
        device_type = "cuda" if torch.cuda.is_available() and enabled else "cpu"
        return _autocast(device_type, enabled=enabled and device_type == "cuda")

except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ppocr_rec.cfg import get_cfg, get_save_dir, yaml_save
from ppocr_rec.utils.callbacks import get_default_callbacks
from ppocr_rec.utils.torch_utils import LOGGER, init_seeds, select_device, unwrap_model


class EarlyStopping:
    """Stop when validation fitness has not improved for `patience` epochs. `patience=0` disables."""

    def __init__(self, patience: int = 50):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float("inf")

    def __call__(self, epoch: int, fitness: float) -> bool:
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        return (epoch - self.best_epoch) >= self.patience


def resolve_resume_ckpt(args) -> Path | None:
    r = getattr(args, "resume", False)
    if r in {False, None, "", "false", "False"}:
        return None
    if r in {True, "true", "True", "last"}:
        cand = Path(args.project or "runs/rec") / (args.name or "exp") / "weights" / "last.pt"
        if not cand.exists():
            raise FileNotFoundError(f"resume=True but checkpoint not found: {cand}")
        return cand
    p = Path(str(r))
    if p.is_dir():
        p = p / "weights" / "last.pt"
    if not p.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {p}")
    return p


class BaseTrainer:
    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, getattr(self.args, "batch", 16))
        init_seeds(int(self.args.seed or 0))
        resume_path = resolve_resume_ckpt(self.args)
        self.resume_ckpt = None
        self.start_epoch = 0
        if resume_path:
            self.resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            self.save_dir = resume_path.parent.parent
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.args.exist_ok = True
        else:
            self.save_dir = get_save_dir(self.args)
        self.wdir = self.save_dir / "weights"
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last = self.wdir / "last.pt"
        self.best = self.wdir / "best.pt"
        self.batch_size = int(self.args.batch)
        self.epochs = int(self.args.epochs)
        self.model = None
        self.trainset = None
        self.testset = None
        self.validator = None
        self.metrics: dict[str, Any] = {}
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.epoch = 0
        self.scaler = GradScaler(enabled=bool(self.args.amp) and self.device.type == "cuda")
        self.callbacks = _callbacks or get_default_callbacks()
        self.csv = self.save_dir / "results.csv"
        yaml_save(self.save_dir / "args.yaml", vars(self.args))

    def run_callbacks(self, event: str):
        for fn in self.callbacks.get(event, []):
            fn(self)

    def train(self):
        self.run_callbacks("on_pretrain_routine_start")
        self._setup()
        self.run_callbacks("on_pretrain_routine_end")
        self._do_train()

    def _setup(self):
        weights = None if self.resume_ckpt else self.args.pretrained
        if self.model is None:
            self.model = self.get_model(cfg=self.args.model, weights=weights)
        self.model.to(self.device)
        self.trainset = self.build_dataset(self.args.data, mode="train")
        self.testset = self.build_dataset(self.args.data, mode="val")
        self.train_loader = self.get_dataloader(self.trainset, self.batch_size, mode="train")
        self.val_loader = self.get_dataloader(self.testset, self.batch_size, mode="val")
        self.optimizer = self.build_optimizer()
        self.scheduler, self.lf = self.build_scheduler()
        if self.resume_ckpt:
            self._load_resume_state()
        self.validator = self.get_validator()

    def _load_resume_state(self):
        ckpt = self.resume_ckpt or {}
        model = unwrap_model(self.model)
        if hasattr(model, "load"):
            model.load(ckpt.get("model", ckpt), strict=False)
        else:
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        opt = ckpt.get("optimizer")
        if opt:
            try:
                self.optimizer.load_state_dict(opt)
            except Exception as e:
                LOGGER.warning("Could not restore optimizer: %s", e)
        sched = ckpt.get("scheduler")
        if sched:
            try:
                self.scheduler.load_state_dict(sched)
            except Exception as e:
                LOGGER.warning("Could not restore scheduler: %s", e)
        scaler = ckpt.get("scaler")
        if scaler:
            try:
                self.scaler.load_state_dict(scaler)
            except Exception:
                pass
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_fitness = float(ckpt.get("best_fitness", 0.0) or 0.0)
        self.best_epoch = int(ckpt.get("best_epoch", max(self.start_epoch - 1, 0)))
        self.metrics = ckpt.get("metrics") or {}
        if self.start_epoch >= self.epochs:
            LOGGER.info("Resume: already completed %s/%s epochs", self.start_epoch, self.epochs)
        else:
            LOGGER.info(
                "Resuming from epoch %s/%s (best acc=%.4f)",
                self.start_epoch + 1,
                self.epochs,
                self.best_fitness,
            )

    def build_optimizer(self):
        name = str(self.args.optimizer).lower()
        lr = float(self.args.lr0)
        wd = float(self.args.weight_decay)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=float(self.args.momentum), weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def build_scheduler(self):
        warmup = float(self.args.warmup_epochs or 0)
        epochs = self.epochs
        lrf = float(self.args.lrf or 0.0)

        def lf(epoch):
            if epoch < warmup:
                return max((epoch + 1) / max(warmup, 1e-6), 0.1)
            if self.args.cos_lr:
                progress = (epoch - warmup) / max(epochs - warmup, 1)
                return lrf + 0.5 * (1 - lrf) * (1 + math.cos(math.pi * progress))
            return 1.0 - progress * (1 - lrf) if (progress := (epoch - warmup) / max(epochs - warmup, 1)) else 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        return scheduler, lf

    def _do_train(self):
        self.run_callbacks("on_train_start")
        if self.start_epoch >= self.epochs:
            self.run_callbacks("on_train_end")
            LOGGER.info("Training complete. best=%s acc=%.4f", self.best, self.best_fitness)
            return
        stopper = EarlyStopping(patience=int(self.args.patience or 0))
        stopper.best_fitness = self.best_fitness
        stopper.best_epoch = self.best_epoch
        nb = len(self.train_loader)
        period = int(getattr(self.args, "save_period", -1) or -1)
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.model.train()
            self.run_callbacks("on_train_epoch_start")
            pbar = tqdm(enumerate(self.train_loader), total=nb, disable=not self.args.verbose)
            running = 0.0
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                batch = self.preprocess_batch(batch)
                with autocast(enabled=self.scaler.is_enabled()):
                    loss_dict = self.model(batch)
                    loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                running += float(loss.detach())
                pbar.set_description(f"epoch {epoch+1}/{self.epochs} loss={running / (i + 1):.4f}")
                self.run_callbacks("on_train_batch_end")
            self.scheduler.step()
            extra = self.wdir / f"epoch{epoch + 1}.pt" if period > 0 and (epoch + 1) % period == 0 else None
            self.save_model(period_path=extra)
            if self.args.val:
                metrics = self.validator(model=self.model)
                fitness = metrics.get("acc", 0.0)
                self.metrics = metrics
                if fitness >= self.best_fitness:
                    self.best_fitness = fitness
                    self.best_epoch = epoch
                    self.save_model(best=True)
                if stopper(epoch, fitness):
                    LOGGER.info(
                        "EarlyStopping: no acc improvement for %s epochs (best=%.4f @ epoch %s)",
                        int(self.args.patience or 0),
                        self.best_fitness,
                        self.best_epoch + 1,
                    )
                    self.run_callbacks("on_train_epoch_end")
                    break
            self.run_callbacks("on_train_epoch_end")
        self.run_callbacks("on_train_end")
        LOGGER.info("Training complete. best=%s acc=%.4f", self.best, self.best_fitness)

    def _ckpt_dict(self) -> dict[str, Any]:
        scaler_state = None
        if self.scaler is not None:
            try:
                scaler_state = self.scaler.state_dict()
            except Exception:
                scaler_state = None
        return {
            "epoch": self.epoch,
            "model": unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": scaler_state,
            "args": vars(self.args),
            "cfg": str(self.args.model),
            "charset": getattr(self, "charset", None),
            "metrics": self.metrics,
            "best_fitness": self.best_fitness,
            "best_epoch": self.best_epoch,
        }

    def save_model(self, best: bool = False, period_path: Path | None = None):
        if not getattr(self.args, "save", True):
            return
        ckpt = self._ckpt_dict()
        torch.save(ckpt, self.last)
        if best:
            torch.save(ckpt, self.best)
        if period_path is not None:
            torch.save(ckpt, period_path)

    def preprocess_batch(self, batch):
        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        raise NotImplementedError

    def get_validator(self):
        raise NotImplementedError

    def get_dataloader(self, dataset, batch_size, mode="train"):
        raise NotImplementedError

    def build_dataset(self, img_path, mode="train", batch=None):
        raise NotImplementedError
