from __future__ import annotations

from pathlib import Path

import torch

from ppocr_rec.cfg import get_cfg, get_save_dir, yaml_save
from ppocr_rec.utils.torch_utils import LOGGER, select_device, unwrap_model


def _float3(value, default: float = 0.5) -> list[float]:
    if value is None:
        return [default, default, default]
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    seq = [float(x) for x in list(value)]
    if len(seq) < 3:
        seq = (seq + [default] * 3)[:3]
    return seq[:3]


def _int_list(value, default=(0,)) -> list[int]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, bool):
        return list(default)
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        return [int(x) for x in parts] if parts else list(default)
    return [int(x) for x in value]


class Exporter:
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)

    def _export_path(self, fmt: str, stem: str) -> Path:
        explicit = getattr(self.args, "output_model", None) or getattr(self.args, "output", None)
        if explicit:
            path = Path(str(explicit))
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        ext = "onnx" if fmt == "onnx" else "torchscript"
        return self.save_dir / f"{stem}.{ext}"

    def __call__(self, model=None) -> str:
        model = unwrap_model(model).eval()
        device = select_device(self.args.device)
        model.to(device)
        imgsz = self.args.imgsz
        h, w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
        dummy = torch.zeros(1, 3, h, w, device=device)
        fmt = str(self.args.format or "torchscript").lower().lstrip(".")
        if fmt in {"pt", "ts", "jit"}:
            fmt = "torchscript"
        stem = Path(getattr(self.args, "model", None) or "model").stem
        charset = getattr(model, "charset", None)
        max_len = int(self.args.max_text_length or 25)
        if charset is not None and not getattr(self.args, "max_text_length", None):
            max_len = int(getattr(charset, "max_text_length", max_len) or max_len)
        ignored_tokens = _int_list(getattr(self.args, "ignored_tokens", None), default=(0,))
        remove_duplicate = bool(getattr(self.args, "remove_duplicate", True))
        normalize = bool(getattr(self.args, "normalize", True))
        postprocess = bool(getattr(self.args, "postprocess", True))
        simplify = bool(getattr(self.args, "simplify", False))
        mean = _float3(getattr(self.args, "mean", None))
        std = _float3(getattr(self.args, "std", None))
        path = self._export_path(fmt, stem)
        sidecar_dir = path.parent
        if fmt == "torchscript":
            ts = torch.jit.trace(model, dummy, strict=False)
            ts.save(str(path))
        elif fmt == "onnx":
            from ppocr_rec.engine.onnx_postprocess import RecOnnxWrapper, stamp_onnx

            wrapped = RecOnnxWrapper(
                model,
                normalize=normalize,
                mean=mean,
                std=std,
                postprocess=postprocess,
                max_text_length=max_len,
                ignored_tokens=ignored_tokens,
                remove_duplicate=remove_duplicate,
            ).to(device).eval()
            if postprocess:
                out_names = ["text_ids", "text_confs"]
                dynamic_axes = {"images": {0: "batch", 3: "width"}, "text_ids": {0: "batch"}, "text_confs": {0: "batch"}}
            else:
                out_names = ["output"]
                dynamic_axes = {"images": {0: "batch", 3: "width"}, "output": {0: "batch", 1: "time"}}
            kwargs = dict(
                input_names=["images"],
                output_names=out_names,
                opset_version=int(self.args.opset or 17),
                dynamic_axes=dynamic_axes if self.args.dynamic else None,
            )
            try:
                torch.onnx.export(wrapped, dummy, str(path), dynamo=False, **kwargs)
            except TypeError:
                torch.onnx.export(wrapped, dummy, str(path), **kwargs)
            stamp_onnx(
                str(path),
                charset,
                extra={
                    "graph_normalize": "1" if normalize else "0",
                    "postprocess_config": str(
                        {
                            "ignored_tokens": ignored_tokens,
                            "remove_duplicate": remove_duplicate,
                            "max_text_length": max_len,
                        }
                    ),
                },
                simplify=simplify,
            )
        else:
            raise ValueError(f"Unsupported export format: {fmt}. Use torchscript or onnx.")
        dict_name = "dict.txt"
        if charset is not None and getattr(charset, "raw_chars", None):
            (sidecar_dir / dict_name).write_text("\n".join(charset.raw_chars) + "\n", encoding="utf-8")
        has_pos_embed = any("pos_embed" in n for n, _ in model.named_parameters())
        meta = {
            "format": fmt,
            "imgsz": [h, w],
            "algorithm": getattr(model, "algorithm", None),
            "dynamic": bool(self.args.dynamic) if fmt == "onnx" else False,
            "opset": int(self.args.opset or 17) if fmt == "onnx" else None,
            "has_pos_embed": has_pos_embed,
            "use_space_char": bool(getattr(charset, "use_space_char", True)),
            "max_text_length": max_len,
            "dict": dict_name if charset is not None else None,
            "normalize": normalize if fmt == "onnx" else False,
            "postprocess": postprocess if fmt == "onnx" else False,
            "mean": mean if fmt == "onnx" else None,
            "std": std if fmt == "onnx" else None,
            "ignored_tokens": ignored_tokens if fmt == "onnx" else None,
            "remove_duplicate": remove_duplicate if fmt == "onnx" else None,
            "simplify": simplify if fmt == "onnx" else False,
        }
        yaml_save(sidecar_dir / "metadata.yaml", meta)
        LOGGER.info("Exported %s", path)
        test_image = getattr(self.args, "test_image", None)
        if test_image:
            self._verify_test_image(str(path), str(test_image), device)
        return str(path)

    def _verify_test_image(self, model_path: str, image_path: str, device) -> None:
        from ppocr_rec.models.rec.model import OCR

        if not Path(image_path).exists():
            LOGGER.warning("test_image not found: %s", image_path)
            return
        rec = OCR(model_path)
        results = rec.predict(image_path, device=device)
        if not results:
            LOGGER.warning("test_image produced no results: %s", image_path)
            return
        r = results[0]
        LOGGER.info("test_image %s -> text=%r conf=%.4f", image_path, r.text, r.conf)
