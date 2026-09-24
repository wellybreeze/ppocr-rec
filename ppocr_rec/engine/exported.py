from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ppocr_rec.cfg import yaml_load
from ppocr_rec.data.charset import Charset

EXPORT_SUFFIXES = {".onnx", ".torchscript"}


def is_exported_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in EXPORT_SUFFIXES


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resolve_dict(path: Path, meta: dict, character_dict=None) -> Path | None:
    if character_dict:
        p = Path(character_dict)
        return p if p.exists() else None
    raw = meta.get("dict")
    if not raw:
        sidecar = path.parent / "dict.txt"
        return sidecar if sidecar.exists() else None
    p = Path(raw)
    if not p.is_absolute():
        p = path.parent / p
    return p if p.exists() else None


def _charset_from_onnx(path: Path, meta: dict) -> Charset | None:
    if path.suffix.lower() != ".onnx":
        return None
    try:
        import onnx

        props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    except Exception:
        return None
    raw = props.get("character")
    if not raw:
        return None
    chars = [ln for ln in raw.splitlines() if ln != ""]
    if not chars:
        return None
    return Charset(
        chars=chars,
        use_space_char=bool(meta.get("use_space_char", True)),
        max_text_length=int(meta.get("max_text_length", 25)),
    )


def load_charset_for_export(path: str | Path, meta: dict, character_dict=None) -> Charset:
    path = Path(path)
    dict_path = _resolve_dict(path, meta, character_dict)
    if dict_path is not None:
        return Charset(
            dict_path,
            use_space_char=bool(meta.get("use_space_char", True)),
            max_text_length=int(meta.get("max_text_length", 25)),
        )
    cs = _charset_from_onnx(path, meta)
    if cs is not None:
        return cs
    raise FileNotFoundError(
        f"Exported model {path} has no character dict. Keep dict.txt next to the "
        "model (written by export), keep ONNX `character` metadata, or pass character_dict=..."
    )


class ExportedRecModel(nn.Module):
    """TorchScript / ONNX Runtime wrapper with the same eval interface as RecognitionModel."""

    def __init__(
        self,
        path: str | Path,
        charset: Charset,
        imgsz: list[int] | None = None,
        algorithm: str | None = None,
        has_pos_embed: bool = False,
        device: str | torch.device = "cpu",
        graph_normalize: bool = False,
        mean=None,
        std=None,
        postprocess: bool = False,
    ):
        super().__init__()
        self.path = Path(path)
        self.charset = charset
        self.imgsz = imgsz or [48, 320]
        self.algorithm = algorithm
        self.has_pos_embed = bool(has_pos_embed)
        self.graph_normalize = bool(graph_normalize)
        self.mean = mean or [0.5, 0.5, 0.5]
        self.std = std or [0.5, 0.5, 0.5]
        self.postprocess = bool(postprocess)
        self.backend = "onnx" if self.path.suffix.lower() == ".onnx" else "torchscript"
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self.session = None
        self.input_name = "images"
        self.output_names: list[str] = []
        self.ts = None
        self._init_backend()

    def _init_backend(self):
        if self.backend == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise ImportError("ONNX predict requires: pip install ppocr-rec[export]") from e
            providers = ["CPUExecutionProvider"]
            if self._device.type == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            if "text_ids" in self.output_names:
                self.postprocess = True
        else:
            self.ts = torch.jit.load(str(self.path), map_location=self._device)
            self.ts.eval()

    def to(self, device):
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        if self.backend == "onnx":
            self._init_backend()
        elif self.ts is not None:
            self.ts.to(self._device)
        return self

    def eval(self):
        if self.ts is not None:
            self.ts.eval()
        return self

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.ts is not None:
            return self.ts(x)
        arr = np.ascontiguousarray(x.detach().cpu().numpy())
        outs = self.session.run(None, {self.input_name: arr})
        if self.postprocess and "text_ids" in self.output_names:
            packed = dict(zip(self.output_names, outs))
            return {
                "text_ids": torch.from_numpy(np.array(packed["text_ids"])),
                "text_confs": torch.from_numpy(np.array(packed["text_confs"])),
            }
        t = torch.from_numpy(np.array(outs[0]))
        return t.to(dtype=x.dtype, device=x.device)


def load_exported(path: str | Path, character_dict=None, device: str | None = None) -> ExportedRecModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    meta_path = path.parent / "metadata.yaml"
    meta = yaml_load(meta_path) if meta_path.exists() else {}
    charset = load_charset_for_export(path, meta, character_dict=character_dict)
    imgsz = meta.get("imgsz") or [48, 320]
    graph_normalize = _truthy(meta.get("normalize"))
    if path.suffix.lower() == ".onnx" and not graph_normalize:
        try:
            import onnx

            props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
            graph_normalize = _truthy(props.get("graph_normalize"))
        except Exception:
            pass
    return ExportedRecModel(
        path,
        charset=charset,
        imgsz=[int(imgsz[0]), int(imgsz[1])] if not isinstance(imgsz, int) else [int(imgsz), int(imgsz)],
        algorithm=meta.get("algorithm"),
        has_pos_embed=bool(meta.get("has_pos_embed", False)),
        device=device or "cpu",
        graph_normalize=graph_normalize,
        mean=meta.get("mean") or [0.5, 0.5, 0.5],
        std=meta.get("std") or [0.5, 0.5, 0.5],
        postprocess=_truthy(meta.get("postprocess")),
    )
