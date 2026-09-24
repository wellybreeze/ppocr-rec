#!/usr/bin/env python3
"""Convert PaddleOCR recognition `.pdparams` to a `ppocr_rec` checkpoint (`.pt`).

Training and inference do not depend on PaddlePaddle. Only this converter needs
`paddlepaddle` to load `.pdparams`.

Run from the repository root:

    python tools/convert_paddle_rec.py \\
        --pdparams ch_PP-OCRv5_rec_train/best_accuracy.pdparams \\
        --yaml ppocrv5_mobile.yaml \\
        --out pretrained/ppocrv5_mobile.pt \\
        --dict ppocr_rec/cfg/dicts/ppocrv5_dict.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


BN_SUFFIX = {
    "_mean": "running_mean",
    "_variance": "running_var",
}

DISTILL_PREFIXES = ("Student.", "student.", "Teacher.", "teacher.")

# Paddle Layer private attrs / NRTR embedding wrapper → our PyTorch names.
KEY_REPLACEMENTS = (
    ("._conv.", ".conv."),
    ("._batch_norm.", ".bn."),
    ("._depthwise_conv.", ".dw."),
    ("._pointwise_conv.", ".pw."),
    ("._se.", ".se."),
    (".embedding.embedding.weight", ".embedding.weight"),
    (".mlp.fc1.", ".linear1."),
    (".mlp.fc2.", ".linear2."),
)


def remap_paddle_key(key: str) -> str:
    """Map a Paddle parameter name toward a PyTorch `state_dict` key."""
    for prefix in DISTILL_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    if key.startswith("module."):
        key = key[7:]
    for src, dst in KEY_REPLACEMENTS:
        key = key.replace(src, dst)
    # Paddle SVTR encoder MLP is `linear1/2`; our CTC encoder uses `mlp.fc1/2`.
    if "svtr_block" in key:
        key = key.replace(".linear1.", ".mlp.fc1.").replace(".linear2.", ".mlp.fc2.")
    for src, dst in BN_SUFFIX.items():
        if key.endswith("." + src):
            key = key[: -len(src)] + dst
            break
    return key


def maybe_transpose(paddle_t: torch.Tensor, torch_shape: torch.Size) -> torch.Tensor:
    """Transpose Linear / permute buffers so Paddle tensors match PyTorch layout.

    Paddle Linear is ``[in, out]``, PyTorch is ``[out, in]``. Square weights still
    need a transpose; 2-D embeddings ``[V, D]`` are left as-is.
    """
    if paddle_t.ndim == 2 and tuple(paddle_t.shape[::-1]) == tuple(torch_shape):
        return paddle_t.transpose(0, 1).contiguous()
    if tuple(paddle_t.shape) == tuple(torch_shape):
        return paddle_t
    if paddle_t.ndim == 3:
        # NRTR pe: Paddle (max_len, 1, dim) vs PyTorch (1, max_len, dim)
        perm = paddle_t.permute(1, 0, 2).contiguous()
        if tuple(perm.shape) == tuple(torch_shape):
            return perm
    return paddle_t


def prefer_student(keys: list[str]) -> bool:
    return any(k.startswith(("Student.", "student.")) for k in keys)


def filter_distill_keys(state: dict) -> dict:
    """If a distillation checkpoint has Student/Teacher, keep Student only."""
    if not prefer_student(list(state.keys())):
        return state
    out = {}
    for k, v in state.items():
        if k.startswith(("Teacher.", "teacher.")):
            continue
        out[k] = v
    return out


def paddle_state_to_torch(pd_state: dict) -> dict[str, torch.Tensor]:
    tensors = {}
    for k, v in pd_state.items():
        if hasattr(v, "numpy"):
            arr = torch.from_numpy(v.numpy())
        elif isinstance(v, torch.Tensor):
            arr = v.detach().cpu()
        else:
            arr = torch.as_tensor(v)
        tensors[k] = arr.float() if arr.is_floating_point() else arr
    return tensors


def load_pdparams(path: str | Path) -> dict:
    try:
        import paddle
    except ImportError as e:
        raise ImportError("paddlepaddle is required to load .pdparams (pip install paddlepaddle)") from e
    obj = paddle.load(str(path))
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return obj


def convert_state(paddle_sd: dict, torch_sd: dict) -> tuple[dict, list[str], list[str], list[str]]:
    paddle_sd = filter_distill_keys(paddle_sd)
    tensors = paddle_state_to_torch(paddle_sd)
    remapped = {remap_paddle_key(k): v for k, v in tensors.items()}
    loaded = {}
    skipped = []
    missing = []
    used = set()
    for tk, tv in torch_sd.items():
        if tk.endswith("num_batches_tracked"):
            continue
        src = remapped.get(tk)
        if src is None:
            missing.append(tk)
            continue
        src = maybe_transpose(src, tv.shape)
        if tuple(src.shape) != tuple(tv.shape):
            skipped.append(f"{tk} paddle{tuple(src.shape)} != torch{tuple(tv.shape)}")
            continue
        loaded[tk] = src.to(dtype=tv.dtype)
        used.add(tk)
    unused = [k for k in remapped if k not in used]
    return loaded, skipped, missing, unused


def build_ocr_model(yaml_name: str, dict_path: str | None = None):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ppocr_rec.data.charset import Charset
    from ppocr_rec.nn.tasks import RecognitionModel

    charset = Charset(dict_path, use_space_char=True) if dict_path else Charset(None, use_space_char=False)
    return RecognitionModel(yaml_name, charset=charset, verbose=False), charset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert PaddleOCR rec .pdparams to ppocr_rec .pt")
    p.add_argument("--pdparams", required=True, help="PaddleOCR .pdparams path")
    p.add_argument("--yaml", required=True, help="ppocr_rec model yaml (e.g. ppocrv5_mobile.yaml)")
    p.add_argument("--out", required=True, help="Output .pt path")
    p.add_argument("--dict", dest="dict_path", default=None, help="Character dict used by the Paddle model (optional)")
    p.add_argument("--strict-report", action="store_true", help="Print missing/unused keys")
    args = p.parse_args(argv)

    model, charset = build_ocr_model(args.yaml, args.dict_path)
    paddle_sd = load_pdparams(args.pdparams)
    loaded, skipped, missing, unused = convert_state(paddle_sd, model.state_dict())
    info = model.load(loaded, strict=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "cfg": args.yaml,
        "charset": charset if args.dict_path else None,
        "convert": {
            "source": str(Path(args.pdparams).resolve()),
            "loaded": len(loaded),
            "skipped_shape": skipped,
            "missing": missing if args.strict_report else len(missing),
            "unused": unused if args.strict_report else len(unused),
            "load_info": {k: (v if k != "skipped" else len(v)) for k, v in info.items()},
        },
    }
    torch.save(ckpt, out)
    print(f"saved {out}  loaded={len(loaded)} skipped_shape={len(skipped)} missing={len(missing)}")
    if skipped:
        print("shape mismatch (classification head is expected when dict differs):")
        for line in skipped[:20]:
            print(" ", line)
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
