from __future__ import annotations

import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from ppocr_rec import DEFAULT_CFG_PATH, MODELS_DIR, PACKAGE_ROOT

LOGGER = logging.getLogger("ppocr_rec")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MODES = ("train", "val", "predict", "export")
TASKS = ("rec",)


def yaml_load(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def yaml_save(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


class IterableSimpleNamespace(SimpleNamespace):
    def __iter__(self):
        return iter(vars(self).items())

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


def _flatten_none(d: dict) -> dict:
    return {k: v for k, v in d.items()}


def get_cfg(cfg=None, overrides: dict | None = None) -> IterableSimpleNamespace:
    """Merge default.yaml ⊂ cfg ⊂ overrides (right wins)."""
    base = yaml_load(DEFAULT_CFG_PATH)
    if cfg is None:
        merged = base
    elif isinstance(cfg, (str, Path)):
        extra = yaml_load(cfg)
        merged = {**base, **extra}
    elif isinstance(cfg, dict):
        merged = {**base, **cfg}
    elif isinstance(cfg, SimpleNamespace):
        merged = {**base, **vars(cfg)}
    else:
        raise TypeError(f"Unsupported cfg type: {type(cfg)}")
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v
    return IterableSimpleNamespace(**_flatten_none(merged))


def resolve_model_yaml(name: str | Path) -> Path:
    p = Path(name)
    if p.exists():
        return p.resolve()
    cand = MODELS_DIR / p.name
    if cand.exists():
        return cand
    if not p.suffix:
        cand = MODELS_DIR / f"{p.name}.yaml"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Model yaml not found: {name}")


def get_save_dir(args) -> Path:
    project = Path(args.project or "runs/rec")
    name = args.name or args.mode or "exp"
    path = project / name
    if path.exists() and not getattr(args, "exist_ok", False):
        i = 2
        while (project / f"{name}{i}").exists():
            i += 1
        path = project / f"{name}{i}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_key_value(argv: list[str]) -> tuple[str | None, str | None, dict]:
    task = None
    mode = None
    overrides: dict[str, Any] = {}
    for token in argv:
        if token in TASKS and task is None:
            task = token
        elif token in MODES and mode is None:
            mode = token
        elif "=" in token:
            k, v = token.split("=", 1)
            k = k.strip().lstrip("-")
            v = v.strip()
            if v.lower() in {"true", "false"}:
                overrides[k] = v.lower() == "true"
            elif (v[:1] in {"[", "{"} and v[-1:] in {"]", "}"}) or (v[:1] == "(" and v[-1:] == ")"):
                try:
                    overrides[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    overrides[k] = v
            else:
                try:
                    if "." in v:
                        overrides[k] = float(v)
                    else:
                        overrides[k] = int(v)
                except ValueError:
                    overrides[k] = v
        else:
            LOGGER.warning("Ignoring CLI token: %s", token)
    return task, mode, overrides


def entrypoint(debug: str = "") -> None:
    import sys

    from ppocr_rec import OCR

    argv = debug.split() if debug else sys.argv[1:]
    task, mode, overrides = parse_key_value(argv)
    mode = mode or overrides.pop("mode", None) or "predict"
    model_name = overrides.pop("model", None) or "crnn.yaml"
    if task:
        overrides["task"] = task
    character_dict = overrides.pop("character_dict", None)
    model = OCR(model_name, task=overrides.get("task", "rec"), character_dict=character_dict)
    if not hasattr(model, mode):
        raise ValueError(f"Unknown mode {mode}. Expected one of {MODES}")
    getattr(model, mode)(**overrides)


if __name__ == "__main__":
    entrypoint()
