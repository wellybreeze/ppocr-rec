from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RecResult:
    text: str
    conf: float
    path: str | None = None

    def __repr__(self) -> str:
        return f"RecResult(text={self.text!r}, conf={self.conf:.4f}, path={self.path!r})"
