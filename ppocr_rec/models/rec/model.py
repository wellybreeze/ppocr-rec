from __future__ import annotations

from typing import Any

from ppocr_rec.engine.model import Model
from ppocr_rec.nn.tasks import RecognitionModel


class OCR(Model):
    """OCR recognition facade."""

    def __init__(
        self,
        model: str = "crnn.yaml",
        task: str | None = "rec",
        verbose: bool = False,
        character_dict: str | None = None,
    ):
        super().__init__(model=model, task=task or "rec", verbose=verbose, character_dict=character_dict)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        from ppocr_rec.models.rec.predict import RecPredictor
        from ppocr_rec.models.rec.train import RecTrainer
        from ppocr_rec.models.rec.val import RecValidator

        return {
            "rec": {
                "model": RecognitionModel,
                "trainer": RecTrainer,
                "validator": RecValidator,
                "predictor": RecPredictor,
            }
        }
