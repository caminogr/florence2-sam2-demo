"""Florence-2 model wrapper for image captioning and object detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/Florence-2-large"


@dataclass
class DetectionResult:
    """Result from Florence-2 object detection."""

    caption: str = ""
    bboxes: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


class Florence2Model:
    """Wrapper around Microsoft Florence-2 for captioning and object detection.

    Args:
        model_id: HuggingFace model identifier.
        device: Device to run inference on. Auto-detected if None.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        """Lazy-load model and processor."""
        if self._model is not None:
            return
        logger.info("Loading Florence-2 model: %s on %s", self.model_id, self.device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        logger.info("Florence-2 model loaded successfully.")

    def _run_task(self, image: Image.Image, task_prompt: str) -> dict[str, Any]:
        """Run a Florence-2 task and return parsed results."""
        self._load()
        inputs = self._processor(text=task_prompt, images=image, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed: dict[str, Any] = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return parsed

    def caption(self, image: Image.Image) -> str:
        """Generate a caption for the image.

        Args:
            image: PIL Image to caption.

        Returns:
            Caption string.
        """
        result = self._run_task(image, "<MORE_DETAILED_CAPTION>")
        return result.get("<MORE_DETAILED_CAPTION>", "")

    def detect(self, image: Image.Image) -> DetectionResult:
        """Detect objects and generate a caption.

        Args:
            image: PIL Image to process.

        Returns:
            DetectionResult with caption, bboxes, and labels.
        """
        caption = self.caption(image)
        od_result = self._run_task(image, "<OD>")

        od_data = od_result.get("<OD>", {})
        bboxes: list[list[float]] = od_data.get("bboxes", [])
        labels: list[str] = od_data.get("labels", [])

        logger.info("Detected %d objects.", len(bboxes))
        return DetectionResult(caption=caption, bboxes=bboxes, labels=labels)
