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


def _patch_florence2_config(model: Any) -> None:
    """Patch all config objects in the model to handle missing generation attrs.

    Florence-2's custom config classes don't define attributes that newer
    transformers' generate() expects. We add a __getattr__ fallback that
    returns None for any missing attribute instead of raising AttributeError.
    """
    patched: set[int] = set()

    def _add_fallback(cfg: Any) -> None:
        cls = type(cfg)
        cls_id = id(cls)
        if cls_id in patched:
            return
        # Only patch Florence2-specific config classes
        if "Florence2" not in cls.__name__:
            return

        original_getattr = getattr(cls, "__getattr__", None)

        def safe_getattr(self: Any, name: str) -> Any:
            if name.startswith("_"):
                raise AttributeError(name)
            if original_getattr is not None:
                try:
                    return original_getattr(self, name)
                except AttributeError:
                    pass
            return None

        cls.__getattr__ = safe_getattr
        patched.add(cls_id)
        logger.info("Patched %s for generate() compatibility", cls.__name__)

    # Patch the main config and all sub-configs
    if hasattr(model, "config"):
        _add_fallback(model.config)
        for attr_name in vars(model.config):
            sub = getattr(model.config, attr_name, None)
            if sub is not None and hasattr(sub, "__class__") and "Config" in type(sub).__name__:
                _add_fallback(sub)

    # Also check language_model sub-module
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        _add_fallback(model.language_model.config)


class Florence2Model:
    """Wrapper around Microsoft Florence-2 for captioning and object detection."""

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
        )
        _patch_florence2_config(self._model)
        self._model = self._model.to(self.device)
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
        """Generate a caption for the image."""
        result = self._run_task(image, "<MORE_DETAILED_CAPTION>")
        return result.get("<MORE_DETAILED_CAPTION>", "")

    def detect(self, image: Image.Image) -> DetectionResult:
        """Detect objects and generate a caption."""
        caption = self.caption(image)
        od_result = self._run_task(image, "<OD>")
        od_data = od_result.get("<OD>", {})
        bboxes: list[list[float]] = od_data.get("bboxes", [])
        labels: list[str] = od_data.get("labels", [])
        logger.info("Detected %d objects.", len(bboxes))
        return DetectionResult(caption=caption, bboxes=bboxes, labels=labels)
