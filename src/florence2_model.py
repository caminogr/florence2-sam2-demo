"""Florence-2 model wrapper for image captioning and object detection."""

from __future__ import annotations

import importlib.machinery
import logging
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def _register_flash_attn_mock() -> None:
    """Register mock flash_attn modules to prevent ImportError on CPU.

    Florence-2's custom modeling code imports flash_attn at the module level.
    On CPU-only environments (e.g. HF Spaces free tier), flash_attn cannot be
    installed because it requires CUDA.  By registering stub modules before
    ``from_pretrained`` triggers the import, the ImportError is avoided.
    The stubs are never actually called because we set
    ``attn_implementation="eager"``.
    """
    if torch.cuda.is_available():
        return

    def _unavailable(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise RuntimeError("flash_attn is not available on CPU")

    _attrs = (
        "flash_attn_func",
        "flash_attn_varlen_func",
        "index_first_axis",
        "pad_input",
        "unpad_input",
    )
    for mod_name in ("flash_attn", "flash_attn.bert_padding"):
        if mod_name not in sys.modules:
            mock = types.ModuleType(mod_name)
            mock.__path__ = []  # type: ignore[attr-defined]
            mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
            for attr in _attrs:
                setattr(mock, attr, _unavailable)
            sys.modules[mod_name] = mock

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/Florence-2-large"
MODEL_REVISION = "21a599d414c4d928c9032694c424fb94458e3594"


@dataclass
class DetectionResult:
    """Result from Florence-2 object detection."""

    caption: str = ""
    bboxes: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


class Florence2Model:
    """Wrapper around Microsoft Florence-2 for captioning and object detection."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.revision = revision
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        """Lazy-load model and processor."""
        if self._model is not None:
            return
        _register_flash_attn_mock()
        logger.info(
            "Loading Florence-2 model: %s (rev: %s) on %s",
            self.model_id, self.revision, self.device,
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa" if self.device == "cuda" else "eager",
        ).to(self.device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
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
