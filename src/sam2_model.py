"""SAM 2 model wrapper for segmentation mask generation.

Uses HuggingFace transformers' SAM2 implementation for broad compatibility,
especially on HF Spaces where building the native sam2 package is impractical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor

logger = logging.getLogger(__name__)

SAM2_MODEL_ID = "facebook/sam2-hiera-large"


@dataclass
class SegmentationResult:
    """Segmentation masks for detected objects."""

    masks: list[np.ndarray] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)


class SAM2Model:
    """Wrapper around Meta's SAM 2 for segmentation mask generation.

    Uses HuggingFace transformers backend for compatibility.

    Args:
        model_id: HuggingFace model identifier for SAM 2.
        device: Device to run inference on. Auto-detected if None.
    """

    def __init__(
        self,
        model_id: str = SAM2_MODEL_ID,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        """Lazy-load SAM 2 model and processor via transformers."""
        if self._model is not None:
            return
        logger.info("Loading SAM 2 model: %s on %s", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForMaskGeneration.from_pretrained(self.model_id).to(
            self.device
        )
        self._model.eval()
        logger.info("SAM 2 model loaded successfully.")

    def segment_from_bboxes(
        self,
        image: Image.Image,
        bboxes: list[list[float]],
        labels: list[str],
    ) -> SegmentationResult:
        """Generate segmentation masks using bounding box prompts.

        Args:
            image: PIL Image to segment.
            bboxes: List of [x1, y1, x2, y2] bounding boxes.
            labels: List of label strings corresponding to each bbox.

        Returns:
            SegmentationResult with masks, labels, and bboxes.
        """
        self._load()

        if not bboxes:
            logger.warning("No bounding boxes provided; returning empty result.")
            return SegmentationResult()

        masks_out: list[np.ndarray] = []

        for bbox in bboxes:
            try:
                inputs = self._processor(
                    images=image,
                    input_boxes=[[[bbox]]],
                    return_tensors="pt",
                ).to(self.device)

                with torch.inference_mode():
                    outputs = self._model(**inputs)

                pred_masks = self._processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"],
                )
                # pred_masks shape: [batch, num_masks, H, W] — pick best
                mask_tensor = pred_masks[0][0]  # first batch, first query
                if mask_tensor.ndim == 3:
                    # Multiple mask proposals — use the first (highest confidence)
                    mask_tensor = mask_tensor[0]
                mask_np = mask_tensor.cpu().numpy().astype(bool)
                masks_out.append(mask_np)
            except Exception as e:
                logger.warning("Failed to segment bbox %s: %s", bbox, e)
                continue

        logger.info("Generated %d segmentation masks.", len(masks_out))
        return SegmentationResult(
            masks=masks_out,
            labels=labels[: len(masks_out)],
            bboxes=bboxes[: len(masks_out)],
        )
