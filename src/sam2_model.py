"""SAM model wrapper for segmentation mask generation.

Uses transformers pipeline("mask-generation") with SAM (v1) for broad
compatibility with transformers==4.44.2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

logger = logging.getLogger(__name__)

# Use SAM v1 which has stable transformers support
SAM_MODEL_ID = "facebook/sam-vit-base"


@dataclass
class SegmentationResult:
    """Segmentation masks for detected objects."""

    masks: list[np.ndarray] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)


class SAM2Model:
    """Wrapper around SAM for segmentation mask generation.

    Uses SAM v1 (facebook/sam-vit-base) via transformers for stable
    compatibility. Takes bounding box prompts from Florence-2.

    Args:
        model_id: HuggingFace model identifier for SAM.
        device: Device to run inference on. Auto-detected if None.
    """

    def __init__(
        self,
        model_id: str = SAM_MODEL_ID,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        """Lazy-load SAM model and processor."""
        if self._model is not None:
            return
        logger.info("Loading SAM model: %s on %s", self.model_id, self.device)
        self._processor = SamProcessor.from_pretrained(self.model_id)
        self._model = SamModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()
        logger.info("SAM model loaded successfully.")

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
                # SamProcessor expects input_boxes as [[[x1, y1, x2, y2]]]
                inputs = self._processor(
                    images=image,
                    input_boxes=[[[bbox]]],
                    return_tensors="pt",
                ).to(self.device)

                with torch.inference_mode():
                    outputs = self._model(**inputs)

                masks = self._processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )
                # masks[0] shape: [1, 3, H, W] — 3 mask proposals
                mask_tensor = masks[0].squeeze(0)  # [3, H, W]
                # Use iou_scores to pick best mask
                scores = outputs.iou_scores[0][0]  # [3]
                best_idx = int(scores.argmax())
                mask_np = mask_tensor[best_idx].numpy().astype(bool)
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
