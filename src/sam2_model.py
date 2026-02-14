"""SAM 2 model wrapper for segmentation mask generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Segmentation masks for detected objects."""

    masks: list[np.ndarray] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)


class SAM2Model:
    """Wrapper around Meta's SAM 2 for segmentation mask generation.

    Uses bounding box prompts from Florence-2 detection results to generate
    high-quality segmentation masks.

    Args:
        model_cfg: SAM 2 model config name.
        checkpoint: Path or HuggingFace checkpoint for SAM 2.
        device: Device to run inference on. Auto-detected if None.
    """

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_large",
        checkpoint: str = "facebook/sam2-hiera-large",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self._predictor: Any = None

    def _load(self) -> None:
        """Lazy-load SAM 2 predictor."""
        if self._predictor is not None:
            return
        logger.info("Loading SAM 2 model: %s on %s", self.checkpoint, self.device)
        try:
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2_hf(self.checkpoint, device=self.device)
            self._predictor = SAM2ImagePredictor(model)
        except ImportError:
            logger.warning(
                "sam2 package not found. Trying alternative import from transformers..."
            )
            self._load_fallback()
        logger.info("SAM 2 model loaded successfully.")

    def _load_fallback(self) -> None:
        """Fallback: use transformers SAM2 if the official package isn't installed."""
        try:
            from transformers import Sam2Model as HFSam2Model
            from transformers import Sam2Processor

            self._hf_model = HFSam2Model.from_pretrained(self.checkpoint).to(self.device)
            self._hf_processor = Sam2Processor.from_pretrained(self.checkpoint)
            self._predictor = "hf_fallback"
        except Exception as e:
            logger.error("Could not load SAM 2 via any method: %s", e)
            raise RuntimeError(
                "SAM 2 is not available. Install with: pip install sam2 "
                "or pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            ) from e

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

        image_np = np.array(image.convert("RGB"))
        masks_out: list[np.ndarray] = []

        if self._predictor == "hf_fallback":
            masks_out = self._segment_hf(image, bboxes)
        else:
            masks_out = self._segment_native(image_np, bboxes)

        logger.info("Generated %d segmentation masks.", len(masks_out))
        return SegmentationResult(
            masks=masks_out,
            labels=labels[: len(masks_out)],
            bboxes=bboxes[: len(masks_out)],
        )

    def _segment_native(
        self, image_np: np.ndarray, bboxes: list[list[float]]
    ) -> list[np.ndarray]:
        """Segment using the native SAM 2 predictor."""
        self._predictor.set_image(image_np)
        masks_out: list[np.ndarray] = []

        for bbox in bboxes:
            try:
                box_np = np.array(bbox, dtype=np.float32)
                masks, scores, _ = self._predictor.predict(
                    box=box_np,
                    multimask_output=True,
                )
                # Pick the mask with the highest score
                best_idx = int(np.argmax(scores))
                masks_out.append(masks[best_idx])
            except Exception as e:
                logger.warning("Failed to segment bbox %s: %s", bbox, e)
                continue

        return masks_out

    def _segment_hf(
        self, image: Image.Image, bboxes: list[list[float]]
    ) -> list[np.ndarray]:
        """Segment using HuggingFace transformers SAM 2."""
        masks_out: list[np.ndarray] = []

        for bbox in bboxes:
            try:
                inputs = self._hf_processor(
                    images=image,
                    input_boxes=[[[bbox]]],
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    outputs = self._hf_model(**inputs)
                pred_masks = self._hf_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"],
                )
                mask = pred_masks[0][0][0].cpu().numpy()
                masks_out.append(mask)
            except Exception as e:
                logger.warning("HF fallback failed for bbox %s: %s", bbox, e)
                continue

        return masks_out
