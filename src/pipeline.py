"""Combined Florence-2 + SAM 2 recognition pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.florence2_model import Florence2Model, DetectionResult
from src.sam2_model import SAM2Model, SegmentationResult
from src.visualization import Visualizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full pipeline output."""

    caption: str
    detection: DetectionResult
    segmentation: SegmentationResult
    visualization_detection: Image.Image
    visualization_segmentation: Image.Image


class RecognitionPipeline:
    """End-to-end image recognition pipeline.

    Runs Florence-2 for captioning + detection, then SAM 2 for segmentation,
    and produces annotated visualizations.

    Args:
        florence2_model_id: HuggingFace model ID for Florence-2.
        sam2_checkpoint: HuggingFace checkpoint for SAM 2.
        device: Device string. Auto-detected if None.
    """

    def __init__(
        self,
        florence2_model_id: str = "microsoft/Florence-2-large",
        sam2_checkpoint: str = "facebook/sam2-hiera-large",
        device: str | None = None,
    ) -> None:
        self.florence2 = Florence2Model(model_id=florence2_model_id, device=device)
        self.sam2 = SAM2Model(model_id=sam2_checkpoint, device=device)
        self.visualizer = Visualizer()

    def run(self, image: Image.Image) -> PipelineResult:
        """Run the full pipeline on an image.

        Args:
            image: PIL Image to process.

        Returns:
            PipelineResult with all outputs and visualizations.
        """
        image = image.convert("RGB")
        logger.info("Running Florence-2 detection...")
        detection = self.florence2.detect(image)

        logger.info("Running SAM 2 segmentation on %d objects...", len(detection.bboxes))
        segmentation = self.sam2.segment_from_bboxes(
            image, detection.bboxes, detection.labels
        )

        vis_det = self.visualizer.draw_detections(image, detection)
        vis_seg = self.visualizer.draw_segmentation(image, segmentation)

        return PipelineResult(
            caption=detection.caption,
            detection=detection,
            segmentation=segmentation,
            visualization_detection=vis_det,
            visualization_segmentation=vis_seg,
        )

    def caption_only(self, image: Image.Image) -> str:
        """Run captioning only.

        Args:
            image: PIL Image.

        Returns:
            Caption string.
        """
        return self.florence2.caption(image.convert("RGB"))
