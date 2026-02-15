"""Visualization utilities for drawing bboxes, masks, and captions on images."""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from src.florence2_model import DetectionResult
    from src.sam_model import SegmentationResult


def _generate_colors(n: int) -> list[tuple[int, int, int]]:
    """Generate n visually distinct colors.

    Args:
        n: Number of colors to generate.

    Returns:
        List of (R, G, B) tuples.
    """
    if n == 0:
        return []
    colors: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _get_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font, fall back to default."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


class Visualizer:
    """Draw detection and segmentation results onto images."""

    def __init__(self, mask_alpha: float = 0.4, font_size: int = 16) -> None:
        """Initialize visualizer.

        Args:
            mask_alpha: Opacity for mask overlay (0-1).
            font_size: Font size for labels.
        """
        self.mask_alpha = mask_alpha
        self.font = _get_font(font_size)

    def draw_detections(
        self, image: Image.Image, detection: DetectionResult
    ) -> Image.Image:
        """Draw bounding boxes and labels on image.

        Args:
            image: Base PIL Image.
            detection: DetectionResult from Florence-2.

        Returns:
            New image with bboxes and labels drawn.
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        colors = _generate_colors(len(detection.bboxes))

        for bbox, label, color in zip(detection.bboxes, detection.labels, colors):
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label background
            text = label
            text_bbox = draw.textbbox((0, 0), text, font=self.font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th - 2), text, fill="white", font=self.font)

        return img

    def draw_segmentation(
        self, image: Image.Image, segmentation: SegmentationResult
    ) -> Image.Image:
        """Draw segmentation masks overlaid on image.

        Args:
            image: Base PIL Image.
            segmentation: SegmentationResult from SAM.

        Returns:
            New image with colored masks overlaid.
        """
        img_np = np.array(image.copy(), dtype=np.float32)
        colors = _generate_colors(len(segmentation.masks))

        for mask, color in zip(segmentation.masks, colors):
            if mask.shape[:2] != img_np.shape[:2]:
                # Resize mask to match image
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize(
                    (image.width, image.height), Image.NEAREST
                )
                mask = np.array(mask_img) > 127

            mask_bool = mask.astype(bool)
            overlay = np.array(color, dtype=np.float32)
            img_np[mask_bool] = (
                img_np[mask_bool] * (1 - self.mask_alpha)
                + overlay * self.mask_alpha
            )

        result = Image.fromarray(img_np.astype(np.uint8))

        # Draw labels at bbox centers
        draw = ImageDraw.Draw(result)
        for bbox, label, color in zip(
            segmentation.bboxes, segmentation.labels, colors
        ):
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            text_bbox = draw.textbbox((0, 0), label, font=self.font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            draw.rectangle(
                [cx - tw / 2 - 2, cy - th / 2 - 2, cx + tw / 2 + 2, cy + th / 2 + 2],
                fill=color,
            )
            draw.text(
                (cx - tw / 2, cy - th / 2), label, fill="white", font=self.font
            )

        return result

    def draw_caption(self, image: Image.Image, caption: str) -> Image.Image:
        """Draw caption text at the top of the image.

        Args:
            image: Base PIL Image.
            caption: Caption text to draw.

        Returns:
            New image with caption banner.
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        text_bbox = draw.textbbox((0, 0), caption, font=self.font)
        th = text_bbox[3] - text_bbox[1]

        # Semi-transparent banner
        banner = Image.new("RGBA", (img.width, th + 16), (0, 0, 0, 180))
        img_rgba = img.convert("RGBA")
        img_rgba.paste(banner, (0, 0), banner)
        draw = ImageDraw.Draw(img_rgba)
        draw.text((8, 8), caption, fill="white", font=self.font)

        return img_rgba.convert("RGB")
