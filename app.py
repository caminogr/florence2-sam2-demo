"""Gradio demo app for Florence-2 + SAM image recognition pipeline."""

from __future__ import annotations

import logging

import gradio as gr
from PIL import Image

from src.pipeline import RecognitionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Global pipeline instance (lazy-loaded on first use)
_pipeline: RecognitionPipeline | None = None


def get_pipeline() -> RecognitionPipeline:
    """Get or create the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing pipeline...")
        _pipeline = RecognitionPipeline()
    return _pipeline


def process_image(
    image: Image.Image,
) -> tuple[str, Image.Image | None, Image.Image | None]:
    """Process an uploaded image through the full pipeline.

    Args:
        image: Uploaded PIL Image.

    Returns:
        Tuple of (caption, detection_image, segmentation_image).
    """
    if image is None:
        return "Please upload an image.", None, None

    try:
        pipeline = get_pipeline()
        result = pipeline.run(image)

        info = f"**Caption:** {result.caption}\n\n"
        info += f"**Objects detected:** {len(result.detection.bboxes)}\n\n"
        if result.detection.labels:
            label_counts: dict[str, int] = {}
            for label in result.detection.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            info += "**Detected objects:** "
            info += ", ".join(f"{label} ({count})" for label, count in label_counts.items())

        return info, result.visualization_detection, result.visualization_segmentation

    except Exception as e:
        logger.exception("Error processing image")
        return f"Error: {e}", None, None


def caption_only(image: Image.Image) -> str:
    """Generate only a caption for the image.

    Args:
        image: Uploaded PIL Image.

    Returns:
        Caption string.
    """
    if image is None:
        return "Please upload an image."
    try:
        pipeline = get_pipeline()
        return pipeline.caption_only(image)
    except Exception as e:
        logger.exception("Error generating caption")
        return f"Error: {e}"


def build_demo() -> gr.Blocks:
    """Build the Gradio demo interface."""
    with gr.Blocks(
        title="Florence-2 + SAM Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🔍 Florence-2 + SAM Image Recognition\n"
            "Upload an image to get **captions**, **object detection**, "
            "and **segmentation masks**."
        )

        with gr.Tab("Full Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Image")
                    run_btn = gr.Button("🚀 Analyze", variant="primary")

                with gr.Column(scale=2):
                    output_info = gr.Markdown(label="Results")

            with gr.Row():
                output_detection = gr.Image(type="pil", label="Object Detection")
                output_segmentation = gr.Image(type="pil", label="Segmentation Masks")

            run_btn.click(
                fn=process_image,
                inputs=[input_image],
                outputs=[output_info, output_detection, output_segmentation],
            )

        with gr.Tab("Caption Only"):
            with gr.Row():
                caption_input = gr.Image(type="pil", label="Upload Image")
                caption_output = gr.Textbox(label="Caption", lines=3)
            caption_btn = gr.Button("📝 Generate Caption", variant="primary")
            caption_btn.click(
                fn=caption_only,
                inputs=[caption_input],
                outputs=[caption_output],
            )

        gr.Markdown(
            "---\n"
            "**Models:** [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) "
            "(Microsoft) + [SAM](https://github.com/facebookresearch/segment-anything) (Meta)\n\n"
            "Models are downloaded on first run (~4GB). CPU supported; GPU recommended for speed."
        )

    return demo


def main() -> None:
    """Launch the Gradio demo."""
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
