# frontend/gradio_app/caption_demo.py
"""
Gradio Caption Demo
Quick UI for testing caption API endpoint
"""

import gradio as gr
import requests
from PIL import Image
import io
import os


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def generate_caption(
    image,
    max_length,
    num_beams,
    temperature,
    enable_safety,
):
    """Call caption API and return result"""
    if image is None:
        return "Please upload an image", {}

    try:
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Call API
        files = {"file": ("image.png", buffer, "image/png")}
        params = {
            "max_length": max_length,
            "num_beams": num_beams,
            "temperature": temperature,
            "enable_safety": enable_safety,
        }

        response = requests.post(
            f"{API_BASE}/api/v1/caption",
            files=files,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()

        # Format output
        caption = result["caption"]
        metadata = {
            "Confidence": f"{result['confidence']:.2%}",
            "Model": result["model"],
            "Safe": "✅" if result["is_safe"] else "❌",
        }

        if result.get("safety_score"):
            metadata["Safety Score"] = f"{result['safety_score']:.2%}"

        return caption, metadata

    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {str(e)}", {}
    except Exception as e:
        return f"❌ Error: {str(e)}", {}


# Build Gradio interface
with gr.Blocks(title="VisionQuest Caption Demo") as demo:
    gr.Markdown("# 🖼️ Image Caption Generator")
    gr.Markdown("Upload an image and generate a descriptive caption using BLIP-2")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
            )

            with gr.Accordion("Advanced Options", open=False):
                max_length = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Max Length",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Num Beams",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                )
                enable_safety = gr.Checkbox(
                    value=True,
                    label="Enable Safety Filter",
                )

            generate_btn = gr.Button("🚀 Generate Caption", variant="primary")

        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="Generated Caption",
                lines=3,
            )
            metadata_output = gr.JSON(label="Metadata")

    # Wire up the function
    generate_btn.click(
        fn=generate_caption,
        inputs=[
            image_input,
            max_length,
            num_beams,
            temperature,
            enable_safety,
        ],
        outputs=[caption_output, metadata_output],
    )

    # Examples
    gr.Examples(
        examples=[
            ["examples/sunset.jpg", 50, 3, 1.0, True],
            ["examples/cat.jpg", 30, 5, 0.8, True],
        ],
        inputs=[image_input, max_length, num_beams, temperature, enable_safety],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
