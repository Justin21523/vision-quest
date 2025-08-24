# frontend/gradio_app/components/caption_ui.py
import gradio as gr
import requests
import io
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from frontend.gradio_app.utils import api_call, handle_api_error

API_BASE = "http://localhost:8000/api/v1"


def create_caption_interface():
    """Create image captioning interface"""

    with gr.Column():
        gr.Markdown("### 📸 BLIP-2 圖像描述生成")

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(label="上傳圖片", type="pil", height=300)

                with gr.Row():
                    max_length = gr.Slider(
                        minimum=10, maximum=200, value=50, step=5, label="最大長度"
                    )
                    num_beams = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1, label="束搜索數量"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="溫度"
                    )

                generate_btn = gr.Button("🚀 生成描述", variant="primary")

            with gr.Column(scale=1):
                caption_output = gr.Textbox(
                    label="生成的圖像描述", lines=5, max_lines=10
                )

                confidence_output = gr.Number(label="信心度", precision=3)

                model_info = gr.Textbox(label="使用模型", interactive=False)

        # Examples
        gr.Examples(
            examples=[
                ["./examples/cat.jpg"],
                ["./examples/landscape.jpg"],
                ["./examples/people.jpg"],
            ],
            inputs=image_input,
            label="範例圖片",
        )

        def generate_caption(image, max_len, beams, temp):
            """Call caption API"""
            if image is None:
                return "請上傳圖片", 0.0, ""

            try:
                # Prepare files for upload
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                files = {"file": ("image.png", img_byte_arr, "image/png")}

                data = {
                    "max_length": int(max_len),
                    "num_beams": int(beams),
                    "temperature": float(temp),
                }

                response = requests.post(
                    f"{API_BASE}/caption/", files=files, data=data, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return (
                        result["caption"],
                        result["confidence"],
                        result["model_used"],
                    )
                else:
                    error_msg = handle_api_error(response)
                    return f"API 錯誤: {error_msg}", 0.0, ""

            except Exception as e:
                return f"請求失敗: {str(e)}", 0.0, ""

        generate_btn.click(
            fn=generate_caption,
            inputs=[image_input, max_length, num_beams, temperature],
            outputs=[caption_output, confidence_output, model_info],
        )
