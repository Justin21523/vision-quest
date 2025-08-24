# frontend/gradio_app/components/vqa_ui.py
import gradio as gr
import requests
import io
from frontend.gradio_app.utils import handle_api_error

API_BASE = "http://localhost:8000/api/v1"


def create_vqa_interface():
    """Create visual question answering interface"""

    with gr.Column():
        gr.Markdown("### 🤔 LLaVA 視覺問答系統")

        with gr.Row():
            with gr.Column(scale=2):
                vqa_image = gr.Image(label="上傳圖片", type="pil", height=300)

                question_input = gr.Textbox(
                    label="問題",
                    placeholder="請描述這張圖片中的內容... (支援中英文)",
                    lines=2,
                )

                with gr.Row():
                    language_select = gr.Dropdown(
                        choices=[
                            ("English", "en"),
                            ("繁體中文", "zh-tw"),
                            ("简体中文", "zh-cn"),
                        ],
                        value="en",
                        label="語言",
                    )

                    max_length_vqa = gr.Slider(
                        minimum=10,
                        maximum=300,
                        value=100,
                        step=10,
                        label="最大回答長度",
                    )

                ask_btn = gr.Button("❓ 提問", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(label="AI 回答", lines=6, max_lines=12)

                vqa_confidence = gr.Number(label="信心度", precision=3)

                vqa_model_info = gr.Textbox(label="使用模型", interactive=False)

        # Predefined questions
        gr.Examples(
            examples=[
                ["這張圖片中有什麼？", "zh-tw"],
                ["What is the main subject in this image?", "en"],
                ["圖片中的人在做什麼？", "zh-tw"],
                ["What colors do you see?", "en"],
                ["這是在哪裡拍攝的？", "zh-tw"],
            ],
            inputs=[question_input, language_select],
            label="常見問題範例",
        )

        def answer_question(image, question, language, max_len):
            """Call VQA API"""
            if image is None:
                return "請上傳圖片", 0.0, ""
            if not question.strip():
                return "請輸入問題", 0.0, ""

            try:
                # Prepare image
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                files = {"file": ("image.png", img_byte_arr, "image/png")}

                data = {
                    "question": question,
                    "lang": language,
                    "max_length": int(max_len),
                }

                response = requests.post(
                    f"{API_BASE}/vqa/", files=files, data=data, timeout=45
                )

                if response.status_code == 200:
                    result = response.json()
                    return (
                        result["answer"],
                        result["confidence"],
                        result["model_used"],
                    )
                else:
                    error_msg = handle_api_error(response)
                    return f"API 錯誤: {error_msg}", 0.0, ""

            except Exception as e:
                return f"請求失敗: {str(e)}", 0.0, ""

        ask_btn.click(
            fn=answer_question,
            inputs=[vqa_image, question_input, language_select, max_length_vqa],
            outputs=[answer_output, vqa_confidence, vqa_model_info],
        )
