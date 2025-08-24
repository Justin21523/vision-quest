# fronted
# Gradio app - see phase1_gradio_app artifact
"""
FastAPI main application for VisionQuest.
Provides multimodal AI capabilities with RESTful API.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
from gradio.themes import Soft
import uvicorn

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from frontend.gradio_app.components.caption_ui import create_caption_interface
from frontend.gradio_app.components.vqa_ui import create_vqa_interface
from backend.app.utils import api_call, encode_image
from backend.app.core.config import settings
from backend.app.api.v1 import health

# Configuration
API_BASE = "http://localhost:8000/api/v1"


def create_main_interface():
    """Create the main Gradio interface with tabs"""

    with gr.Blocks(title="VisionQuest - 多模態 AI 工具箱", theme=Soft()) as demo:

        gr.Markdown(
            """
            # 🎯 VisionQuest - Vision + Language AI

            **Phase 2 Demo**: 圖像理解與視覺問答系統
            - 📸 **Caption**: BLIP-2 圖像描述生成
            - 🤔 **VQA**: LLaVA 視覺問答 (支援中英文)
            - 💡 **Chat**: 文字推理對話 (即將推出)
            """
        )

        with gr.Tab("📸 圖像描述 (Caption)"):
            caption_interface = create_caption_interface()

        with gr.Tab("🤔 視覺問答 (VQA)"):
            vqa_interface = create_vqa_interface()

        with gr.Tab("💡 文字聊天 (Chat)"):
            gr.Markdown("### 🚧 開發中 - Phase 3 即將推出")
            gr.Textbox(
                value="Phase 3 將支援 Qwen/Llama 聊天推理功能", interactive=False
            )

        with gr.Tab("🔍 系統狀態"):
            create_health_interface()

    return demo


def create_health_interface():
    """Create system health monitoring interface"""

    with gr.Column():
        gr.Markdown("### 系統健康狀態")

        status_output = gr.JSON(label="系統狀態")
        refresh_btn = gr.Button("🔄 刷新狀態", variant="secondary")

        def get_health_status():
            try:
                response = requests.get(f"{API_BASE}/health")
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"API 錯誤: {response.status_code}"}
            except Exception as e:
                return {"error": f"連接失敗: {str(e)}"}

        refresh_btn.click(fn=get_health_status, outputs=status_output)

        # Auto-refresh on load
        demo.load(fn=get_health_status, outputs=status_output)


if __name__ == "__main__":
    demo = create_main_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
