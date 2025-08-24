# scripts/download_models.py
#!/usr/bin/env python3
"""
下載和設置 AI 模型
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import argparse


def download_model(model_name, cache_dir="./models", force_download=False):
    """Download model from Hugging Face"""
    try:
        print(f"📥 下載模型: {model_name}")

        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=True,
        )

        print(f"✅ 模型下載完成: {local_path}")
        return local_path

    except Exception as e:
        print(f"❌ 模型下載失敗: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="下載 VisionQuest 模型")
    parser.add_argument("--model", help="指定模型名稱")
    parser.add_argument("--all", action="store_true", help="下載所有預設模型")
    parser.add_argument("--cache-dir", default="./models", help="模型快取目錄")
    parser.add_argument("--force", action="store_true", help="強制重新下載")

    args = parser.parse_args()

    # Default models for Phase 2
    default_models = {
        "caption": "Salesforce/blip2-opt-2.7b",
        "vqa": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llm": "Qwen/Qwen2-7B-Instruct",
        "embeddings": "BAAI/bge-m3",
    }

    if args.model:
        download_model(args.model, args.cache_dir, args.force)
    elif args.all:
        print("📦 下載所有預設模型...")
        for model_type, model_name in default_models.items():
            print(f"\n🔄 下載 {model_type} 模型...")
            download_model(model_name, args.cache_dir, args.force)
    else:
        print("可用的預設模型:")
        for model_type, model_name in default_models.items():
            print(f"  {model_type}: {model_name}")
        print("\n使用 --all 下載所有模型，或 --model <模型名稱> 下載特定模型")


if __name__ == "__main__":
    main()
