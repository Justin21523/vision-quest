# scripts/setup_env.py
#!/usr/bin/env python3
"""
VisionQuest 環境設置腳本
"""
import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, check=True):
    """Execute command and return result"""
    print(f"執行: {cmd}")
    result = subprocess.run(
        cmd, shell=True, check=check, capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr and check:
        print(f"錯誤: {result.stderr}")
    return result


def setup_conda_env():
    """Setup conda environment"""
    print("🔧 設置 Conda 環境...")

    # Check if conda is available
    try:
        run_command("conda --version")
    except subprocess.CalledProcessError:
        print("❌ Conda 未安裝，請先安裝 Anaconda 或 Miniconda")
        return False

    # Create environment
    env_name = "multi-modal-lab"

    # Check if environment exists
    result = run_command(f"conda env list | grep {env_name}", check=False)
    if result.returncode == 0:
        print(f"⚠️  環境 {env_name} 已存在，是否重新建立？(y/N)")
        if input().lower() != "y":
            return True
        run_command(f"conda env remove -n {env_name} -y")

    # Create new environment
    python_version = "3.10"
    run_command(f"conda create -n {env_name} python={python_version} -y")

    print(f"✅ Conda 環境 {env_name} 建立完成")
    return True


def install_pytorch():
    """Install PyTorch with CUDA support if available"""
    print("🔥 安裝 PyTorch...")

    # Detect CUDA availability
    if platform.system() == "Windows":
        cuda_check = "nvidia-smi"
    else:
        cuda_check = "which nvidia-smi"

    try:
        run_command(cuda_check)
        print("🎮 檢測到 NVIDIA GPU，安裝 CUDA 版本...")
        pytorch_cmd = "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
    except subprocess.CalledProcessError:
        print("💻 未檢測到 GPU，安裝 CPU 版本...")
        pytorch_cmd = (
            "conda install pytorch torchvision torchaudio cpuonly -c pytorch -y"
        )

    run_command(pytorch_cmd)
    print("✅ PyTorch 安裝完成")


def install_dependencies():
    """Install Python dependencies"""
    print("📦 安裝 Python 依賴...")

    # Backend dependencies
    backend_deps = [
        "pip install fastapi uvicorn[standard]",
        "pip install transformers accelerate",
        "pip install Pillow opencv-python",
        "pip install gradio",
        "pip install requests aiofiles",
        "pip install pydantic[email] python-multipart",
        "pip install faiss-cpu",  # Vector database
        "pip install python-dotenv",
        "pip install pytest pytest-asyncio httpx",
        "pip install black isort mypy ruff",  # Development tools
    ]

    for cmd in backend_deps:
        run_command(cmd)

    # Optional: PyQt for desktop app
    print("🖥️  安裝 PyQt (桌面應用)...")
    try:
        run_command("pip install PyQt6")
        print("✅ PyQt6 安裝完成")
    except subprocess.CalledProcessError:
        print("⚠️  PyQt6 安裝失敗，跳過桌面應用支援")


def setup_directories():
    """Create necessary directories"""
    print("📁 建立目錄結構...")

    directories = [
        "models",
        "data/uploads",
        "data/outputs",
        "data/kb",
        "data/game_saves",
        "logs",
        "frontend/gradio_app/examples",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📂 建立: {dir_path}")


def setup_env_file():
    """Create .env file from example"""
    print("⚙️  設置環境變數...")

    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil

            shutil.copy(".env.example", ".env")
            print("✅ .env 檔案已建立，請根據需要修改設定")
        else:
            print("⚠️  .env.example 檔案不存在")
    else:
        print("ℹ️  .env 檔案已存在")


def setup_node_env():
    """Setup Node.js environment for React app"""
    print("🟢 設置 Node.js 環境...")

    # Check Node.js version
    try:
        result = run_command("node --version")
        version = result.stdout.strip()
        if version:
            print(f"✅ Node.js 版本: {version}")

        # Install React app dependencies
        if Path("frontend/react_app/package.json").exists():
            print("📦 安裝 React 依賴...")
            run_command("cd frontend/react_app && npm install")
            print("✅ React 依賴安裝完成")

    except subprocess.CalledProcessError:
        print("⚠️  Node.js 未安裝，跳過 React 應用設置")
        print("請安裝 Node.js 18+ 以使用 Web 介面")


def download_example_images():
    """Download example images for testing"""
    print("🖼️  下載範例圖片...")

    examples_dir = Path("frontend/gradio_app/examples")
    examples_dir.mkdir(exist_ok=True)

    # Create placeholder images if they don't exist
    example_files = ["cat.jpg", "landscape.jpg", "people.jpg"]

    for filename in example_files:
        filepath = examples_dir / filename
        if not filepath.exists():
            # Create a simple placeholder
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new("RGB", (400, 300), color="lightblue")
            draw = ImageDraw.Draw(img)

            try:
                # Try to use a font
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            text = f"範例圖片\n{filename}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            position = ((400 - text_width) // 2, (300 - text_height) // 2)
            draw.text(position, text, fill="darkblue", font=font)

            img.save(filepath)
            print(f"📁 建立範例圖片: {filepath}")


def main():
    """Main setup function"""
    print("🎯 VisionQuest 環境設置開始...")
    print("=" * 50)

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    try:
        # Setup steps
        if not setup_conda_env():
            print("❌ Conda 環境設置失敗")
            return

        print("\n請啟動 conda 環境並重新執行:")
        print(f"conda activate multi-modal-lab")
        print("python scripts/setup_env.py --install-deps")

    except KeyboardInterrupt:
        print("\n⚠️  設置已取消")
    except Exception as e:
        print(f"❌ 設置過程中發生錯誤: {e}")


def install_deps_only():
    """Install dependencies only (for activated conda env)"""
    print("📦 在已啟動的環境中安裝依賴...")

    install_pytorch()
    install_dependencies()
    setup_directories()
    setup_env_file()
    setup_node_env()
    download_example_images()

    print("\n🎉 VisionQuest 環境設置完成!")
    print("\n▶️  啟動指令:")
    print("   後端: uvicorn backend.app.main:app --reload")
    print("   Gradio: python frontend/gradio_app/app.py")
    print("   React: cd frontend/react_app && npm run dev")
    print("   PyQt: python frontend/pyqt_app/app.py")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--install-deps":
        install_deps_only()
    else:
        main()
