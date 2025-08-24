
# backend/tests/test_api/test_vision_endpoints.py
import pytest
import requests
from PIL import Image
import io
import time

API_BASE = "http://localhost:8000/api/v1"

@pytest.fixture
def test_image():
    """Create a test image"""
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_health_endpoint():
    """Test health check endpoint"""
    response = requests.get(f"{API_BASE}/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "models" in data
    assert "system" in data
    assert "gpu" in data
    assert "version" in data

def test_caption_endpoint(test_image):
    """Test caption generation endpoint"""
    files = {'file': ('test.png', test_image, 'image/png')}
    data = {
        'max_length': 50,
        'num_beams': 3,
        'temperature': 1.0
    }

    response = requests.post(f"{API_BASE}/caption/", files=files, data=data)

    assert response.status_code == 200
    result = response.json()

    # Check response structure
    assert "caption" in result
    assert "confidence" in result
    assert "model_used" in result
    assert isinstance(result["caption"], str)
    assert 0 <= result["confidence"] <= 1

    print(f"✅ Caption test passed: {result['caption']}")

def test_vqa_endpoint(test_image):
    """Test VQA endpoint"""
    files = {'file': ('test.png', test_image, 'image/png')}
    data = {
        'question': 'What color is this image?',
        'lang': 'en',
        'max_length': 50
    }

    response = requests.post(f"{API_BASE}/vqa/", files=files, data=data)

    assert response.status_code == 200
    result = response.json()

    # Check response structure
    assert "answer" in result
    assert "question" in result
    assert "language" in result
    assert "confidence" in result
    assert isinstance(result["answer"], str)

    print(f"✅ VQA test passed: Q: {result['question']} A: {result['answer']}")

def test_vqa_chinese(test_image):
    """Test VQA with Chinese"""
    files = {'file': ('test.png', test_image, 'image/png')}
    data = {
        'question': '這張圖片是什麼顏色？',
        'lang': 'zh-tw',
        'max_length': 50
    }

    response = requests.post(f"{API_BASE}/vqa/", files=files, data=data)

    assert response.status_code == 200
    result = response.json()

    assert result["language"] == "zh-tw"
    print(f"✅ Chinese VQA test passed: {result['answer']}")

def test_invalid_image():
    """Test with invalid image"""
    files = {'file': ('test.txt', io.StringIO("not an image"), 'text/plain')}

    response = requests.post(f"{API_BASE}/caption/", files=files)

    assert response.status_code == 400
    print("✅ Invalid image test passed")

def test_missing_question():
    """Test VQA without question"""
    files = {'file': ('test.png', Image.new('RGB', (100, 100)), 'image/png')}
    data = {'lang': 'en'}

    response = requests.post(f"{API_BASE}/vqa/", files=files, data=data)

    assert response.status_code == 422  # Validation error
    print("✅ Missing question test passed")

@pytest.mark.performance
def test_caption_performance(test_image):
    """Test caption generation performance"""
    files = {'file': ('test.png', test_image, 'image/png')}
    data = {'max_length': 30, 'num_beams': 1}  # Fast settings

    start_time = time.time()
    response = requests.post(f"{API_BASE}/caption/", files=files, data=data, timeout=30)
    end_time = time.time()

    assert response.status_code == 200
    processing_time = end_time - start_time

    # Should complete within 30 seconds (adjust based on hardware)
    assert processing_time < 30
    print(f"✅ Performance test passed: {processing_time:.2f}s")

# backend/tests/conftest.py
import pytest
import subprocess
import time
import requests
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def ensure_api_server():
    """Ensure API server is running for tests"""
    try:
        # Check if server is already running
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server already running")
            return
    except requests.exceptions.RequestException:
        pass

    # Start server if not running
    print("🚀 Starting API server for tests...")

    # Change to backend directory
    backend_dir = Path(__file__).parent.parent

    # Start server process
    process = subprocess.Popen(
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to start
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
            if response.status_code == 200:
                print("✅ API server started successfully")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        process.terminate()
        raise RuntimeError("Failed to start API server for tests")

    # Clean up after tests
    yield

    print("🛑 Stopping test API server...")
    process.terminate()
    process.wait()

# scripts/run_smoke_tests.py
#!/usr/bin/env python3
"""
Phase 2 煙霧測試執行器
"""
import subprocess
import sys
from pathlib import Path
import requests
import time

def check_dependencies():
    """檢查測試依賴"""
    try:
        import pytest
        import requests
        import PIL
        print("✅ 測試依賴已安裝")
        return True
    except ImportError as e:
        print(f"❌ 缺少測試依賴: {e}")
        print("請執行: pip install pytest requests pillow")
        return False

def check_api_server():
    """檢查 API 服務器"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API 服務器運行正常")
            return True
        else:
            print(f"⚠️  API 服務器回應異常: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ 無法連接到 API 服務器")
        print("請先啟動後端: uvicorn backend.app.main:app --reload")
        return False

def run_basic_tests():
    """執行基礎 API 測試"""
    print("🧪 執行基礎 API 測試...")

    backend_dir = Path(__file__).parent.parent / "backend"

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_api/test_vision_endpoints.py::test_health_endpoint",
        "-v"
    ]

    result = subprocess.run(cmd, cwd=backend_dir)
    return result.returncode == 0

def run_vision_tests():
    """執行視覺功能測試"""
    print("👁️  執行視覺功能測試...")

    backend_dir = Path(__file__).parent.parent / "backend"

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_api/test_vision_endpoints.py",
        "-v", "--tb=short"
    ]

    result = subprocess.run(cmd, cwd=backend_dir)
    return result.returncode == 0

def run_performance_tests():
    """執行效能測試"""
    print("⚡ 執行效能測試...")

    backend_dir = Path(__file__).parent.parent / "backend"

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_api/test_vision_endpoints.py::test_caption_performance",
        "-v", "-m", "performance"
    ]

    result = subprocess.run(cmd, cwd=backend_dir)
    return result.returncode == 0

def test_gradio_ui():
    """測試 Gradio UI 可達性"""
    print("🎛️  測試 Gradio UI...")

    try:
        response = requests.get("http://localhost:7860", timeout=10)
        if response.status_code == 200:
            print("✅ Gradio UI 運行正常")
            return True
        else:
            print(f"⚠️  Gradio UI 回應異常: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ 無法連接到 Gradio UI")
        print("請啟動 Gradio: cd frontend/gradio_app && python app.py")
        return False

def main():
    """主測試流程"""
    print("🎯 VisionQuest Phase 2 煙霧測試")
    print("=" * 50)

    # Check prerequisites
    if not check_dependencies():
        return False

    if not check_api_server():
        return False

    # Run test suites
    test_results = []

    # Basic tests
    test_results.append(("基礎 API 測試", run_basic_tests()))

    # Vision tests
    test_results.append(("視覺功能測試", run_vision_tests()))

    # Performance tests (optional)
    try:
        test_results.append(("效能測試", run_performance_tests()))
    except Exception as e:
        print(f"⚠️  效能測試跳過: {e}")

    # UI tests (optional)
    test_results.append(("Gradio UI 測試", test_gradio_ui()))

    # Summary
    print("\n" + "=" * 50)
    print("📊 測試結果摘要:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n📈 總計: {passed}/{total} 測試通過")

    if passed == total:
        print("🎉 所有測試通過！Phase 2 準備就緒")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查相關功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# backend/tests/test_services/test_caption_service.py
import pytest
from PIL import Image
from app.services.caption_service import CaptionService
import asyncio

@pytest.fixture
def caption_service():
    """Create caption service instance"""
    return CaptionService()

@pytest.fixture
def test_image():
    """Create test image"""
    return Image.new('RGB', (224, 224), color='blue')

@pytest.mark.asyncio
async def test_caption_generation(caption_service, test_image):
    """Test caption generation"""
    result = await caption_service.generate_caption(
        image=test_image,
        max_length=30,
        num_beams=3
    )

    assert "caption" in result
    assert "confidence" in result
    assert "model_used" in result
    assert isinstance(result["caption"], str)
    assert len(result["caption"]) > 0

@pytest.mark.asyncio
async def test_caption_parameters(caption_service, test_image):
    """Test different parameters"""
    # Test with different max_length
    result1 = await caption_service.generate_caption(
        image=test_image,
        max_length=10
    )

    result2 = await caption_service.generate_caption(
        image=test_image,
        max_length=50
    )

    # Longer max_length might produce longer captions
    assert len(result1["caption"]) <= len(result2["caption"]) + 20  # Some tolerance

# requirements.txt files for different components

# backend/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic[email]==2.5.0
python-multipart==0.0.6
aiofiles==23.2.1
python-dotenv==1.0.0

# AI/ML dependencies
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.36.0
accelerate>=0.25.0
peft>=0.7.0
datasets>=2.15.0

# Vision processing
Pillow>=10.1.0
opencv-python>=4.8.1.78

# Vector database
faiss-cpu>=1.7.4

# Development tools
pytest>=7.4.3
pytest-asyncio>=0.21.1
httpx>=0.25.2
black>=23.11.0
isort>=5.12.0
mypy>=1.7.1
ruff>=0.1.6

# frontend/gradio_app/requirements.txt
gradio>=4.7.1
requests>=2.31.0
Pillow>=10.1.0

# frontend/react_app/package.json
{
  "name": "visionquest-react",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.37",
    "@types/react-dom": "^18.2.15",
    "@typescript-eslint/eslint-plugin": "^6.10.0",
    "@typescript-eslint/parser": "^6.10.0",
    "@vitejs/plugin-react": "^4.1.0",
    "eslint": "^8.53.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.4",
    "typescript": "^5.2.2",
    "vite": "^5.0.0"
  }
}

# frontend/pyqt_app/requirements.txt
PyQt6>=6.6.0
requests>=2.31.0
Pillow>=10.1.0