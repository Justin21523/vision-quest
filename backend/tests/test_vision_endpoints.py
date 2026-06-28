import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _image_bytes(color: str = "red") -> io.BytesIO:
    image = Image.new("RGB", (224, 224), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_health_endpoint():
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["mock_mode"] is True
    assert "models" in data


def test_caption_endpoint():
    response = client.post(
        "/api/v1/caption/",
        files={"file": ("test.png", _image_bytes(), "image/png")},
        params={"max_length": 50, "num_beams": 3, "temperature": 1.0},
    )
    assert response.status_code == 200
    result = response.json()
    assert "caption" in result
    assert "model_used" in result
    assert 0 <= result["confidence"] <= 1


def test_vqa_endpoint():
    response = client.post(
        "/api/v1/vqa/",
        files={"file": ("test.png", _image_bytes("blue"), "image/png")},
        data={"question": "What color is this image?", "lang": "en", "max_length": 50},
    )
    assert response.status_code == 200
    result = response.json()
    assert "answer" in result
    assert result["question"] == "What color is this image?"


def test_invalid_image():
    response = client.post("/api/v1/caption/", files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")})
    assert response.status_code == 400
