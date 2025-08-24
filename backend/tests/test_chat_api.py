import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_chat_basic():
    """測試基本對話功能"""
    response = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "你好"}], "model": "qwen"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"]["role"] == "assistant"
    assert len(data["message"]["content"]) > 0


def test_chat_with_history():
    """測試多輪對話"""
    response = client.post(
        "/api/v1/chat",
        json={
            "messages": [
                {"role": "user", "content": "我叫小明"},
                {"role": "assistant", "content": "你好小明！"},
                {"role": "user", "content": "你還記得我的名字嗎？"},
            ],
            "model": "qwen",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "小明" in data["message"]["content"]


def test_get_models():
    """測試模型列表端點"""
    response = client.get("/api/v1/chat/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0
