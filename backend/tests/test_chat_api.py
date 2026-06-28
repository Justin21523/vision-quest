from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_chat_basic():
    response = client.post("/api/v1/chat/", json={"messages": [{"role": "user", "content": "你好"}], "model": "demo-qwen"})
    assert response.status_code == 200
    data = response.json()
    assert data["message"]["role"] == "assistant"
    assert data["message"]["content"]


def test_chat_with_history():
    response = client.post(
        "/api/v1/chat/",
        json={
            "messages": [
                {"role": "user", "content": "我叫小明"},
                {"role": "assistant", "content": "你好小明！"},
                {"role": "user", "content": "你還記得我的名字嗎？"},
            ],
            "model": "demo-qwen",
        },
    )
    assert response.status_code == 200
    assert "名字" in response.json()["message"]["content"] or "記得" in response.json()["message"]["content"]


def test_get_models():
    response = client.get("/api/v1/chat/models")
    assert response.status_code == 200
    assert response.json()["models"]
