from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == settings.APP_NAME
    assert data["status"] == "running"


def test_openapi_json():
    response = client.get(f"{settings.API_PREFIX}/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == settings.APP_NAME


def test_docs_endpoint():
    response = client.get(f"{settings.API_PREFIX}/docs")
    assert response.status_code == 200


def test_health_readiness_liveness():
    assert client.get(f"{settings.API_PREFIX}/health/readiness").json()["ready"] is True
    assert client.get(f"{settings.API_PREFIX}/health/liveness").json()["alive"] is True
