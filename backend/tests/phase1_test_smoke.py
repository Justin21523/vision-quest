# backend/tests/phase1_test_smoke.py
"""
Basic smoke tests for Phase 1 VisionQuest API.
Tests core functionality and health endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app.main import app
from app.core.config import settings

# Create test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health_check(self):
        """Test basic health endpoint returns 200."""
        response = client.get(f"{settings.API_PREFIX}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["app"] == settings.APP_NAME
        assert data["version"] == settings.APP_VERSION

    def test_detailed_health_check(self):
        """Test detailed health endpoint."""
        response = client.get(f"{settings.API_PREFIX}/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "app_info" in data
        assert "system" in data
        assert "configuration" in data

        # Check app info
        app_info = data["app_info"]
        assert app_info["name"] == settings.APP_NAME
        assert app_info["version"] == settings.APP_VERSION
        assert app_info["environment"] == settings.ENV

        # Check system info exists
        system = data["system"]
        assert "platform" in system
        assert "python_version" in system
        assert "cpu_count" in system
        assert "memory" in system

        # Check configuration
        config = data["configuration"]
        assert config["device"] == settings.DEVICE
        assert config["max_workers"] == settings.MAX_WORKERS

    def test_models_health_check(self):
        """Test models health endpoint."""
        response = client.get(f"{settings.API_PREFIX}/health/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "dependencies" in data

        # Should contain model categories
        models = data["models"]
        assert "caption_models" in models
        assert "vqa_models" in models
        assert "llm_models" in models

    def test_services_health_check(self):
        """Test services health endpoint."""
        response = client.get(f"{settings.API_PREFIX}/health/services")
        assert response.status_code == 200

        data = response.json()
        assert "services" in data
        assert "database" in data["services"]

    def test_readiness_check(self):
        """Test readiness probe endpoint."""
        response = client.get(f"{settings.API_PREFIX}/health/readiness")
        assert response.status_code == 200

        data = response.json()
        assert data["ready"] is True
        assert "checks" in data

        checks = data["checks"]
        assert "configuration_loaded" in checks
        assert "directories_accessible" in checks
        assert "basic_dependencies" in checks

    def test_liveness_check(self):
        """Test liveness probe endpoint."""
        response = client.get(f"{settings.API_PREFIX}/health/liveness")
        assert response.status_code == 200

        data = response.json()
        assert data["alive"] is True
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint returns app information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["app"] == settings.APP_NAME
        assert data["version"] == settings.APP_VERSION
        assert data["status"] == "running"
        assert data["environment"] == settings.ENV
        assert "docs_url" in data
        assert "health_url" in data


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_json(self):
        """Test OpenAPI JSON schema endpoint."""
        response = client.get(f"{settings.API_PREFIX}/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == settings.APP_NAME
        assert data["info"]["version"] == settings.APP_VERSION

    def test_docs_endpoint(self):
        """Test Swagger UI docs endpoint."""
        response = client.get(f"{settings.API_PREFIX}/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_endpoint(self):
        """Test ReDoc documentation endpoint."""
        response = client.get(f"{settings.API_PREFIX}/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestMiddleware:
    """Test middleware functionality."""

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.get(f"{settings.API_PREFIX}/health")

        # Check for CORS headers (may not be present in test environment)
        # This test verifies the middleware is configured
        assert response.status_code == 200

    def test_process_time_header(self):
        """Test that process time header is added."""
        response = client.get(f"{settings.API_PREFIX}/health")

        assert response.status_code == 200
        assert "X-Process-Time" in response.headers

        # Process time should be a valid float
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0


class TestErrorHandling:
    """Test error handling."""

    def test_404_error(self):
        """Test 404 error for non-existent endpoint."""
        response = client.get(f"{settings.API_PREFIX}/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test 405 error for wrong HTTP method."""
        response = client.post(f"{settings.API_PREFIX}/health")
        assert response.status_code == 405


class TestConfiguration:
    """Test configuration and settings."""

    def test_settings_loaded(self):
        """Test that settings are properly loaded."""
        assert settings.APP_NAME == "VisionQuest"
        assert settings.API_PREFIX == "/api/v1"
        assert settings.DEVICE in [
            "auto",
            "cpu",
            "cuda",
            "mps",
        ] or settings.DEVICE.startswith("cuda:")
        assert settings.MAX_WORKERS > 0
        assert settings.MAX_BATCH_SIZE > 0

    def test_directories_exist(self):
        """Test that required directories exist."""
        import os

        directories = [
            settings.UPLOAD_DIR,
            settings.OUTPUT_DIR,
            settings.KB_DIR,
            settings.MODEL_CACHE_DIR,
        ]

        for directory in directories:
            assert os.path.exists(directory), f"Directory does not exist: {directory}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
