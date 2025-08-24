# frontend/pyqt_app/utils.py
import requests
from typing import Optional, Dict, Any


def test_api_connection(base_url: str = "http://localhost:8000") -> bool:
    """Test if API is reachable"""
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
