# frontend/gradio_app/utils.py
import requests
import json
from typing import Any, Dict


def handle_api_error(response: requests.Response) -> str:
    """Handle API error responses"""
    try:
        error_data = response.json()
        if "detail" in error_data:
            return str(error_data["detail"])
        return f"狀態碼 {response.status_code}"
    except:
        return f"HTTP {response.status_code}"


def api_call(endpoint: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
    """Generic API call helper"""
    base_url = "http://localhost:8000/api/v1"
    url = f"{base_url}/{endpoint.lstrip('/')}"

    try:
        if method.upper() == "GET":
            response = requests.get(url, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"API 調用失敗: {str(e)}")


def encode_image(image) -> str:
    """Encode PIL image to base64"""
    import io
    import base64

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")
