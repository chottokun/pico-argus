import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from pico.ollama_client import OllamaVisionClient

def test_ollama_vision_client_health_check() -> None:
    # httpx.AsyncClient.get をモック
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = OllamaVisionClient(base_url="http://localhost:11434")
        
        # asyncを実行
        is_healthy = asyncio.run(client.health_check())
        assert is_healthy is True
        mock_get.assert_called_once_with("http://localhost:11434/")

def test_ollama_vision_client_analyze_scene() -> None:
    # httpx.AsyncClient.post をモック
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Ollama の chat レスポンス模擬
        mock_response.json.return_value = {
            "message": {
                "content": "部屋の中に人が立っています。"
            }
        }
        mock_post.return_value = mock_response

        client = OllamaVisionClient(
            base_url="http://localhost:11434",
            model="gemma4:e2b"
        )

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        response_text = asyncio.run(client.analyze_scene(dummy_frame, prompt="何が見えますか？"))

        assert response_text == "部屋の中に人が立っています。"
        
        # 送信データの形式検証
        called_args, called_kwargs = mock_post.call_args
        json_data = called_kwargs["json"]
        assert json_data["model"] == "gemma4:e2b"
        assert json_data["messages"][0]["role"] == "user"
        assert json_data["messages"][0]["content"] == "何が見えますか？"
        assert len(json_data["messages"][0]["images"]) == 1
        assert isinstance(json_data["messages"][0]["images"][0], str)  # base64文字列
