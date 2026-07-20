import base64
import cv2
import httpx
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class OllamaVisionClient:
    """WSL2 などで動作する Ollama の REST API (VLM) を利用するためのクライアントクラス。"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma4:e2b") -> None:
        self.base_url: str = base_url
        self.model: str = model
        self.client: httpx.AsyncClient = httpx.AsyncClient(timeout=120.0)

    async def health_check(self) -> bool:
        """Ollama サーバーが起動し、応答するかヘルスチェックを実行する。"""
        try:
            # Ollama はルートパスへの GET リクエストに対して "Ollama is running" を返す
            response = await self.client.get(f"{self.base_url}/")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed at {self.base_url}: {e}")
            return False

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """OpenCV の画像を JPEG 圧縮し、base64 文字列にエンコードする（データプレフィックスは除外）。"""
        # JPEG品質75に圧縮して転送サイズを最適化
        success, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not success:
            raise ValueError("Failed to encode frame to JPEG for VLM analysis.")
        
        # base64エンコード
        base64_bytes = base64.b64encode(encoded_img)
        return base64_bytes.decode('utf-8')

    async def analyze_scene(self, frame: np.ndarray, prompt: str) -> Optional[str]:
        """指定されたフレーム画像を VLM に送信し、指示プロンプトに基づいた解析結果（テキスト）を取得する。"""
        try:
            base64_str = self._frame_to_base64(frame)
        except Exception as e:
            logger.error(f"Image preprocessing failed for VLM input: {e}")
            return None

        # /api/chat エンドポイント用ペイロードの構築
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_str]
                }
            ],
            "stream": False  # ストリーム応答を無効化し、一括で結果を取得する
        }

        try:
            url = f"{self.base_url}/api/chat"
            response = await self.client.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content")
                return content
            else:
                logger.error(f"Ollama API returned error status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Failed to call Ollama API at {self.base_url}: {e} (Type: {type(e).__name__})", exc_info=True)
            return None

    async def close(self) -> None:
        """HTTP クライアントセッションを閉じる。"""
        await self.client.aclose()
        logger.info("OllamaVisionClient session closed.")
