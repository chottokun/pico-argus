import os
import json
import logging
from typing import Final
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# デフォルト定数
DEFAULT_OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL: Final[str] = "gemma4:e2b"  # 最も軽量なモデルをテストで優先
DEFAULT_MAX_LIMIT_X: Final[float] = 1.0
DEFAULT_MAX_LIMIT_Y: Final[float] = 0.5

RTSP_SCHEME: Final[str] = "rtsp"
RTSP_PORT: Final[int] = 554
DEFAULT_STREAM: Final[str] = "stream1"


def build_rtsp_url(user: str, password: str, host: str, stream: str = DEFAULT_STREAM) -> str:
    """TapoカメラのRTSPストリーム接続用URLを組み立てる。"""
    return f"{RTSP_SCHEME}://{user}:{password}@{host}:{RTSP_PORT}/{stream}"


class AppConfig:
    """アプリケーション全体の設定およびカメラの可動限界情報を一元管理するクラス。"""

    def __init__(self, env_path: str | None = None, config_path: str = "tapo_config.json") -> None:
        # .envファイルのロード
        load_dotenv(dotenv_path=env_path)

        # 必須環境変数の検証と取得
        self.tapo_user: str = self._require_env("TAPO_USER")
        self.tapo_pass: str = self._require_env("TAPO_PASS")
        self.tapo_ip: str = self._require_env("TAPO_IP")

        # Ollama 設定（デフォルト値と .env による上書き）
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.cognition_target_rule: str = os.getenv("TAPO_COGNITION_TARGET_RULE", "a person wearing a hat")

        # 起動時アライメント設定 (デフォルト True、"false", "no", "0" で無効化)
        align_env = os.getenv("TAPO_ALIGN_TO_HOME", "true").lower()
        self.align_to_home: bool = align_env not in ("false", "no", "0")

        # キャリブレーション限界値と反転設定の読み込み
        self.max_limit_x: float = DEFAULT_MAX_LIMIT_X
        self.max_limit_y: float = DEFAULT_MAX_LIMIT_Y
        self.invert_pan: bool = False
        self.invert_tilt: bool = False
        self._load_calibration_config(config_path)

    def _require_env(self, name: str) -> str:
        """必須の環境変数を取得し、存在しない場合は例外を発生させる。"""
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    def _load_calibration_config(self, config_path: str) -> None:
        """キャリブレーション結果の JSON ファイルを読み込む。"""
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.max_limit_x = data.get("MAX_LIMIT_X", DEFAULT_MAX_LIMIT_X)
                    self.max_limit_y = data.get("MAX_LIMIT_Y", DEFAULT_MAX_LIMIT_Y)
                    self.invert_pan = data.get("INVERT_PAN", False)
                    self.invert_tilt = data.get("INVERT_TILT", False)
                    logger.info(
                        f"Loaded calibration config from {config_path} - "
                        f"Limits X: ±{self.max_limit_x}, Y: ±{self.max_limit_y} | "
                        f"Invert Pan: {self.invert_pan}, Tilt: {self.invert_tilt}"
                    )
            except Exception as e:
                logger.error(f"Failed to load calibration config: {e}. Using default values.")
        else:
            logger.warning(f"Calibration config {config_path} not found. Using default values.")
