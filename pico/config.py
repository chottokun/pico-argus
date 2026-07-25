import os
import json
import logging
from typing import Final
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# デフォルト定数
DEFAULT_OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL: Final[str] = "gemma4:e2b"  # 最も軽量なモデルをテストで優先
DEFAULT_OLLAMA_MAX_RPM: Final[int] = 12          # ベンチマーク結果に基づく推奨値
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

    def __init__(self, env_path: str | None = None, config_path: str = "camera_config.json") -> None:
        # .envファイルのロード
        load_dotenv(dotenv_path=env_path)

        # 必須環境変数の検証と取得
        self.tapo_user: str = self._require_env("TAPO_USER")
        self.tapo_pass: str = self._require_env("TAPO_PASS")
        self.tapo_ip: str = self._require_env("TAPO_IP")

        # Ollama 設定（デフォルト値と .env による上書き）
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.ollama_max_rpm: int = int(os.getenv("OLLAMA_MAX_RPM", str(DEFAULT_OLLAMA_MAX_RPM)))
        self.cognition_target_rule: str = os.getenv("TAPO_COGNITION_TARGET_RULE", "a person wearing a hat")

        # 起動時アライメント設定 (デフォルト True、"false", "no", "0" で無効化)
        align_env = os.getenv("TAPO_ALIGN_TO_HOME", "true").lower()
        self.align_to_home: bool = align_env not in ("false", "no", "0")

        # モニター表示設定 (デフォルト False、SHOW_MONITOR または TAPO_SHOW_MONITOR で指定)
        show_env = os.getenv("SHOW_MONITOR", os.getenv("TAPO_SHOW_MONITOR", "false")).lower()
        self.show_monitor: bool = show_env in ("true", "1", "yes")

        # キャリブレーション限界値とアライメントステップ設定の読み込み
        self.max_limit_x: float = DEFAULT_MAX_LIMIT_X
        self.max_limit_y: float = DEFAULT_MAX_LIMIT_Y
        self.invert_pan: bool = False
        self.invert_tilt: bool = False
        self.step_size_x: float = 0.15
        self.step_size_y: float = 0.10
        self.return_steps_x: int = 7
        self.return_steps_y: int = 9
        self.hunt_steps_x: int = 25
        self.hunt_steps_y: int = 25
        self._load_calibration_config(config_path)

    def _require_env(self, name: str) -> str:
        """必須の環境変数を取得し、存在しない場合は例外を発生させる。"""
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    def _load_calibration_config(self, config_path: str) -> None:
        """キャリブレーション結果の JSON ファイル (camera_config.json / tapo_config.json) を読み込む。"""
        target_path = config_path
        if not os.path.exists(target_path) and os.path.exists("tapo_config.json"):
            target_path = "tapo_config.json"

        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.max_limit_x = data.get("MAX_LIMIT_X", DEFAULT_MAX_LIMIT_X)
                    self.max_limit_y = data.get("MAX_LIMIT_Y", DEFAULT_MAX_LIMIT_Y)
                    self.invert_pan = data.get("INVERT_PAN", False)
                    self.invert_tilt = data.get("INVERT_TILT", False)
                    self.step_size_x = data.get("STEP_SIZE_X", 0.15)
                    self.step_size_y = data.get("STEP_SIZE_Y", 0.10)
                    self.return_steps_x = data.get("RETURN_STEPS_X", int(round(self.max_limit_x / self.step_size_x)))
                    self.return_steps_y = data.get("RETURN_STEPS_Y", int(round(self.max_limit_y / self.step_size_y)))
                    self.hunt_steps_x = data.get("HUNT_STEPS_X", 25)
                    self.hunt_steps_y = data.get("HUNT_STEPS_Y", 25)
                    logger.info(
                        f"Loaded calibration config from {target_path} - "
                        f"Limits X: ±{self.max_limit_x}, Y: ±{self.max_limit_y} | "
                        f"Return Steps: X={self.return_steps_x}, Y={self.return_steps_y} | "
                        f"Invert Pan: {self.invert_pan}, Tilt: {self.invert_tilt}"
                    )
            except Exception as e:
                logger.error(f"Failed to load calibration config: {e}. Using default values.")
        else:
            logger.warning(f"Calibration config {target_path} not found. Using default values.")
