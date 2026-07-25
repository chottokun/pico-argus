import os
import json
from unittest.mock import patch, mock_open
from pico.config import AppConfig

def test_app_config_loads_env_and_defaults(monkeypatch) -> None:
    # 環境変数をモック
    monkeypatch.setenv("TAPO_USER", "test_user")
    monkeypatch.setenv("TAPO_PASS", "test_pass")
    monkeypatch.setenv("TAPO_IP", "192.168.1.100")
    # OLLAMA_MODEL は未指定でデフォルト値を確認
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    # tapo_config.json のモック用データ
    mock_json_data = json.dumps({
        "MAX_LIMIT_X": 1.15,
        "MAX_LIMIT_Y": 0.90,
        "INVERT_PAN": True,
        "INVERT_TILT": True,
        "CALIBRATED_AT": "2026-07-20 12:00:00"
    })

    # dotenv の探索に影響を与えないよう、tapo_config.json に対してのみ True を返すモック
    original_exists = os.path.exists
    def mock_exists(path):
        if "tapo_config.json" in str(path):
            return True
        return original_exists(path)

    with patch("builtins.open", mock_open(read_data=mock_json_data)):
        with patch("os.path.exists", side_effect=mock_exists):
            config = AppConfig()
            assert config.tapo_user == "test_user"
            assert config.tapo_pass == "test_pass"
            assert config.tapo_ip == "192.168.1.100"
            assert config.ollama_base_url == "http://localhost:11434"  # デフォルト値
            assert config.ollama_model == "gemma4:e2b"  # デフォルトで最軽量
            assert config.ollama_max_rpm == 12  # デフォルト推奨値
            assert config.cognition_target_rule == "a person wearing a hat"  # デフォルト値
            assert config.max_limit_x == 1.15
            assert config.max_limit_y == 0.90
            assert config.invert_pan is True
            assert config.invert_tilt is True

def test_app_config_supports_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TAPO_USER", "test_user")
    monkeypatch.setenv("TAPO_PASS", "test_pass")
    monkeypatch.setenv("TAPO_IP", "192.168.1.100")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:latest")  # 上書き
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.50:11434")  # 上書き
    monkeypatch.setenv("OLLAMA_MAX_RPM", "20")  # 上書き
    monkeypatch.setenv("TAPO_COGNITION_TARGET_RULE", "a person with red clothing")  # 上書き

    # dotenvの探索に影響を与えないよう、tapo_config.jsonに対してのみFalseを返すモック
    original_exists = os.path.exists
    def mock_exists(path):
        if "tapo_config.json" in str(path):
            return False
        return original_exists(path)

    with patch("os.path.exists", side_effect=mock_exists):
        config = AppConfig()
        assert config.ollama_model == "gemma4:latest"
        assert config.ollama_base_url == "http://192.168.1.50:11434"
        assert config.ollama_max_rpm == 20
        assert config.cognition_target_rule == "a person with red clothing"
        # JSONがない場合のデフォルト値
        assert config.max_limit_x == 1.0
        assert config.max_limit_y == 0.5

def test_app_config_show_monitor_option(monkeypatch) -> None:
    monkeypatch.setenv("TAPO_USER", "test_user")
    monkeypatch.setenv("TAPO_PASS", "test_pass")
    monkeypatch.setenv("TAPO_IP", "192.168.1.100")

    with patch("pico.config.load_dotenv"):
        # 未指定の場合のデフォルト (False)
        monkeypatch.delenv("SHOW_MONITOR", raising=False)
        monkeypatch.delenv("TAPO_SHOW_MONITOR", raising=False)
        config1 = AppConfig()
        assert config1.show_monitor is False

        # SHOW_MONITOR=true
        monkeypatch.setenv("SHOW_MONITOR", "true")
        config2 = AppConfig()
        assert config2.show_monitor is True

        # SHOW_MONITOR=1
        monkeypatch.setenv("SHOW_MONITOR", "1")
        config3 = AppConfig()
        assert config3.show_monitor is True


def test_build_rtsp_url_uses_stream1_by_default() -> None:
    from pico.config import build_rtsp_url
    assert build_rtsp_url("user", "pass", "192.168.0.10") == "rtsp://user:pass@192.168.0.10:554/stream1"


def test_build_rtsp_url_supports_custom_stream() -> None:
    from pico.config import build_rtsp_url
    assert build_rtsp_url("user", "pass", "192.168.0.10", "stream2") == "rtsp://user:pass@192.168.0.10:554/stream2"

