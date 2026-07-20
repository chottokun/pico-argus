import pytest
from pico.config import build_rtsp_url, AppConfig

def test_build_rtsp_url_uses_stream1_by_default() -> None:
    assert build_rtsp_url("user", "pass", "192.168.0.10") == "rtsp://user:pass@192.168.0.10:554/stream1"


def test_build_rtsp_url_supports_custom_stream() -> None:
    assert build_rtsp_url("user", "pass", "192.168.0.10", "stream2") == "rtsp://user:pass@192.168.0.10:554/stream2"


def test_require_env_raises_for_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # 必要な環境変数が欠けている場合に ValueError を投げることを確認
    monkeypatch.delenv("TAPO_USER", raising=False)
    with pytest.raises(ValueError, match="TAPO_USER"):
        AppConfig(env_path="")

