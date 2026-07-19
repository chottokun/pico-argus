import pytest

from camera_config import build_rtsp_url, require_env


def test_build_rtsp_url_uses_stream1_by_default() -> None:
    assert build_rtsp_url("user", "pass", "192.168.0.10") == "rtsp://user:pass@192.168.0.10:554/stream1"


def test_build_rtsp_url_supports_custom_stream() -> None:
    assert build_rtsp_url("user", "pass", "192.168.0.10", "stream2") == "rtsp://user:pass@192.168.0.10:554/stream2"


def test_require_env_returns_existing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_TAPO_USER", "demo")
    assert require_env("TEST_TAPO_USER") == "demo"


def test_require_env_raises_for_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_TAPO_USER", raising=False)
    with pytest.raises(ValueError, match="TEST_TAPO_USER"):
        require_env("TEST_TAPO_USER")
