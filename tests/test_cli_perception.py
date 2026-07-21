import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import json
import time
from pico.config import AppConfig
from pico.cli.perception import OnDemandPerceptionCLI
from pico.detector import Detection

@pytest.fixture
def mock_config():
    config = MagicMock(spec=AppConfig)
    config.tapo_ip = "192.168.0.100"
    config.tapo_user = "admin"
    config.tapo_pass = "password"
    config.ollama_base_url = "http://localhost:11434"
    config.ollama_model = "gemma4:e2b"
    return config

@patch("pico.cli.perception.RTSPVideoReader")
@patch("pico.cli.perception.YoloDetector")
@patch("pico.cli.perception.OllamaVisionClient")
def test_get_tracks(mock_vlm_class, mock_detector_class, mock_reader_class, mock_config, capsys):
    # Setup
    mock_reader = MagicMock()
    mock_reader_class.return_value = mock_reader
    mock_reader.get_last_frame_time.return_value = time.monotonic()
    
    # 640x480のフレームを返す
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_reader.read.return_value = (True, dummy_frame)

    mock_detector = MagicMock()
    mock_detector_class.return_value = mock_detector
    mock_detector.detect.return_value = [
        Detection(class_id=0, confidence=0.9, bbox=(10, 20, 100, 200))
    ]

    mock_vlm = MagicMock()
    mock_vlm_class.return_value = mock_vlm
    mock_vlm.close = AsyncMock()

    cli = OnDemandPerceptionCLI(mock_config)
    try:
        cli.get_tracks()
        captured = capsys.readouterr()
        res = json.loads(captured.out)
        
        assert "tracks" in res
        assert len(res["tracks"]) == 1
        assert res["tracks"][0]["class"] == "person"
        assert res["tracks"][0]["track_id"] == 1
        assert res["tracks"][0]["bbox"] == [10, 20, 100, 200]
    finally:
        cli.close()

@patch("pico.cli.perception.RTSPVideoReader")
@patch("pico.cli.perception.YoloDetector")
@patch("pico.cli.perception.OllamaVisionClient")
@patch("cv2.resize")
def test_analyze_crop_clamping(mock_resize, mock_vlm_class, mock_detector_class, mock_reader_class, mock_config, capsys):
    # Setup
    mock_reader = MagicMock()
    mock_reader_class.return_value = mock_reader
    mock_reader.get_last_frame_time.return_value = time.monotonic()
    
    # 大きなダミー画像 1500x1500
    dummy_frame = np.zeros((1500, 1500, 3), dtype=np.uint8)
    mock_reader.read.return_value = (True, dummy_frame)

    mock_detector = MagicMock()
    mock_detector_class.return_value = mock_detector
    # 大きなオブジェクトの検出 (1200x1200)
    mock_detector.detect.return_value = [
        Detection(class_id=15, confidence=0.85, bbox=(100, 100, 1200, 1200))
    ]

    mock_vlm = MagicMock()
    mock_vlm_class.return_value = mock_vlm
    mock_vlm.analyze_scene = AsyncMock(return_value="This is a cat.")
    mock_vlm.close = AsyncMock()

    # cv2.resize が縮小後の画像を返すようにする
    mock_resize.return_value = np.zeros((1024, 1024, 3), dtype=np.uint8)

    cli = OnDemandPerceptionCLI(mock_config)
    try:
        cli.analyze_crop(track_id=1, query="What is this?")
        captured = capsys.readouterr()
        res = json.loads(captured.out)
        
        assert res["status"] == "success"
        assert res["response"] == "This is a cat."
        
        mock_resize.assert_called_once()
        args, kwargs = mock_resize.call_args
        assert args[1] == (1024, 1024)

    finally:
        cli.close()
