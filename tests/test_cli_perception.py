import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import json
import time
import os
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
@patch("pico.cli.perception.MonitorWindow")
def test_get_tracks(mock_monitor, mock_vlm_class, mock_detector_class, mock_reader_class, mock_config, capsys):
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

    cli = OnDemandPerceptionCLI(mock_config, shared_reader=mock_reader)
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
@patch("pico.cli.perception.MonitorWindow")
@patch("cv2.resize")
def test_analyze_crop_clamping(mock_resize, mock_monitor, mock_vlm_class, mock_detector_class, mock_reader_class, mock_config, capsys):
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

    cli = OnDemandPerceptionCLI(mock_config, shared_reader=mock_reader)
    try:
        from pico.tracker import TrackedObject
        cli.tracker.tracked_objects[1] = TrackedObject(track_id=1, class_id=15, bbox=(100, 100, 1200, 1200), confidence=0.85)
        
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

@patch("pico.cli.perception.RTSPVideoReader")
@patch("pico.cli.perception.YoloDetector")
@patch("pico.cli.perception.OllamaVisionClient")
@patch("pico.cli.perception.MonitorWindow")
@patch("cv2.imwrite")
def test_analyze_crop_class_filter(mock_write, mock_monitor, mock_vlm_class, mock_detector_class, mock_reader_class, mock_config):
    # Setup
    mock_reader = MagicMock()
    mock_reader_class.return_value = mock_reader
    mock_reader.get_last_frame_time.return_value = time.monotonic()
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_reader.read.return_value = (True, dummy_frame)

    mock_detector = MagicMock()
    mock_detector_class.return_value = mock_detector
    # index 28 = suitcase
    mock_detector.detect.return_value = [
        Detection(class_id=28, confidence=0.95, bbox=(50, 50, 100, 100))
    ]

    mock_vlm = MagicMock()
    mock_vlm_class.return_value = mock_vlm
    mock_vlm.analyze_scene = AsyncMock(return_value="A red suitcase.")
    mock_vlm.close = AsyncMock()

    cli = OnDemandPerceptionCLI(mock_config, shared_reader=mock_reader)
    try:
        # track_id なし、class_filter='suitcase' で曖昧マッチングを検証
        res = cli.analyze_crop_data(class_filter="suitcase", query="What color?")
        assert res["status"] == "success"
        assert res["response"] == "A red suitcase."
        mock_write.assert_called()
    finally:
        cli.close()

@patch("pico.cli.perception.RTSPVideoReader")
@patch("pico.cli.perception.YoloDetector")
@patch("pico.cli.perception.OllamaVisionClient")
@patch("pico.cli.perception.MonitorWindow")
@patch("cv2.imwrite")
def test_get_live_snapshot(mock_write, mock_monitor, mock_vlm_class, mock_detector_class, mock_reader_class, mock_config):
    mock_reader = MagicMock()
    mock_reader_class.return_value = mock_reader
    mock_reader.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    
    mock_vlm = MagicMock()
    mock_vlm_class.return_value = mock_vlm
    mock_vlm.close = AsyncMock()

    cli = OnDemandPerceptionCLI(mock_config, shared_reader=mock_reader)
    try:
        res = cli.get_live_snapshot_data()
        assert res["status"] == "success"
        assert "live_snapshot.jpg" in res["filepath"]
        mock_write.assert_called_once()
    finally:
        cli.close()
