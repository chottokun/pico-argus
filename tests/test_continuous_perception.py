import time
import pytest
from unittest.mock import MagicMock, patch
from pico.config import AppConfig
from pico.cli.perception import ContinuousPerceptionLoop

@pytest.fixture
def mock_config():
    config = MagicMock(spec=AppConfig)
    config.tapo_ip = "192.168.0.100"
    config.tapo_user = "admin"
    config.tapo_pass = "password"
    config.ollama_base_url = "http://localhost:11434"
    config.ollama_model = "gemma4:e2b"
    return config

@patch("pico.cli.perception.YoloDetector")
@patch("pico.cli.perception.SimpleIoUTracker")
def test_continuous_perception_loop_runs_and_caches_tracks(mock_tracker_class, mock_detector_class):
    mock_reader = MagicMock()
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_reader.read.return_value = (True, dummy_frame)
    mock_reader.get_last_frame_time.return_value = time.monotonic()

    mock_det_instance = MagicMock()
    mock_detector_class.return_value = mock_det_instance
    mock_det_instance.detect.return_value = []

    mock_track_instance = MagicMock()
    mock_tracker_class.return_value = mock_track_instance
    
    dummy_obj = MagicMock()
    dummy_obj.track_id = 1
    dummy_obj.class_id = 0
    dummy_obj.confidence = 0.85
    dummy_obj.bbox = [10, 10, 100, 100]
    mock_track_instance.update.return_value = [dummy_obj]
    mock_track_instance.tracked_objects = {1: dummy_obj}

    loop_engine = ContinuousPerceptionLoop(reader=mock_reader)
    loop_engine.start()
    
    time.sleep(0.3)  # ループが数回回るのを待つ
    
    tracks = loop_engine.get_cached_tracks()
    assert len(tracks) >= 1
    assert tracks[0]["track_id"] == 1
    assert tracks[0]["class"] == "person"
    
    status = loop_engine.get_status()
    assert status["engine_status"] == "RUNNING"
    assert status["active_track_count"] == 1

    loop_engine.stop()
    assert loop_engine.running is False
