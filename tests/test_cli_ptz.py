import pytest
from unittest.mock import MagicMock, patch
from pico.config import AppConfig
from pico.cli.ptz import PTZActuator

@pytest.fixture
def mock_config():
    config = MagicMock(spec=AppConfig)
    config.tapo_ip = "192.168.0.100"
    config.tapo_user = "admin"
    config.tapo_pass = "password"
    config.max_limit_x = 1.0
    config.max_limit_y = 0.5
    config.invert_pan = False
    config.invert_tilt = False
    config.align_to_home = False
    return config

@patch("pico.cli.ptz.PTZController")
def test_send_pulse_move(mock_ptz_class, mock_config):
    # Setup
    mock_ptz_instance = MagicMock()
    mock_ptz_class.return_value = mock_ptz_instance
    mock_ptz_instance.safe_move.return_value = (0.1, -0.05)

    actuator = PTZActuator(mock_config)
    
    # Run
    actuator.send_pulse_move(0.1, -0.05)

    # Assert
    mock_ptz_class.assert_called_once_with(
        ip="192.168.0.100",
        user="admin",
        password="password",
        max_limit_x=1.0,
        max_limit_y=0.5,
        align_to_home=False,
        video_reader=None,
        invert_pan=False,
        invert_tilt=False,
        step_size_x=0.15,
        step_size_y=0.10,
        total_steps_x=15,
        total_steps_y=20,
        return_steps_x=None,
        return_steps_y=None,
        hunt_steps_x=25,
        hunt_steps_y=25
    )
    mock_ptz_instance.safe_move.assert_called_once_with(0.1, -0.05)

@patch("pico.cli.ptz.PTZController")
def test_emergency_stop(mock_ptz_class, mock_config):
    # Setup
    mock_ptz_instance = MagicMock()
    mock_ptz_class.return_value = mock_ptz_instance

    actuator = PTZActuator(mock_config)
    
    # Run
    actuator.emergency_stop()

    # Assert
    mock_ptz_instance.relative_move.assert_called_once_with(0.0, 0.0)

@patch("pico.cli.ptz.PTZController")
@patch("pico.cli.ptz.RTSPVideoReader")
@patch("pico.cli.ptz.YoloDetector")
@patch("pico.cli.ptz.SimpleIoUTracker")
def test_lockon_loop_exits_on_interrupt(mock_tracker_class, mock_detector_class, mock_reader_class, mock_ptz_class, mock_config):
    # Setup
    mock_ptz_instance = MagicMock()
    mock_ptz_class.return_value = mock_ptz_instance
    
    mock_reader_instance = MagicMock()
    mock_reader_class.return_value = mock_reader_instance
    import time
    mock_reader_instance.get_last_frame_time.return_value = time.monotonic()
    mock_reader_instance.read.side_effect = KeyboardInterrupt() # Force exit loop immediately

    actuator = PTZActuator(mock_config)

    # Run
    actuator.lockon(mock_reader_instance, 42)

    # Assert
    mock_ptz_instance.shutdown.assert_called_once()

@patch("pico.cli.ptz.PTZController")
@patch("pico.cli.ptz.RTSPVideoReader")
@patch("pico.cli.ptz.YoloDetector")
@patch("pico.cli.ptz.SimpleIoUTracker")
def test_lockon_loop_exits_on_class_filter(mock_tracker_class, mock_detector_class, mock_reader_class, mock_ptz_class, mock_config):
    # Setup
    mock_ptz_instance = MagicMock()
    mock_ptz_class.return_value = mock_ptz_instance
    
    mock_reader_instance = MagicMock()
    mock_reader_class.return_value = mock_reader_instance
    import time
    mock_reader_instance.get_last_frame_time.return_value = time.monotonic()
    mock_reader_instance.read.side_effect = KeyboardInterrupt() # ループを即時脱出させる

    actuator = PTZActuator(mock_config)

    # Run
    actuator.lockon(mock_reader_instance, class_filter="person")

    # Assert
    assert actuator.lockon_class_name is None
    mock_ptz_instance.shutdown.assert_called_once()
