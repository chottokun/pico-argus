import time
import pytest
from unittest.mock import MagicMock, patch
from pico.onvif_client import PTZController


@pytest.fixture
def mock_onvif_camera():
    with patch("pico.onvif_client.ONVIFCamera") as mock_cam_class:
        mock_cam = MagicMock()
        mock_ptz = MagicMock()
        mock_media = MagicMock()
        
        # GetProfiles() のモックデータ
        mock_profile = MagicMock()
        mock_profile.token = "profile_token_123"
        mock_media.GetProfiles.return_value = [mock_profile]
        
        mock_cam.create_ptz_service.return_value = mock_ptz
        mock_cam.create_media_service.return_value = mock_media
        mock_cam_class.return_value = mock_cam
        
        yield mock_cam, mock_ptz, mock_media

def test_ptz_controller_initialization(mock_onvif_camera) -> None:
    controller = PTZController(ip="192.168.1.100", user="user", password="pwd")
    assert controller.profile_token == "profile_token_123"
    assert controller.current_x == 0.0
    assert controller.current_y == 0.0

def test_ptz_controller_relative_move(mock_onvif_camera) -> None:
    mock_cam, mock_ptz, mock_media = mock_onvif_camera
    
    # RelativeMove の引数作成のためのモック
    mock_request = MagicMock()
    mock_ptz.create_type.return_value = mock_request
    
    controller = PTZController(ip="192.168.1.100", user="user", password="pwd")
    
    # モーション完了を待つためにジョブを実行させる
    controller.relative_move(0.1, -0.05)
    
    # 非同期スレッドが実行するのを少し待つ
    import time
    time.sleep(0.2)
    
    # 呼び出し検証
    mock_ptz.create_type.assert_called_with("RelativeMove")
    mock_ptz.RelativeMove.assert_called_once_with(mock_request)
    assert mock_request.ProfileToken == "profile_token_123"
    assert mock_request.Translation == {'PanTilt': {'x': 0.1, 'y': -0.05}}

def test_ptz_controller_safe_move(mock_onvif_camera) -> None:
    mock_cam, mock_ptz, mock_media = mock_onvif_camera
    mock_request = MagicMock()
    mock_ptz.create_type.return_value = mock_request

    controller = PTZController(
        ip="192.168.1.100", user="user", password="pwd",
        max_limit_x=1.0, max_limit_y=0.5
    )

    # 1回目：限界内の移動
    controller.safe_move(0.5, 0.2)
    time.sleep(0.2)
    assert controller.current_x == 0.5
    assert controller.current_y == 0.2

    # 2回目：限界を超える移動（クランプされるはず）
    # x: 0.5 + 0.6 = 1.1 (Limit 1.0) -> 移動量は 1.0 - 0.5 = 0.5 にクランプ
    # y: 0.2 + 0.4 = 0.6 (Limit 0.5) -> 移動量は 0.5 - 0.2 = 0.3 にクランプ
    controller.safe_move(0.6, 0.4)
    time.sleep(0.2)
    
    # クランプされた後の実際の位置
    assert controller.current_x == 1.0
    assert controller.current_y == 0.5
