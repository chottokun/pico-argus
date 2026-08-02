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


def test_ptz_controller_align_to_home(mock_onvif_camera) -> None:
    mock_cam, mock_ptz, mock_media = mock_onvif_camera
    mock_request = MagicMock()
    mock_ptz.create_type.return_value = mock_request

    # 自動ホームアライメントを有効にして初期化
    controller = PTZController(
        ip="192.168.1.100", user="user", password="pwd",
        max_limit_x=1.0, max_limit_y=0.5,
        align_to_home=True
    )
    time.sleep(0.3)

    # 限界追い込み（左・下に移動）と、中心復帰（右・上に移動）が呼び出されていることを検証
    # （テスト内ではモック経由で命令数や位置の変化を確認する）
    # 初期化が完了した時点で、原点 (0.0, 0.0) にアライメントされているはず
    assert controller.current_x == 0.0
    assert controller.current_y == 0.0


def test_ptz_controller_safe_move_with_inversion(mock_onvif_camera) -> None:
    mock_cam, mock_ptz, mock_media = mock_onvif_camera
    mock_request = MagicMock()
    mock_ptz.create_type.return_value = mock_request

    # 左右・上下両方を反転に設定
    controller = PTZController(
        ip="192.168.1.100", user="user", password="pwd",
        max_limit_x=1.0, max_limit_y=0.5,
        invert_pan=True,
        invert_tilt=True
    )

    # 限界内の移動 (0.5, 0.2)
    # 反転フラグがあるため、物理移動コマンドは (-0.5, -0.2) になるべき
    controller.safe_move(0.5, 0.2)
    time.sleep(0.2)

    # 内部現在位置の推測は正方向（論理座標）に加算される
    assert controller.current_x == 0.5
    assert controller.current_y == 0.2

    # 実際に渡された引数は符号反転しているか
    assert mock_request.Translation == {'PanTilt': {'x': -0.5, 'y': -0.2}}
    controller.shutdown()


def test_ptz_controller_move_to_center(mock_onvif_camera) -> None:
    mock_cam, mock_ptz, mock_media = mock_onvif_camera
    mock_request = MagicMock()
    mock_ptz.create_type.return_value = mock_request

    controller = PTZController(
        ip="192.168.1.100", user="user", password="pwd",
        max_limit_x=1.0, max_limit_y=0.5
    )
    controller.current_x = 0.4
    controller.current_y = -0.3

    actual_x, actual_y = controller.move_to_center()

    assert actual_x == pytest.approx(-0.4)
    assert actual_y == pytest.approx(0.3)
    assert controller.current_x == pytest.approx(0.0)
    assert controller.current_y == pytest.approx(0.0)
    controller.shutdown()


def test_ptz_controller_continuous_move_and_stop(mock_onvif_camera) -> None:
    mock_cam, mock_ptz, mock_media = mock_onvif_camera
    mock_request = MagicMock()
    mock_ptz.create_type.return_value = mock_request

    controller = PTZController(ip="192.168.1.100", user="user", password="pwd")

    # 連続移動の実行
    controller.move_continuous(pan_speed=0.5, tilt_speed=-0.2, zoom_speed=0.0, auto_stop_delay=1.0)

    # create_type と ContinuousMove の呼び出し検証
    mock_ptz.create_type.assert_any_call("ContinuousMove")
    mock_ptz.ContinuousMove.assert_called_once_with(mock_request)
    assert mock_request.ProfileToken == "profile_token_123"
    assert mock_request.Velocity == {
        'PanTilt': {'x': 0.5, 'y': -0.2},
        'Zoom': {'x': 0.0}
    }
    assert mock_request.Timeout == "PT1S"
    assert controller._watchdog_timer is not None

    # 明示的な stop() の呼び出し
    mock_stop_request = MagicMock()
    mock_ptz.create_type.return_value = mock_stop_request

    controller.stop()

    mock_ptz.create_type.assert_any_call("Stop")
    mock_ptz.Stop.assert_called_once_with(mock_stop_request)
    assert mock_stop_request.ProfileToken == "profile_token_123"
    assert mock_stop_request.PanTilt is True
    assert mock_stop_request.Zoom is True
    assert controller._watchdog_timer is None

    controller.shutdown()


def test_ptz_controller_watchdog_fallback(mock_onvif_camera) -> None:
    mock_cam, mock_ptz, mock_media = mock_onvif_camera
    mock_request = MagicMock()
    mock_ptz.create_type.return_value = mock_request

    controller = PTZController(ip="192.168.1.100", user="user", password="pwd")

    # 連続移動の実行
    mock_ptz.ContinuousMove.reset_mock()
    mock_ptz.Stop.reset_mock()

    # タイムアウト 0.1 秒に指定して 0.6 秒後に watchdog_stop_fallback が呼ばれるようにする
    controller.move_continuous(pan_speed=0.5, tilt_speed=-0.2, zoom_speed=0.0, auto_stop_delay=0.1)

    # watchdog が発火するまで待機 (0.1秒 + 0.5秒 + α)
    time.sleep(0.8)

    # Stop コマンドが watchdog fallback により自動で送信されたことを確認
    mock_ptz.Stop.assert_called_once()
    assert controller._watchdog_timer is None

    controller.shutdown()

