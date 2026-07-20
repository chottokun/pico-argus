import time
from unittest.mock import MagicMock, patch
import numpy as np
from pico.video_reader import RTSPVideoReader

@patch("cv2.VideoCapture")
def test_rtsp_video_reader_reading(mock_video_capture) -> None:
    # cv2.VideoCapture のモック作成
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # 1フレーム目は正常に取得、2フレーム目以降はNoneを返す模擬
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, dummy_frame),
        (True, dummy_frame),
        (False, None)
    ]
    mock_video_capture.return_value = mock_cap

    # 起動
    reader = RTSPVideoReader("rtsp://dummy_url")
    time.sleep(0.1)  # スレッドが動くのを少し待つ

    ret, frame = reader.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)

    # 停止確認
    reader.release()
    assert reader.running is False
    mock_cap.release.assert_called_once()
