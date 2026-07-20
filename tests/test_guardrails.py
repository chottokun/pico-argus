import time
import numpy as np
from pico.guardrails import GuardRails, FrameStatus

def test_guardrails_frame_timeout() -> None:
    guard = GuardRails(timeout_limit=3.0)
    
    # 正常ケース (0.5秒の遅延)
    status_ok = guard.check_frame_health(last_frame_time=time.monotonic() - 0.5)
    assert status_ok == FrameStatus.OK
    
    # タイムアウトケース (4.0秒の遅延)
    status_timeout = guard.check_frame_health(last_frame_time=time.monotonic() - 4.0)
    assert status_timeout == FrameStatus.TIMEOUT

def test_guardrails_bbox_sanity() -> None:
    guard = GuardRails()
    
    # 正常なBBox (画面640x480に対して 100x100 ＝ 面積比約 3%)
    assert guard.is_bbox_sane(bbox=(100, 100, 100, 100), frame_shape=(480, 640)) is True
    
    # 異常なBBox (画面640x480に対して 600x450 ＝ 面積比約 87% > 閾値75%)
    assert guard.is_bbox_sane(bbox=(20, 15, 600, 450), frame_shape=(480, 640)) is False

def test_guardrails_night_mode_detection() -> None:
    guard = GuardRails(night_std_threshold=10.0)
    
    # カラフルなフレーム (彩度の標準偏差が大きい)
    color_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    assert guard.check_night_mode(color_frame) is False
    
    # モノクローム（夜間赤外線）フレーム (HSVのS成分がほぼ0)
    gray_frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)  # 全て黒＝モノクロ
    assert guard.check_night_mode(gray_frame_bgr) is True
