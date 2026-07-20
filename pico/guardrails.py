import time
import cv2
import numpy as np
import logging
from enum import Enum, auto
from typing import Tuple

logger = logging.getLogger(__name__)

class FrameStatus(Enum):
    OK = auto()
    TIMEOUT = auto()
    CORRUPTED = auto()

class GuardRails:
    """自律エージェントの暴走やネットワーク障害から物理モータおよびシステムを守るためのルールベース安全ガードレールクラス。"""

    def __init__(
        self, timeout_limit: float = 3.0, max_area_ratio: float = 0.75,
        night_std_threshold: float = 12.0
    ) -> None:
        self.timeout_limit: float = timeout_limit
        self.max_area_ratio: float = max_area_ratio
        self.night_std_threshold: float = night_std_threshold

    def check_frame_health(self, last_frame_time: float) -> FrameStatus:
        """RTSP の映像フレーム取得タイムスタンプを検証し、フリーズや通信途絶がないか判定する。"""
        current_time = time.monotonic()
        elapsed = current_time - last_frame_time
        
        if elapsed > self.timeout_limit:
            logger.error(f"RTSP Stream Timeout detected! Elapsed time: {elapsed:.2f}s (Limit: {self.timeout_limit}s)")
            return FrameStatus.TIMEOUT
            
        return FrameStatus.OK

    def is_bbox_sane(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> bool:
        """YOLO や VLM からのバウンディングボックスの妥当性を検証し、ハルシネーションによる異常値を検知する。"""
        _, _, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        bbox_area = w * h
        frame_area = frame_w * frame_h
        
        if frame_area <= 0:
            return False

        area_ratio = bbox_area / frame_area
        
        # BBox が画面の75%を超える場合、または座標サイズが負である場合は異常値（ハルシネーション）とみなす
        if area_ratio > self.max_area_ratio or w <= 0 or h <= 0:
            logger.warning(
                f"Abnormal BBox detected (area ratio: {area_ratio:.2%}, size: {w}x{h}). "
                f"Flagged as potential hallucination."
            )
            return False
            
        return True

    def check_night_mode(self, frame: np.ndarray) -> bool:
        """カメラの入力画像から、夜間（赤外線・モノクロ）モードに入っているか自動検知する。"""
        if frame is None or frame.size == 0:
            return False

        # HSV空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 彩度（Saturation）チャンネルを取得
        s_channel = hsv[:, :, 1]
        
        # 彩度の標準偏差（または平均）を算出。モノクロ画像であれば全画素の彩度がほぼ 0 に近く、バラつき（std）も極小になる
        s_std = float(np.std(s_channel))
        
        is_night = s_std < self.night_std_threshold
        if is_night:
            logger.debug(f"Night (monochrome) mode detected automatically (Saturation std: {s_std:.2f} < {self.night_std_threshold})")
            
        return is_night
