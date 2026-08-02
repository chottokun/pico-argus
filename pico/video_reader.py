import cv2
import threading
import time
import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class RTSPVideoReader:
    """RTSP ストリームからバックグラウンドスレッドで映像を取得し、最新のフレームのみを保持するデコーダークラス。"""

    def __init__(self, url: str) -> None:
        self.url: str = url
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False
        self.last_frame_time: float = time.monotonic()
        
        self.running: bool = True
        self.stop_event: threading.Event = threading.Event()
        self.lock: threading.Lock = threading.Lock()
        
        self._connect_cap()
        self.thread: threading.Thread = threading.Thread(target=self._keep_reading, daemon=True)
        self.thread.start()

    def _connect_cap(self) -> None:
        """RTSPキャプチャをオープンし、FFmpegの内部バッファを最小1に制限して遅延をゼロ化する。"""
        with self.lock:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            # 内部バッファサイズを最小1に設定（遅延・スタック防止）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _keep_reading(self) -> None:
        """バックグラウンドでフレームを常時読み込むループ。自動再接続ウォッチドッグ付き。"""
        consecutive_failures = 0

        while not self.stop_event.is_set():
            # ウォッチドッグ: 3.0秒以上フレームが更新されていない、または連続失敗時は自動再接続
            now = time.monotonic()
            with self.lock:
                stale = (now - self.last_frame_time) > 3.0
                opened = self.cap is not None and self.cap.isOpened()

            if not opened or stale or consecutive_failures >= 5:
                logger.warning(f"⚠️ RTSP stream stale or connection lost (stale={stale}, fails={consecutive_failures}). Reconnecting to {self.url}...")
                self._connect_cap()
                consecutive_failures = 0
                time.sleep(0.5)
                continue
            
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.ret = ret
                        self.last_frame_time = time.monotonic()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    time.sleep(0.02)
            except Exception as e:
                logger.error(f"Error reading frame from RTSP stream: {e}")
                consecutive_failures += 1
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """最新のフレームをスレッド安全に取得する。

        Returns:
            Tuple[bool, Optional[np.ndarray]]: 取得成否の真偽値と、取得したフレーム画像(numpy.ndarray)。
        """
        with self.lock:
            return self.ret, self.frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """最新のフレーム画像(numpy.ndarray)を直接取得する。"""
        with self.lock:
            return self.frame if self.ret else None

    def get_last_frame_time(self) -> float:
        """最新フレームを取得した時刻（タイムスタンプ）を取得する。"""
        with self.lock:
            return self.last_frame_time

    def release(self) -> None:
        """読み込みスレッドを停止し、キャプチャリソースを解放する。"""
        self.running = False
        self.stop_event.set()
        
        # スレッドが終了するのを少し待つ
        self.thread.join(timeout=0.5)
        
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()
            logger.info("RTSPVideoReader released resources successfully.")
