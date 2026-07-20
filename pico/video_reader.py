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
        self.cap: cv2.VideoCapture = cv2.VideoCapture(url)
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False
        self.last_frame_time: float = 0.0
        
        self.running: bool = True
        self.stop_event: threading.Event = threading.Event()
        self.lock: threading.Lock = threading.Lock()
        
        self.thread: threading.Thread = threading.Thread(target=self._keep_reading, daemon=True)
        self.thread.start()

    def _keep_reading(self) -> None:
        """バックグラウンドでフレームを常時読み込むループ。"""
        while not self.stop_event.is_set():
            if not self.cap.isOpened():
                logger.error(f"VideoCapture is not opened for url: {self.url}")
                time.sleep(1.0)
                continue
            
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.ret = ret
                        self.last_frame_time = time.monotonic()
                else:
                    # 通信途絶時のリトライのために短い待機
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error reading frame from RTSP stream: {e}")
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """最新のフレームをスレッド安全に取得する。

        Returns:
            Tuple[bool, Optional[np.ndarray]]: 取得成否の真偽値と、取得したフレーム画像(numpy.ndarray)。
        """
        with self.lock:
            return self.ret, self.frame

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
