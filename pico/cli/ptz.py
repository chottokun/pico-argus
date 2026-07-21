import argparse
import time
import logging
import os
import urllib.request
import sys
from pico.config import AppConfig, build_rtsp_url
from pico.onvif_client import PTZController
from pico.pid_controller import AdaptivePIDController
from pico.video_reader import RTSPVideoReader
from pico.detector import YoloDetector
from pico.tracker import SimpleIoUTracker
from pico.guardrails import GuardRails, FrameStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PTZActuator:
    """ONVIF PTZ物理制御 & 衝突防止セーフガード (RelativeMove + safe_move クランプ)"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.ptz = None
        self.pid = AdaptivePIDController(
            kp_base=0.35,
            ki=0.03,
            kd=0.005,
            dead_zone=0.10,
            min_speed=0.03,
            max_step=0.12,
            integral_limit=0.2
        )
        self.guard = GuardRails(timeout_limit=3.0, max_area_ratio=0.75)
        self.lockon_active = False

    def _init_ptz(self, video_reader=None, align=False):
        if self.ptz is None:
            self.ptz = PTZController(
                ip=self.config.tapo_ip,
                user=self.config.tapo_user,
                password=self.config.tapo_pass,
                max_limit_x=self.config.max_limit_x,
                max_limit_y=self.config.max_limit_y,
                align_to_home=align,
                video_reader=video_reader,
                invert_pan=self.config.invert_pan,
                invert_tilt=self.config.invert_tilt
            )

    def send_pulse_move(self, pan: float, tilt: float):
        """指定したパン・チルト量で安全クランプを適用してカメラを動かす"""
        self._init_ptz()
        # safe_move は指定量 requested_x, requested_y を現在位置に足してクランプしたのち送信する
        actual_x, actual_y = self.ptz.safe_move(pan, tilt)
        logger.info(f"Pulse Move Executed: Requested(pan={pan}, tilt={tilt}) -> Actual(pan={actual_x}, tilt={actual_y})")
        # 非同期送信キューの駆動ラグを考慮して待機
        time.sleep(0.2)

    def emergency_stop(self):
        """物理緊急停止（0.0移動を送信してキュー処理完了を待つ）"""
        self._init_ptz()
        self.ptz.relative_move(0.0, 0.0)
        logger.info("Emergency stop sent to camera PTZ queue.")
        time.sleep(0.2)

    def stop_lockon(self):
        """PID追従ループを停止する"""
        self.lockon_active = False
        logger.info("PTZ lockon stop requested.")

    def lockon(self, track_id: int):
        """指定のトラックIDに対してリアルタイムPID追従ループを実行する"""
        self.lockon_active = True
        rtsp_url = build_rtsp_url(self.config.tapo_user, self.config.tapo_pass, self.config.tapo_ip)
        reader = RTSPVideoReader(rtsp_url)
        time.sleep(1.5)  # ストリーム開始待ち

        try:
            self._init_ptz(video_reader=reader, align=self.config.align_to_home)
            self.ptz.lock_on_id = track_id

            onnx_path = "yolov8s.onnx"
            if not os.path.exists(onnx_path):
                logger.info("⏳ Downloading YOLOv8 ONNX model...")
                url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"
                urllib.request.urlretrieve(url, onnx_path)

            detector = YoloDetector(model_path=onnx_path)
            tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=30)

            logger.info(f"🎯 PTZ Lockon Loop started for Track ID: {track_id}. Press Ctrl+C to exit.")
            last_move_time = time.monotonic()
            TRACK_INTERVAL = 0.45

            while self.lockon_active:
                last_frame_t = reader.get_last_frame_time()
                if self.guard.check_frame_health(last_frame_t) == FrameStatus.TIMEOUT:
                    logger.warning("⚠️ RTSP feed lost. Pausing control...")
                    self.pid.reset()
                    time.sleep(0.5)
                    continue

                ret, frame = reader.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                detections = detector.detect(frame)
                sane = [d for d in detections if self.guard.is_bbox_sane(d.bbox, frame.shape)]
                tracked = tracker.update(sane)

                target = next((t for t in tracked if t.track_id == self.ptz.lock_on_id), None)
                current_time = time.monotonic()
                dt = current_time - last_move_time

                if dt > TRACK_INTERVAL:
                    if target is not None:
                        x, y, w, h = target.bbox
                        cx = (x + w / 2) / frame.shape[1]
                        cy = (y + h / 2) / frame.shape[0]

                        dx, dy = self.pid.calculate_step(cx, 1.0 - cy, dt)
                        if dx != 0.0 or dy != 0.0:
                            self.ptz.safe_move(dx, dy)
                    else:
                        logger.warning(f"🎯 Lock-on Target ID {self.ptz.lock_on_id} not visible in scene.")
                        self.pid.reset()
                    last_move_time = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Lockon loop interrupted by user.")
        finally:
            logger.info("Cleaning up resources...")
            reader.release()
            if self.ptz:
                self.ptz.shutdown()

    def shutdown(self):
        if self.ptz:
            self.ptz.shutdown()

def main():
    parser = argparse.ArgumentParser(description="PTZ Actuator CLI tool")
    parser.add_argument("--action", choices=["lockon", "move", "stop"], required=True, help="Action to perform")
    parser.add_argument("--id", type=int, help="Track ID to lockon")
    parser.add_argument("--pan", type=type(0.0), default=0.0, help="Relative pan offset (-1.0 to 1.0)")
    parser.add_argument("--tilt", type=type(0.0), default=0.0, help="Relative tilt offset (-1.0 to 1.0)")
    args = parser.parse_args()

    try:
        config = AppConfig()
    except Exception as e:
        logger.error(f"Configuration load failed: {e}")
        sys.exit(1)

    actuator = PTZActuator(config)
    try:
        if args.action == "move":
            actuator.send_pulse_move(args.pan, args.tilt)
        elif args.action == "stop":
            actuator.emergency_stop()
        elif args.action == "lockon":
            if args.id is None:
                parser.error("--id is required for lockon action")
            actuator.lockon(args.id)
    finally:
        actuator.shutdown()

if __name__ == "__main__":
    main()
