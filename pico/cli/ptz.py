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
from pico.cli.perception import COCO_CLASSES

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
        self.lockon_target_id = None
        self.lockon_class_name = None
        self.active_tracker = None

    def _init_ptz(self, video_reader=None, align: bool | None = None):
        if align is None:
            align = getattr(self.config, "align_to_home", True)
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
                invert_tilt=self.config.invert_tilt,
                step_size_x=getattr(self.config, "step_size_x", 0.15),
                step_size_y=getattr(self.config, "step_size_y", 0.10),
                total_steps_x=getattr(self.config, "total_steps_x", 15),
                total_steps_y=getattr(self.config, "total_steps_y", 20),
                return_steps_x=getattr(self.config, "return_steps_x", None),
                return_steps_y=getattr(self.config, "return_steps_y", None),
                hunt_steps_x=getattr(self.config, "hunt_steps_x", 25),
                hunt_steps_y=getattr(self.config, "hunt_steps_y", 25)
            )

    def send_pulse_move(self, pan: float, tilt: float):
        """指定したパン・チルト量で安全クランプを適用してカメラを動かす（アライメント割り込みなし）"""
        self._init_ptz(align=False)
        # safe_move は指定量 requested_x, requested_y を現在位置に足してクランプしたのち送信する
        actual_x, actual_y = self.ptz.safe_move(pan, tilt)
        logger.info(f"Pulse Move Executed: Requested(pan={pan}, tilt={tilt}) -> Actual(pan={actual_x}, tilt={actual_y})")
        # 非同期送信キューの駆動ラグを考慮して待機
        time.sleep(0.2)
        return actual_x, actual_y

    def move_to_center(self):
        """カメラを推測中心原点(0.0, 0.0)へ復帰させる"""
        self._init_ptz(align=False)
        actual_x, actual_y = self.ptz.move_to_center()
        logger.info(f"Move to Center Executed: Actual(pan={actual_x}, tilt={actual_y}) | Est Pos: ({self.ptz.current_x:.2f}, {self.ptz.current_y:.2f})")
        time.sleep(0.5)
        return actual_x, actual_y

    def calibrate_home(self, video_reader=None):
        """物理限界ブラインド突き当てによるカメラゼロ点補正（ホームアライメント）を明示的に実行する"""
        self._init_ptz(align=False)
        self.ptz._align_to_home_position(video_reader=video_reader)
        logger.info("Explicit Home Alignment (Zero-point calibration) completed successfully.")
        return self.ptz.current_x, self.ptz.current_y

    def emergency_stop(self):
        """物理緊急停止（0.0移動を送信してキュー処理完了を待つ）"""
        self._init_ptz()
        self.ptz.relative_move(0.0, 0.0)
        logger.info("Emergency stop sent to camera PTZ queue.")
        time.sleep(0.2)

    def stop_lockon(self):
        """PID追従ループを停止する"""
        self.lockon_active = False
        self.lockon_target_id = None
        self.lockon_class_name = None
        logger.info("PTZ lockon stop requested.")

    def start_lockon(self, track_id: int | None = None, class_filter: str | None = None):
        """常時知覚エンジンと連携して、指定ターゲットの自動物理ロックオン追尾を開始する"""
        if track_id is None and class_filter is None:
            raise ValueError("Either track_id or class_filter must be provided for lockon.")
        self.lockon_active = True
        self.lockon_target_id = track_id
        self.lockon_class_name = class_filter
        logger.info(f"🎯 PTZ Lockon Activated. Target ID: {track_id}, Class: {class_filter}")

    def lockon(self, reader, track_id: int | None = None, class_filter: str | None = None):
        """指定のトラックID、または特定のオブジェクトクラス（曖昧指定）に対してリアルタイムPID追従ループを実行する"""
        if track_id is None and class_filter is None:
            raise ValueError("Either track_id or class_filter must be provided for lockon.")

        self.lockon_active = True
        self.lockon_target_id = track_id
        self.lockon_class_name = class_filter

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
            self.active_tracker = tracker

            logger.info(f"🎯 PTZ Lockon Loop started. ID: {track_id}, Class Filter: {class_filter}. Press Ctrl+C to exit.")
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

                # ターゲットの選定と同期
                target = None
                if self.lockon_target_id is not None:
                    # IDが指定（または決定）されている場合、そのオブジェクトを追尾
                    target = next((t for t in tracked if t.track_id == self.lockon_target_id), None)
                    if target is not None:
                        c_name = COCO_CLASSES[target.class_id] if target.class_id < len(COCO_CLASSES) else f"unknown_{target.class_id}"
                        self.lockon_class_name = c_name
                elif self.lockon_class_name is not None:
                    # IDが決まっていない、またはロストした場合に、指定クラスの最確オブジェクトを探して再捕捉 (Auto-Relock)
                    candidates = []
                    for t in tracked:
                        c_name = COCO_CLASSES[t.class_id] if t.class_id < len(COCO_CLASSES) else f"unknown_{t.class_id}"
                        if c_name == self.lockon_class_name:
                            candidates.append(t)
                    if candidates:
                        best_candidate = max(candidates, key=lambda x: x.confidence)
                        self.lockon_target_id = best_candidate.track_id
                        target = best_candidate
                        logger.info(f"🎯 Auto-Relock: Automatically locked onto class '{self.lockon_class_name}' (ID: {self.lockon_target_id})")

                # ONVIF コントローラーのロックIDを動的同期
                self.ptz.lock_on_id = self.lockon_target_id

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
                        # 追従対象を見失った場合の処理 (class_filter が指定されている場合はターゲットIDをリセットし次フレームでの再捕捉を促す)
                        if class_filter is not None and self.lockon_target_id is not None:
                            logger.warning(f"🎯 Target lost (ID: {self.lockon_target_id}). Scanning for another '{class_filter}'...")
                            self.lockon_target_id = None
                        self.pid.reset()
                    last_move_time = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Lockon loop interrupted by user.")
        finally:
            logger.info("Cleaning up resources...")
            self.lockon_active = False
            self.lockon_target_id = None
            self.lockon_class_name = None
            self.active_tracker = None
            if self.ptz:
                self.ptz.shutdown()

    def shutdown(self):
        if self.ptz:
            self.ptz.shutdown()

def main():
    parser = argparse.ArgumentParser(description="PTZ Actuator CLI tool")
    parser.add_argument("--action", choices=["lockon", "move", "stop"], required=True, help="Action to perform")
    parser.add_argument("--id", type=int, help="Track ID to lockon")
    parser.add_argument("--class", dest="class_name", type=str, help="Object class name to lockon (e.g., 'person', 'suitcase')")
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
            if args.id is None and args.class_name is None:
                parser.error("Either --id or --class is required for lockon action")
            rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip)
            reader = RTSPVideoReader(rtsp_url)
            time.sleep(1.5)  # ストリーム開始待ち
            try:
                actuator.lockon(reader, track_id=args.id, class_filter=args.class_name)
            finally:
                reader.release()
    finally:
        actuator.shutdown()

if __name__ == "__main__":
    main()
