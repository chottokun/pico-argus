import argparse
import asyncio
import cv2
import json
import logging
import os
import sys
import time
import urllib.request
import threading
from typing import List, Dict, Any, Optional

from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader
from pico.detector import YoloDetector
from pico.tracker import SimpleIoUTracker
from pico.ollama_client import OllamaVisionClient
from pico.guardrails import GuardRails
from pico.event_engine import PerceptionEventEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "maze", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class ContinuousPerceptionLoop:
    """バックグラウンドで15〜30 FPSのYOLO+ByteTrack連続検知およびイベント過剰発火制御ループを実行する常時知覚エンジン"""

    def __init__(
        self,
        reader: RTSPVideoReader,
        model_path: str = "yolov8s.onnx",
        ptz_actuator: Optional[Any] = None,
        min_stable_frames: int = 3,
        cooldown_sec: float = 5.0
    ):
        self.reader = reader
        self.model_path = model_path
        self.ptz_actuator = ptz_actuator
        self.detector: Optional[YoloDetector] = None
        self.tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=30)
        self.guard = GuardRails(timeout_limit=3.0, max_area_ratio=0.75)
        self.event_engine = PerceptionEventEngine(min_stable_frames=min_stable_frames, cooldown_sec=cooldown_sec)

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # キャッシュ状態
        self._latest_tracks: List[Dict[str, Any]] = []
        self._latest_display_frame: Optional[Any] = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = time.time()
        self._last_events_summary: List[Dict[str, Any]] = []

    def _init_detector(self):
        if self.detector is None:
            if not os.path.exists(self.model_path):
                logger.info("⏳ Downloading YOLOv8 ONNX model...")
                url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"
                urllib.request.urlretrieve(url, self.model_path)
            self.detector = YoloDetector(model_path=self.model_path)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("🚀 Continuous Perception Loop thread started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            logger.info("🛑 Continuous Perception Loop thread stopped.")

    def _run(self):
        self._init_detector()
        self._start_time = time.time()
        self._frame_count = 0

        while self.running:
            loop_start = time.time()
            ret, frame = self.reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # YOLO検出 ＆ トラッカー更新
            detections = self.detector.detect(frame)
            sane = [d for d in detections if self.guard.is_bbox_sane(d.bbox, frame.shape)]
            
            # active_tracker の参照（PTZ追尾中であればPTZのトラッカー、無ければ自前のトラッカー）
            active_tracker = self.tracker
            if self.ptz_actuator and getattr(self.ptz_actuator, "active_tracker", None) is not None:
                active_tracker = self.ptz_actuator.active_tracker
                
            tracked = active_tracker.update(sane)

            # トラックリストメタデータの構築
            track_list = []
            for t in tracked:
                class_name = COCO_CLASSES[t.class_id] if t.class_id < len(COCO_CLASSES) else f"unknown_{t.class_id}"
                t.class_name = class_name
                track_list.append({
                    "track_id": t.track_id,
                    "class": class_name,
                    "bbox": list(t.bbox),
                    "confidence": round(t.confidence, 2)
                })

            # 能動的イベントエンジンの処理
            emitted_events = self.event_engine.process_frame(tracked)
            if emitted_events:
                for evt in emitted_events:
                    logger.info(f"⚡ [PERCEPTION EVENT] {evt.event_type.value}: Track {evt.track_id} ({evt.class_name})")

            # モニター描画用アノテーションフレームの作成
            display_frame = frame.copy()
            img_h, img_w = display_frame.shape[:2]

            lockon_id = None
            lockon_class = None
            if self.ptz_actuator:
                lockon_id = self.ptz_actuator.lockon_target_id
                lockon_class = self.ptz_actuator.lockon_class_name

            for obj in tracked:
                x, y, w, h = obj.bbox
                c_name = getattr(obj, "class_name", "unknown")
                is_lockon = (lockon_id is not None and obj.track_id == lockon_id)
                color = (0, 0, 255) if is_lockon else (0, 255, 0)
                thickness = 3 if is_lockon else 2

                cv2.rectangle(display_frame, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
                label = f"[LOCKON] ID {obj.track_id}: {c_name}" if is_lockon else f"ID {obj.track_id}: {c_name}"
                cv2.putText(display_frame, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if is_lockon:
                    cx, cy = int(x + w / 2), int(y + h / 2)
                    mid_x, mid_y = int(img_w / 2), int(img_h / 2)
                    cv2.drawMarker(display_frame, (mid_x, mid_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                    cv2.line(display_frame, (mid_x, mid_y), (cx, cy), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Diff X:{cx - mid_x} Y:{cy - mid_y}", (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            status_text = f"LOCKON ACTIVE (Target ID: {lockon_id}, Class: {lockon_class})" if lockon_id is not None else "LOCKON IDLE"
            cv2.putText(display_frame, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            recent = self.event_engine.get_status_summary()["recent_events"]
            if recent:
                latest_evt = recent[0]
                evt_text = f"EVENT: {latest_evt['event_type']} -> ID {latest_evt['track_id']} ({latest_evt['class_name']})"
                cv2.putText(display_frame, evt_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            self._frame_count += 1
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self._fps = round(self._frame_count / elapsed, 1)

            with self._lock:
                self._latest_tracks = track_list
                self._latest_display_frame = display_frame
                self._last_events_summary = recent

            elapsed_loop = time.time() - loop_start
            sleep_time = max(0.001, 0.033 - elapsed_loop)
            time.sleep(sleep_time)

    def get_cached_tracks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._latest_tracks)

    def get_latest_frame(self) -> Optional[Any]:
        with self._lock:
            return self._latest_display_frame.copy() if self._latest_display_frame is not None else None

    def get_status(self) -> Dict[str, Any]:
        summary = self.event_engine.get_status_summary()
        with self._lock:
            return {
                "engine_status": "RUNNING" if self.running else "STOPPED",
                "fps": self._fps,
                "active_track_count": len(self._latest_tracks),
                "active_tracks": list(self._latest_tracks),
                "cooldown_sec": summary["cooldown_sec"],
                "min_stable_frames": summary["min_stable_frames"],
                "allowed_classes": summary["allowed_classes"],
                "recent_events": self._last_events_summary
            }

    def configure_event_filter(self, cooldown_sec: Optional[float] = None, allowed_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        if cooldown_sec is not None:
            self.event_engine.cooldown_sec = cooldown_sec
        if allowed_classes is not None:
            self.event_engine.set_allowed_classes(allowed_classes)
        return self.get_status()


class MonitorWindow:
    def __init__(self, perception_loop: ContinuousPerceptionLoop):
        self.perception_loop = perception_loop
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _run(self):
        logger.info("📺 OpenCV Monitor window thread started.")
        cv2.namedWindow("Cognitive Surveillance Monitor", cv2.WINDOW_NORMAL)

        while self.running:
            frame = self.perception_loop.get_latest_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            cv2.imshow("Cognitive Surveillance Monitor", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()
        logger.info("📺 OpenCV Monitor window thread stopped.")


class OnDemandPerceptionCLI:
    """常時バックグラウンドYOLO知覚および0ms低遅延アクセスを提供する CLI / Core インターフェース"""

    def __init__(self, config: AppConfig, shared_reader: Optional[RTSPVideoReader] = None, model_path: str = "yolov8s.onnx"):
        self.config = config
        self.model_path = model_path
        self.vlm = OllamaVisionClient(base_url=config.ollama_base_url, model=config.ollama_model)
        self.guard = GuardRails(timeout_limit=3.0, max_area_ratio=0.75)

        if shared_reader is not None:
            self.reader = shared_reader
            self.is_reader_owned = False
        else:
            rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip)
            self.reader = RTSPVideoReader(rtsp_url)
            time.sleep(1.5)
            self.is_reader_owned = True

        self.ptz_actuator = None
        self.perception_loop = ContinuousPerceptionLoop(reader=self.reader, model_path=model_path)
        self.perception_loop.start()
        self.monitor: Optional[MonitorWindow] = None

    @property
    def detector(self):
        return self.perception_loop.detector

    @detector.setter
    def detector(self, value):
        self.perception_loop.detector = value

    @property
    def tracker(self):
        return self.perception_loop.tracker

    @tracker.setter
    def tracker(self, value):
        self.perception_loop.tracker = value

    def start_monitor(self):
        if self.monitor is None:
            self.monitor = MonitorWindow(self.perception_loop)
            self.monitor.start()

    def set_ptz_actuator(self, ptz_actuator):
        self.ptz_actuator = ptz_actuator
        self.perception_loop.ptz_actuator = ptz_actuator

    def get_tracks_data(self) -> List[Dict[str, Any]]:
        """最新の常時追跡オブジェクトメタデータを取得（モック環境等で空なら同期フォールバック）"""
        self.start_monitor()
        tracks = self.perception_loop.get_cached_tracks()

        if not tracks:
            # モック環境やスレッド起動ラグ対策の同期フォールバック
            ret, frame = self.reader.read()
            if ret and frame is not None:
                self.perception_loop._init_detector()
                detections = self.perception_loop.detector.detect(frame)
                sane = [d for d in detections if self.guard.is_bbox_sane(d.bbox, frame.shape)]
                tracked = self.perception_loop.tracker.update(sane)
                results = []
                for t in tracked:
                    class_name = COCO_CLASSES[t.class_id] if t.class_id < len(COCO_CLASSES) else f"unknown_{t.class_id}"
                    results.append({
                        "track_id": t.track_id,
                        "class": class_name,
                        "bbox": list(t.bbox),
                        "confidence": round(t.confidence, 2)
                    })
                with self.perception_loop._lock:
                    self.perception_loop._latest_tracks = results
                return results
        return tracks

    def get_tracks(self):
        results = self.get_tracks_data()
        print(json.dumps({"tracks": results}, indent=2, ensure_ascii=False))

    def get_perception_status_data(self) -> Dict[str, Any]:
        self.start_monitor()
        return self.perception_loop.get_status()

    def configure_event_filter_data(self, cooldown_sec: Optional[float] = None, allowed_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.perception_loop.configure_event_filter(cooldown_sec=cooldown_sec, allowed_classes=allowed_classes)

    def analyze_crop_data(self, track_id: Optional[int] = None, class_filter: Optional[str] = None, query: str = "") -> dict:
        self.start_monitor()
        ret, frame = self.reader.read()
        if not ret or frame is None:
            return {"error": "Failed to read frame"}

        # 最新の検出・追跡状態を同期更新
        self.get_tracks_data()

        target_bbox = None
        # 1. track_id マッチング
        if track_id is not None:
            if track_id in self.perception_loop.tracker.tracked_objects:
                target_bbox = list(self.perception_loop.tracker.tracked_objects[track_id].bbox)

        # 2. class_filter マッチング
        if not target_bbox and class_filter is not None:
            tracked_list = self.perception_loop.get_cached_tracks()
            for t in tracked_list:
                if t["class"] == class_filter:
                    target_bbox = t["bbox"]
                    break

        if track_id is None and class_filter is None:
            crop = frame
        else:
            if not target_bbox:
                return {"error": f"Target not found for ID: {track_id}, class_filter: {class_filter}"}

            x, y, w, h = map(int, target_bbox)
            img_h, img_w = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            return {"error": "Cropped region is empty"}

        ch, cw = crop.shape[:2]
        max_size = 1024
        if cw > max_size or ch > max_size:
            scale = max_size / float(max(cw, ch))
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)

        os.makedirs("monitor", exist_ok=True)
        cv2.imwrite("monitor/latest_crop.jpg", crop)

        vlm_response = asyncio.run(self.vlm.analyze_scene(crop, query))
        if vlm_response:
            return {"status": "success", "response": vlm_response}
        else:
            return {"status": "error", "message": "Empty response from VLM"}

    def analyze_crop(self, track_id: Optional[int] = None, class_filter: Optional[str] = None, query: str = ""):
        res = self.analyze_crop_data(track_id=track_id, class_filter=class_filter, query=query)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    def get_live_snapshot_data(self) -> dict:
        self.start_monitor()
        ret, frame = self.reader.read()
        if not ret or frame is None:
            return {"error": "Failed to read frame"}
        os.makedirs("monitor", exist_ok=True)
        save_path = "monitor/live_snapshot.jpg"
        cv2.imwrite(save_path, frame)
        return {"status": "success", "filepath": save_path}

    def close(self):
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        if self.perception_loop:
            self.perception_loop.stop()
        if getattr(self, "is_reader_owned", True):
            self.reader.release()
        try:
            asyncio.run(self.vlm.close())
        except Exception as e:
            logger.debug(f"VLM client close exception ignored: {e}")


def main():
    parser = argparse.ArgumentParser(description="Perception CLI tool")
    parser.add_argument("--action", choices=["get_tracks", "analyze_crop", "snapshot", "status"], required=True, help="Action to perform")
    parser.add_argument("--id", type=int, help="Track ID to crop and analyze")
    parser.add_argument("--class", dest="class_name", type=str, help="Object class filter (e.g. 'person')")
    parser.add_argument("--query", type=str, help="VLM text query")
    parser.add_argument("--model", type=str, default="yolov8s.onnx", help="YOLOv8 ONNX model path")
    args = parser.parse_args()

    try:
        config = AppConfig()
    except Exception as e:
        logger.error(f"Configuration load failed: {e}")
        sys.exit(1)

    cli = OnDemandPerceptionCLI(config, model_path=args.model)
    try:
        if args.action == "get_tracks":
            cli.get_tracks()
        elif args.action == "status":
            res = cli.get_perception_status_data()
            print(json.dumps(res, indent=2, ensure_ascii=False))
        elif args.action == "snapshot":
            res = cli.get_live_snapshot_data()
            print(json.dumps(res, indent=2, ensure_ascii=False))
        elif args.action == "analyze_crop":
            if args.id is None and args.class_name is None:
                parser.error("Either --id or --class is required for analyze_crop")
            if args.query is None:
                parser.error("--query is required for analyze_crop")
            res = cli.analyze_crop_data(track_id=args.id, class_filter=args.class_name, query=args.query)
            print(json.dumps(res, indent=2, ensure_ascii=False))
    finally:
        cli.close()

if __name__ == "__main__":
    main()
