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
from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader
from pico.detector import YoloDetector
from pico.tracker import SimpleIoUTracker
from pico.ollama_client import OllamaVisionClient
from pico.guardrails import GuardRails, FrameStatus

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

class MonitorWindow:
    """デスクトップ上に OpenCV ウィンドウを立ち上げ、常時映像と追尾状態を描画するモニタースレッド"""
    def __init__(self, reader: RTSPVideoReader, tracker: SimpleIoUTracker, ptz_actuator=None):
        self.reader = reader
        self.tracker = tracker
        self.ptz_actuator = ptz_actuator
        self.running = False
        self.thread = None

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
            ret, frame = self.reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            display_frame = frame.copy()
            img_h, img_w = display_frame.shape[:2]

            # PTZ から現在の追従状態とロックターゲットを取得
            lockon_id = None
            lockon_class = None
            if self.ptz_actuator:
                lockon_id = self.ptz_actuator.lockon_target_id
                lockon_class = self.ptz_actuator.lockon_class_name

            # 検出枠の描画
            active_tracker = self.tracker
            if self.ptz_actuator and getattr(self.ptz_actuator, "active_tracker", None) is not None:
                active_tracker = self.ptz_actuator.active_tracker

            for obj in list(active_tracker.tracked_objects.values()):
                x, y, w, h = obj.bbox
                class_name = COCO_CLASSES[obj.class_id] if obj.class_id < len(COCO_CLASSES) else f"unknown_{obj.class_id}"
                
                is_lockon = (lockon_id is not None and obj.track_id == lockon_id)
                
                color = (0, 0, 255) if is_lockon else (0, 255, 0)  # LOCKON対象は赤、それ以外は緑
                thickness = 3 if is_lockon else 2
                
                cv2.rectangle(display_frame, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)
                
                label = f"[LOCKON] ID {obj.track_id}: {class_name}" if is_lockon else f"ID {obj.track_id}: {class_name}"
                cv2.putText(display_frame, label, (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # PID制御偏差の可視化
                if is_lockon:
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)
                    mid_x = int(img_w / 2)
                    mid_y = int(img_h / 2)
                    
                    # 画面中央の目標点
                    cv2.drawMarker(display_frame, (mid_x, mid_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                    # ターゲットへの偏差ライン
                    cv2.line(display_frame, (mid_x, mid_y), (cx, cy), (0, 0, 255), 2)
                    
                    dx = cx - mid_x
                    dy = cy - mid_y
                    cv2.putText(display_frame, f"Diff X:{dx} Y:{dy}", (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # ステータスの表示
            status_text = f"LOCKON ACTIVE (Target ID: {lockon_id}, Class: {lockon_class})" if lockon_id is not None else "LOCKON IDLE"
            color_text = (0, 0, 255) if lockon_id is not None else (0, 255, 0)
            cv2.putText(display_frame, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)

            cv2.imshow("Cognitive Surveillance Monitor", display_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC キーで終了可能
                break

        cv2.destroyAllWindows()
        logger.info("📺 OpenCV Monitor window thread stopped.")


class OnDemandPerceptionCLI:
    """軽量YOLOテキスト常時取得 ＆ 1024pxクランプによるオンデマンドVLM起動を行う CLI"""
    def __init__(self, config: AppConfig, shared_reader: RTSPVideoReader = None, model_path: str = "yolov8s.onnx"):
        self.config = config
        self.model_path = model_path
        self.detector = None
        self.tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=30)
        self.vlm = OllamaVisionClient(base_url=config.ollama_base_url, model=config.ollama_model)
        self.guard = GuardRails(timeout_limit=3.0, max_area_ratio=0.75)
        
        if shared_reader is not None:
            self.reader = shared_reader
            self.is_reader_owned = False
        else:
            rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip)
            self.reader = RTSPVideoReader(rtsp_url)
            time.sleep(1.5)  # ストリームのバッファリング待ち
            self.is_reader_owned = True
        
        self.ptz_actuator = None
        self.monitor = None

    def _init_detector(self):
        if self.detector is None:
            if not os.path.exists(self.model_path):
                logger.info("⏳ Downloading YOLOv8 ONNX model...")
                url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"
                urllib.request.urlretrieve(url, self.model_path)
            self.detector = YoloDetector(model_path=self.model_path)

    def start_monitor(self):
        """必要時に OpenCV モニターウィンドウを自動起動する"""
        if self.monitor is None:
            self.monitor = MonitorWindow(self.reader, self.tracker, self.ptz_actuator)
            self.monitor.start()

    def set_ptz_actuator(self, ptz_actuator):
        """PTZコントローラーと連携してロックオン可視化を描画できるようにする"""
        self.ptz_actuator = ptz_actuator
        if self.monitor:
            self.monitor.ptz_actuator = ptz_actuator

    def get_tracks_data(self) -> list:
        """現在の追跡結果をリストで取得する"""
        self._init_detector()
        self.start_monitor()
        
        last_frame_t = self.reader.get_last_frame_time()
        if self.guard.check_frame_health(last_frame_t) == FrameStatus.TIMEOUT:
            logger.error("RTSP Stream Timeout. Failed to get tracks.")
            return []
            
        ret, frame = self.reader.read()
        if not ret or frame is None:
            return []

        detections = self.detector.detect(frame)
        sane = [d for d in detections if self.guard.is_bbox_sane(d.bbox, frame.shape)]
        tracked = self.tracker.update(sane)

        results = []
        for t in tracked:
            class_name = COCO_CLASSES[t.class_id] if t.class_id < len(COCO_CLASSES) else f"unknown_{t.class_id}"
            results.append({
                "track_id": t.track_id,
                "class": class_name,
                "bbox": list(t.bbox),
                "confidence": round(t.confidence, 2)
            })
        return results

    def get_tracks(self):
        """現在の追跡結果をテキスト（JSON）で取得する（VLMは起動しない）"""
        results = self.get_tracks_data()
        print(json.dumps({"tracks": results}, indent=2, ensure_ascii=False))

    def analyze_crop_data(self, track_id: int | None = None, class_filter: str | None = None, query: str = "") -> dict:
        """指定のトラックID、またはクラス曖昧指定に合致するBBox領域を光学切り出し -> 1024pxクランプしてVLMへ入力"""
        self._init_detector()
        self.start_monitor()
        
        last_frame_t = self.reader.get_last_frame_time()
        if self.guard.check_frame_health(last_frame_t) == FrameStatus.TIMEOUT:
            logger.error("RTSP Stream Timeout. Failed to analyze crop.")
            return {"error": "RTSP Stream Timeout"}
            
        ret, frame = self.reader.read()
        if not ret or frame is None:
            logger.error("Failed to read frame.")
            return {"error": "Failed to read frame"}

        detections = self.detector.detect(frame)
        sane = [d for d in detections if self.guard.is_bbox_sane(d.bbox, frame.shape)]
        tracked = self.tracker.update(sane)

        target = None
        
        # 1. track_id のマッチング試行
        if track_id is not None:
            target = next((t for t in tracked if t.track_id == track_id), None)
            
            # フォールバック 1: IoUマッチング
            if not target and track_id in self.tracker.tracked_objects:
                prev_obj = self.tracker.tracked_objects[track_id]
                best_iou = 0.0
                best_det = None
                for det in sane:
                    iou = self.tracker._calculate_iou(prev_obj.bbox, det.bbox)
                    if iou > best_iou and iou >= 0.2:
                        best_iou = iou
                        best_det = det
                if best_det:
                    logger.info(f"Smart Match: Track ID {track_id} matched via IoU ({best_iou:.2f})")
                    class DummyTarget:
                        def __init__(self, bbox):
                            self.bbox = bbox
                    target = DummyTarget(best_det.bbox)

        # 2. class_filter（曖昧マッチング）の試行
        if not target and class_filter is not None:
            # 追跡中の同一クラスで確信度最高のものを選択
            candidates = []
            for t in tracked:
                c_name = COCO_CLASSES[t.class_id] if t.class_id < len(COCO_CLASSES) else f"unknown_{t.class_id}"
                if c_name == class_filter:
                    candidates.append(t)
            
            if candidates:
                best_candidate = max(candidates, key=lambda x: x.confidence)
                logger.info(f"Smart Match: Found target for class '{class_filter}' (ID: {best_candidate.track_id})")
                target = best_candidate
            else:
                # 直近の生 YOLO 検出からマッチング
                class_idx = None
                try:
                    class_idx = COCO_CLASSES.index(class_filter)
                except ValueError:
                    pass
                
                if class_idx is not None:
                    raw_candidates = [d for d in sane if d.class_id == class_idx]
                    if raw_candidates:
                        best_raw = max(raw_candidates, key=lambda x: x.confidence)
                        logger.info(f"Smart Match: Found class '{class_filter}' from raw YOLO detection.")
                        class DummyTarget:
                            def __init__(self, bbox):
                                self.bbox = bbox
                        target = DummyTarget(best_raw.bbox)

        if not target:
            logger.error(f"Target not found for ID: {track_id}, class_filter: {class_filter}")
            return {"error": f"Target not found for ID: {track_id}, class_filter: {class_filter}"}

        x, y, w, h = target.bbox
        img_h, img_w = frame.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            logger.error("Cropped region is empty.")
            return {"error": "Cropped region is empty"}

        ch, cw = crop.shape[:2]
        max_size = 1024
        if cw > max_size or ch > max_size:
            scale = max_size / float(max(cw, ch))
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)

        # 念のためクロップ画像を monitor/latest_crop.jpg に書き出して確認できるようにする (GUIと併用)
        os.makedirs("monitor", exist_ok=True)
        cv2.imwrite("monitor/latest_crop.jpg", crop)

        logger.info(f"Sending cropped image (size {crop.shape[1]}x{crop.shape[0]}) to VLM for query: '{query}'")
        vlm_response = asyncio.run(self.vlm.analyze_scene(crop, query))
        
        if vlm_response:
            return {"status": "success", "response": vlm_response}
        else:
            return {"status": "error", "message": "Empty response from VLM"}

    def get_live_snapshot_data(self) -> dict:
        """現在の最新カメラ映像をキャプチャし、画像として保存してパスを返却する"""
        self.start_monitor()
        ret, frame = self.reader.read()
        if not ret or frame is None:
            return {"error": "Failed to read frame"}
        
        os.makedirs("monitor", exist_ok=True)
        save_path = "monitor/live_snapshot.jpg"
        cv2.imwrite(save_path, frame)
        return {"status": "success", "filepath": save_path}

    def analyze_crop(self, track_id: int, query: str):
        """CLI実行用: クロップとVLM解析"""
        res = self.analyze_crop_data(track_id=track_id, query=query)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    def close(self):
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        if getattr(self, "is_reader_owned", True):
            self.reader.release()
        asyncio.run(self.vlm.close())


def main():
    parser = argparse.ArgumentParser(description="Perception CLI tool")
    parser.add_argument("--action", choices=["get_tracks", "analyze_crop", "snapshot"], required=True, help="Action to perform")
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
