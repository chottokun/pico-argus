import argparse
import asyncio
import cv2
import json
import logging
import os
import sys
import time
import urllib.request
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

class OnDemandPerceptionCLI:
    """軽量YOLOテキスト常時取得 ＆ 1024pxクランプによるオンデマンドVLM起動を行う CLI"""
    def __init__(self, config: AppConfig, model_path: str = "yolov8s.onnx"):
        self.config = config
        self.model_path = model_path
        self.detector = None
        self.tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=30)
        self.vlm = OllamaVisionClient(base_url=config.ollama_base_url, model=config.ollama_model)
        self.guard = GuardRails(timeout_limit=3.0, max_area_ratio=0.75)
        
        rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip)
        self.reader = RTSPVideoReader(rtsp_url)
        time.sleep(1.5)  # ストリームのバッファリング待ち

    def _init_detector(self):
        if self.detector is None:
            if not os.path.exists(self.model_path):
                logger.info("⏳ Downloading YOLOv8 ONNX model...")
                url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"
                urllib.request.urlretrieve(url, self.model_path)
            self.detector = YoloDetector(model_path=self.model_path)

    def get_tracks_data(self) -> list:
        """現在の追跡結果をリストで取得する (printなし)"""
        self._init_detector()
        
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

    def analyze_crop_data(self, track_id: int, query: str) -> dict:
        """指定のトラックIDのBBox領域をクランプ・VLM入力して結果を辞書で返却 (printなし)"""
        self._init_detector()
        
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

        target = next((t for t in tracked if t.track_id == track_id), None)
        if not target:
            logger.error(f"Track ID {track_id} not found in current scene.")
            return {"error": f"Track ID {track_id} not found"}

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

        logger.info(f"Sending cropped image (size {crop.shape[1]}x{crop.shape[0]}) to VLM for query: '{query}'")
        vlm_response = asyncio.run(self.vlm.analyze_scene(crop, query))
        
        if vlm_response:
            return {"status": "success", "response": vlm_response}
        else:
            return {"status": "error", "message": "Empty response from VLM"}

    def analyze_crop(self, track_id: int, query: str):
        """指定のトラックIDのBBox領域を高解像度で光学切り出し -> 1024px以下にクランプしてVLMへ入力"""
        res = self.analyze_crop_data(track_id, query)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    def close(self):
        self.reader.release()
        asyncio.run(self.vlm.close())

def main():
    parser = argparse.ArgumentParser(description="Perception CLI tool")
    parser.add_argument("--action", choices=["get_tracks", "analyze_crop"], required=True, help="Action to perform")
    parser.add_argument("--id", type=int, help="Track ID to crop and analyze")
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
        elif args.action == "analyze_crop":
            if args.id is None or args.query is None:
                parser.error("--id and --query are required for analyze_crop action")
            cli.analyze_crop(args.id, args.query)
    finally:
        cli.close()

if __name__ == "__main__":
    main()
