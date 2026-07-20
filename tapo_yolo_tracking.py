import os
import cv2
import time
import urllib.request
import logging
import asyncio
import threading
from dotenv import load_dotenv

from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader
from pico.onvif_client import PTZController
from pico.pid_controller import AdaptivePIDController
from pico.detector import YoloDetector
from pico.tracker import SimpleIoUTracker
from pico.ollama_client import OllamaVisionClient
from pico.memory import MemoryStore
from pico.cognition import CognitionEngine
from pico.guardrails import GuardRails, FrameStatus

# --- 📁 0. ログシステムの設定 ---
log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("tapo_tracking_experiment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

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

load_dotenv()

try:
    config = AppConfig()
    ptz = PTZController(
        ip=config.tapo_ip,
        user=config.tapo_user,
        password=config.tapo_pass,
        max_limit_x=config.max_limit_x,
        max_limit_y=config.max_limit_y
    )
    logging.info("✅ ONVIF/PTZ初期化成功。")
except Exception as e:
    logging.error(f"ONVIF初期化エラー: {e}")
    raise SystemExit(1)

# --- 🎯 1. 制御用パラメータ & 各モジュールの初期化 ---
TRACK_TARGET_ID = 0  # COCOでの 'person' ID は 0
CONF_THRESHOLD = 0.45
TARGET_CONF_THRESHOLD = 0.60

pid = AdaptivePIDController(
    kp_base=0.35,
    ki=0.03,
    kd=0.005,
    dead_zone=0.10,    # 通常時の不感帯
    min_speed=0.03,
    max_step=0.12,
    integral_limit=0.2
)

# YOLOv8-small ONNXモデルの準備
onnx_path = "yolov8s.onnx"
if not os.path.exists(onnx_path):
    logging.info("⏳ 高精度版 YOLOv8-small ONNXモデルをダウンロード中...")
    url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"
    urllib.request.urlretrieve(url, onnx_path)

detector = YoloDetector(model_path=onnx_path, conf_threshold=CONF_THRESHOLD)
tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=3)
guard = GuardRails(timeout_limit=3.0, max_area_ratio=0.75)

TRACK_INTERVAL = 0.45
last_move_time = 0.0

# --- 🧠 2. 認知ループ（Ollama + SQLite 長期記憶）のバックグラウンド起動 ---
memory_store = MemoryStore(db_path="wiki.db")
memory_store.add_entry(
    filepath="rules/general.md",
    doc_type="rule",
    title="エッジセキュリティ基本指示",
    tags="security person warning",
    content="人(person)を検知した場合、不審行動を分析すること。gemma4:e2bは最優先で使用すること。"
)

vlm_client = OllamaVisionClient(base_url=config.ollama_base_url, model=config.ollama_model)
cognition_engine = CognitionEngine(vlm_client=vlm_client, memory_store=memory_store)

cognition_loop = asyncio.new_event_loop()

def start_cognition_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(cognition_engine.run())

cognition_thread = threading.Thread(
    target=start_cognition_thread, 
    args=(cognition_loop,), 
    daemon=True
)
cognition_thread.start()

submitted_track_ids = set()

# --- 📹 3. RTSP映像ストリームの開始 ---
rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip, "stream1")
video_reader = RTSPVideoReader(rtsp_url)

time.sleep(1.0)
ret, test_frame = video_reader.read()
if not ret or test_frame is None:
    logging.error("映像ストリーム受信失敗")
    video_reader.release()
    ptz.shutdown()
    raise SystemExit(1)

width, height = int(test_frame.shape[1]), int(test_frame.shape[0])
center_x, center_y = width // 2, height // 2

logging.info("🚀 反射・認知・安全ガードレール統合追尾システム稼働！")

try:
    while True:
        # 🔒 安全ガードレール: フレーム時間監視 (RTSP断絶検知)
        last_frame_t = video_reader.get_last_frame_time()
        frame_health = guard.check_frame_health(last_frame_t)
        
        if frame_health == FrameStatus.TIMEOUT:
            logging.warning("⚠️ [Safety Guard] RTSP feed lost. Pausing control loop...")
            pid.reset()  # 積分値リセット
            time.sleep(0.5)
            continue

        ret, frame = video_reader.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # 🔒 安全ガードレール: 夜間低照度の自動検知とパラメータ自動チューニング
        is_night = guard.check_night_mode(frame)
        if is_night:
            # 夜間は不感帯を 5% に狭めて反応性を引き上げる
            pid.dead_zone = 0.05
        else:
            pid.dead_zone = 0.10

        # 検出と追跡の実行
        detections = detector.detect(frame)
        
        # 🔒 安全ガードレール: ハルシネーション（異常面積BBox）のフィルタリング
        sane_detections = []
        for det in detections:
            if guard.is_bbox_sane(det.bbox, frame.shape):
                sane_detections.append(det)
        
        tracked_objects = tracker.update(sane_detections)

        # ガイドライン描画
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 1)
        
        # 現在の不感帯を黄色い枠で描画
        cv2.rectangle(
            frame, 
            (center_x - int(width * pid.dead_zone), center_y - int(height * pid.dead_zone)), 
            (center_x + int(width * pid.dead_zone), center_y + int(height * pid.dead_zone)), 
            (0, 255, 255), 1
        )
        
        # 夜間検知時の表示
        if is_night:
            cv2.putText(frame, "NIGHT MODE ACTIVE (DeadZone: 5%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        track_candidates = []
        detected_objects_summary = []

        for obj in tracked_objects:
            x, y, w, h = obj.bbox
            cid = obj.class_id
            conf = obj.confidence
            track_id = obj.track_id
            class_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"ID {cid}"
            
            detected_objects_summary.append(f"{class_name}#{track_id}({conf:.2f})")

            if cid == TRACK_TARGET_ID and conf >= TARGET_CONF_THRESHOLD:
                color = (0, 255, 0)
                track_candidates.append(obj)
                
                # 🧠 認知イベントの非同期トリガー
                if track_id not in submitted_track_ids:
                    submitted_track_ids.add(track_id)
                    
                    analysis_frame = frame.copy()
                    event = {
                        "class_name": class_name,
                        "track_id": track_id,
                        "frame": analysis_frame
                    }
                    cognition_engine.trigger_event_from_thread(event, cognition_loop)
                    logging.info(f"🧠 [Event Triggered] Sent {class_name}#{track_id} to Cognition Loop.")
            else:
                color = (0, 165, 255)
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{class_name}#{track_id}: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 優先ターゲットの選定
        main_track_obj = None
        if len(track_candidates) > 0:
            main_track_obj = max(track_candidates, key=lambda o: o.bbox[2] * o.bbox[3])
            mx, my, mw, mh = main_track_obj.bbox
            person_cx, person_cy = mx + mw // 2, my + mh // 2
            cv2.circle(frame, (person_cx, person_cy), 6, (0, 0, 255), -1)

        # --- ⏱️ モーター制御（反射ループ） ---
        current_time = time.monotonic()
        dt = current_time - last_move_time

        if dt > TRACK_INTERVAL:
            if detected_objects_summary:
                logging.info(f"👁️ 画面内の追跡状況: {', '.join(detected_objects_summary)}")
                
            if main_track_obj is not None:
                mx, my, mw, mh = main_track_obj.bbox
                person_cx, person_cy = mx + mw // 2, my + mh // 2
                
                # 正規化
                norm_cx = person_cx / width
                norm_cy = person_cy / height
                
                # PID計算
                dx, dy = pid.calculate_step(norm_cx, 1.0 - norm_cy, dt)
                # 左右逆転補正（画面左にある場合カメラを左へ振るために符号反転）
                dx = -dx
                
                logging.info(
                    f"🎯 [PID debug] Target: cx={norm_cx:.2f}, cy={norm_cy:.2f} | "
                    f"dt={dt:.3f}s | PID step: dx={dx:+.3f}, dy={dy:+.3f} | "
                    f"PTZ Pos: X={ptz.current_x:.2f}, Y={ptz.current_y:.2f}"
                )
                
                if dx != 0.0 or dy != 0.0:
                    ptz.safe_move(dx, dy)
            else:
                pid.reset()
            
            last_move_time = current_time

        cv2.imshow("Tapo ONVIF YOLOv8 Experiment System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    logging.info("🛑 システム終了処理...")
    cognition_loop.call_soon_threadsafe(cognition_loop.stop)
    cognition_thread.join(timeout=1.0)
    
    try:
        asyncio.run(vlm_client.close())
    except Exception as e:
        logging.warning(f"Failed to close VLM client safely: {e}")
    memory_store.close()

    
    video_reader.release()
    cv2.destroyAllWindows()
    ptz.shutdown()