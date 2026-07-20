import os
import cv2
import numpy as np
import time
import urllib.request
import logging
from dotenv import load_dotenv

from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader
from pico.onvif_client import PTZController

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

# --- 🎯 1. トラッキング制御用パラメータ ---
TRACK_TARGET_ID = 0          
CONF_THRESHOLD = 0.45        
TARGET_CONF_THRESHOLD = 0.60 

KP_X = 0.18            
KP_Y = 0.16            
MAX_STEP = 0.12        
TRACK_INTERVAL = 0.45  
last_move_time = 0

# --- 📹 2. RTSP映像ストリームの開始 ---
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

DEAD_ZONE_X = int(width * 0.15)  
DEAD_ZONE_Y = int(height * 0.10) 

x_factor, y_factor = width / 640, height / 640

# --- 🚶 3. YOLOv8-small ONNXモデルの準備 ---
onnx_path = "yolov8s.onnx"
if not os.path.exists(onnx_path):
    logging.info("⏳ 高精度版 YOLOv8-small ONNXモデルをダウンロード中...")
    url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"
    urllib.request.urlretrieve(url, onnx_path)

net = cv2.dnn.readNetFromONNX(onnx_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

logging.info("🚀 上下キビキビ追尾システム稼働！")

try:
    while True:
        ret, frame = video_reader.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        data = outputs[0].T
        class_scores = data[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences_all = np.max(class_scores, axis=1)
        
        valid_indices = np.where(confidences_all > CONF_THRESHOLD)[0]
        boxes, confidences, box_class_ids = [], [], []

        for i in valid_indices:
            row = data[i]
            cx, cy, w, h = row[0:4]
            boxes.append([int((cx - w / 2) * x_factor), int((cy - h / 2) * y_factor), int(w * x_factor), int(h * y_factor)])
            confidences.append(float(confidences_all[i]))
            box_class_ids.append(int(class_ids[i]))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.4)
        if len(indices) > 0:
            indices = np.array(indices).flatten()

        # ガイドライン描画
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 1)
        cv2.rectangle(frame, (center_x - DEAD_ZONE_X, center_y - DEAD_ZONE_Y), (center_x + DEAD_ZONE_X, center_y + DEAD_ZONE_Y), (0, 255, 255), 1)

        track_candidates = []
        detected_objects_summary = []

        for idx in indices:
            x, y, w, h = boxes[idx]
            cid = box_class_ids[idx]
            conf = confidences[idx]
            class_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"ID {cid}"
            
            detected_objects_summary.append(f"{class_name}({conf:.2f})")

            if cid == TRACK_TARGET_ID and conf >= TARGET_CONF_THRESHOLD:
                color = (0, 255, 0)      
                track_candidates.append(idx)
            else:
                color = (0, 165, 255)    
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{class_name}: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        main_track_idx = None
        if len(track_candidates) > 0:
            main_track_idx = max(track_candidates, key=lambda i: boxes[i][2] * boxes[i][3])
            mx, my, mw, mh = boxes[main_track_idx]
            person_cx, person_cy = mx + mw // 2, my + mh // 2
            cv2.circle(frame, (person_cx, person_cy), 6, (0, 0, 255), -1)

        # --- ⏱️ モーター制御 ---
        current_time = time.time()
        if current_time - last_move_time > TRACK_INTERVAL:
            if detected_objects_summary:
                logging.info(f"👁️ 画面内の検知状況: {', '.join(detected_objects_summary)}")
                
            move_x, move_y = 0.0, 0.0
            if main_track_idx is not None:
                mx, my, mw, mh = boxes[main_track_idx]
                person_cx, person_cy = mx + mw // 2, my + mh // 2
                error_x = person_cx - center_x
                error_y = center_y - person_cy

                if abs(error_x) > DEAD_ZONE_X:
                    move_x = -float(error_x / center_x) * KP_X
                    move_x = np.clip(move_x, -MAX_STEP, MAX_STEP)
                    
                if abs(error_y) > DEAD_ZONE_Y:
                    move_y = float(error_y / center_y) * KP_Y
                    move_y = np.clip(move_y, -MAX_STEP, MAX_STEP)

                if move_x != 0.0 or move_y != 0.0:
                    ptz.safe_move(move_x, move_y)
            
            last_move_time = current_time

        cv2.imshow("Tapo ONVIF YOLOv8 Experiment System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    logging.info("🛑 システム終了処理...")
    video_reader.release()
    cv2.destroyAllWindows()
    ptz.shutdown()