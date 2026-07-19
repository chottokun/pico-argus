import os
import cv2
import numpy as np
import threading
import time
import urllib.request
import queue
import logging
import json
from dotenv import load_dotenv
from onvif import ONVIFCamera

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
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

load_dotenv()
TAPO_USER = os.getenv("TAPO_USER")
TAPO_PASS = os.getenv("TAPO_PASS")
TAPO_IP = os.getenv("TAPO_IP")

# --- 🛠️ 1. ONVIFの初期化 & PTZプロファイルの自動探索 ---
try:
    mycam = ONVIFCamera(TAPO_IP, 2020, TAPO_USER, TAPO_PASS)
    ptz = mycam.create_ptz_service()
    media = mycam.create_media_service()
    profiles = media.GetProfiles()
    profile_token = next((p.token for p in profiles if hasattr(p, 'PTZConfiguration') and p.PTZConfiguration is not None), profiles[0].token)
    logging.info("✅ ONVIF初期化成功。")
except Exception as e:
    logging.error(f"ONVIF初期化エラー: {e}"); exit(1)

# --- 🔒 2. 安全対策・スレッド競合防止（キューシステム） ---
move_queue = queue.Queue(maxsize=1)
def ptz_worker():
    while True:
        command = move_queue.get()
        if command is None: break
        x, y = command
        try:
            request = ptz.create_type('RelativeMove')
            request.ProfileToken = profile_token
            request.Translation = {'PanTilt': {'x': x, 'y': y}}
            ptz.RelativeMove(request)
            time.sleep(0.15)
        except Exception as e:
            logging.error(f"モーター駆動エラー: {e}")
        move_queue.task_done()

threading.Thread(target=ptz_worker, daemon=True).start()

def send_move_command(x, y):
    if move_queue.full():
        try: move_queue.get_nowait()
        except queue.Empty: pass
    move_queue.put((x, y))

# --- 🎯 3. トラッキング制御用パラメータ ＆ JSON読み込み ---
TRACK_TARGET_ID = 0          
CONF_THRESHOLD = 0.45        
TARGET_CONF_THRESHOLD = 0.60 # 人間の合格ライン（75%以上）

# 💡【大幅改善】上下の追尾感度とリミッターを解放
KP_X = 0.18            
KP_Y = 0.16            # 🛠️ 0.08 から 0.16 に倍増（左右と同等のキレにする）
MAX_STEP = 0.12        # 🛠️ 1回の最大制限を 0.07 から 0.12 に引き上げ
TRACK_INTERVAL = 0.45  # ⚙️ 命令間隔を 0.55秒 から 0.45秒 に少し縮めてレスポンス向上
last_move_time = 0

# 可動限界のロード
current_internal_x = 0.0
current_internal_y = 0.0
MAX_LIMIT_X = 1.2      
MAX_LIMIT_Y = 0.4      

config_path = "tapo_config.json"
if os.path.exists(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            MAX_LIMIT_X = config_data.get("MAX_LIMIT_X", MAX_LIMIT_X)
            MAX_LIMIT_Y = config_data.get("MAX_LIMIT_Y", MAX_LIMIT_Y)
            logging.info(f"💾 キャリブレーション限界ロード ➔ X: ±{MAX_LIMIT_X}, Y: ±{MAX_LIMIT_Y}")
    except Exception as e:
        logging.error(f"⚠️ JSON読込失敗: {e}")

def send_safe_move_command(requested_x, requested_y):
    global current_internal_x, current_internal_y
    next_x = current_internal_x + requested_x
    next_y = current_internal_y + requested_y
    actual_move_x, actual_move_y = requested_x, requested_y

    if next_x > MAX_LIMIT_X: actual_move_x = MAX_LIMIT_X - current_internal_x
    elif next_x < -MAX_LIMIT_X: actual_move_x = -MAX_LIMIT_X - current_internal_x
    if next_y > MAX_LIMIT_Y: actual_move_y = MAX_LIMIT_Y - current_internal_y
    elif next_y < -MAX_LIMIT_Y: actual_move_y = -MAX_LIMIT_Y - current_internal_y

    if abs(actual_move_x) > 0.001 or abs(actual_move_y) > 0.001:
        send_move_command(actual_move_x, actual_move_y)
        current_internal_x += actual_move_x
        current_internal_y += actual_move_y
        logging.info(f"🎯 [追尾送信] 首振り量: X={actual_move_x:+.3f}, Y={actual_move_y:+.3f} | 推測位置: X={current_internal_x:+.2f}, Y={current_internal_y:+.2f}")

# --- 📹 4. RTSPバッファ自動消滅スレッド ---
class RTSPVideoReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._keep_reading, daemon=True)
        self.thread.start()

    def _keep_reading(self):
        while self.running:
            if not self.cap.isOpened(): break
            try:
                ret, frame = self.cap.read()
                if ret: self.frame, self.ret = frame, ret
                else: time.sleep(0.01)
            except Exception: break

    def read(self): return self.ret, self.frame
    def release(self):
        self.running = False
        time.sleep(0.2)
        if self.cap.isOpened(): self.cap.release()

rtsp_url = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream1"
video_reader = RTSPVideoReader(rtsp_url)
time.sleep(1.0)

# --- 🚶 5. YOLOv8-small ONNXモデルの準備 ---
onnx_path = "yolov8s.onnx"
if not os.path.exists(onnx_path):
    logging.info("⏳ 高精度版 YOLOv8-small ONNXモデルをダウンロード中...")
    url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"
    urllib.request.urlretrieve(url, onnx_path)

net = cv2.dnn.readNetFromONNX(onnx_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

ret, test_frame = video_reader.read()
if not ret or test_frame is None: logging.error("映像ストリーム受信失敗"); exit()

width, height = int(test_frame.shape[1]), int(test_frame.shape[0])
center_x, center_y = width // 2, height // 2

# 💡【大幅改善】不感帯（黄色い枠）を狭くして、わずかな上下移動でも即反応させる
DEAD_ZONE_X = int(width * 0.15)  # 20% から 15% に縮小
DEAD_ZONE_Y = int(height * 0.10) # 🛠️ 20% から 10% に大幅縮小（上下のサボりを解消！）

x_factor, y_factor = width / 640, height / 640

logging.info(f"🚀 上下キビキビ追尾システム稼働！")

while True:
    ret, frame = video_reader.read()
    if not ret or frame is None:
        time.sleep(0.01); continue

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
    if len(indices) > 0: indices = np.array(indices).flatten()

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
                
            # 💡 上下の不感帯を抜けた時の計算
            if abs(error_y) > DEAD_ZONE_Y:
                move_y = float(error_y / center_y) * KP_Y
                move_y = np.clip(move_y, -MAX_STEP, MAX_STEP)

            if move_x != 0.0 or move_y != 0.0:
                send_safe_move_command(move_x, move_y)
        
        last_move_time = current_time

    cv2.imshow("Tapo ONVIF YOLOv8 Experiment System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

logging.info("🛑 システム終了処理...")
move_queue.put(None)
video_reader.release()
cv2.destroyAllWindows()