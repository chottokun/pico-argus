import os
import cv2
import time
import logging
import asyncio
import threading
import urllib.request
from dotenv import load_dotenv

from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader
from pico.onvif_client import PTZController
from pico.pid_controller import AdaptivePIDController
from pico.detector import YoloDetector
from pico.tracker import SimpleIoUTracker
from pico.perception_buffer import PerceptionBuffer
from pico.ollama_client import OllamaVisionClient
from pico.memory import MemoryStore
from pico.agent_tools import AgentTools
from pico.agent import SurveillanceAgent
from pico.guardrails import GuardRails, FrameStatus

# --- 📁 0. ログシステムの設定 ---
log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("tapo_agent_v2.log", encoding="utf-8"),
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
    
    # 映像ストリームを開始
    rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip, "stream1")
    video_reader = RTSPVideoReader(rtsp_url)
    time.sleep(1.5)  # 受信開始まで待機
    
    ptz = PTZController(
        ip=config.tapo_ip,
        user=config.tapo_user,
        password=config.tapo_pass,
        max_limit_x=config.max_limit_x,
        max_limit_y=config.max_limit_y,
        align_to_home=config.align_to_home,
        video_reader=video_reader,
        invert_pan=config.invert_pan,
        invert_tilt=config.invert_tilt
    )
    logging.info("✅ [System Initialization] ONVIF/PTZ client started.")
except Exception as e:
    logging.error(f"❌ [System Initialization Failed] ONVIF startup error: {e}")
    raise SystemExit(1)

# --- 🎯 1. 制御パラメータ & モジュールの初期化 ---
TRACK_TARGET_ID = 0  # 'person'
CONF_THRESHOLD = 0.45
TARGET_CONF_THRESHOLD = 0.60

pid = AdaptivePIDController(
    kp_base=0.35,
    ki=0.03,
    kd=0.005,
    dead_zone=0.10,
    min_speed=0.03,
    max_step=0.12,
    integral_limit=0.2
)

# YOLOモデル
onnx_path = "yolov8s.onnx"
if not os.path.exists(onnx_path):
    logging.info("⏳ YOLOv8-small ONNXモデルをダウンロード中...")
    url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"
    urllib.request.urlretrieve(url, onnx_path)

detector = YoloDetector(model_path=onnx_path, conf_threshold=CONF_THRESHOLD)
tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=30)
guard = GuardRails(timeout_limit=3.0, max_area_ratio=0.75)
perception_buffer = PerceptionBuffer()

# 長期記憶・VLMクライアント
memory_store = MemoryStore(db_path="wiki.db")
memory_store.add_entry(
    filepath="rules/general.md",
    doc_type="rule",
    title="エッジ状況分析基本指示",
    tags="status person action",
    content="人(person)やオブジェクトを検知した場合、その状態や行動・状況を客観的に観察し分析すること。gemma4:e2bは最優先で使用すること。"
)
vlm_client = OllamaVisionClient(base_url=config.ollama_base_url, model=config.ollama_model)

# 司令塔エージェントの準備
agent_tools = AgentTools(ptz_controller=ptz, vlm_client=vlm_client, memory_store=memory_store)
agent = SurveillanceAgent(tools=agent_tools, perception_buffer=perception_buffer, ollama_client=vlm_client)

# --- 🧠 2. 非同期エージェントループのバックグラウンド起動 ---
agent_loop = asyncio.new_event_loop()

def start_agent_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

agent_thread = threading.Thread(
    target=start_agent_thread,
    args=(agent_loop,),
    daemon=True
)
agent_thread.start()

# --- 📹 3. RTSP映像ストリーム確認 ---
ret, test_frame = video_reader.read()
if not ret or test_frame is None:
    logging.error("映像ストリーム受信失敗")
    video_reader.release()
    ptz.shutdown()
    raise SystemExit(1)

width, height = int(test_frame.shape[1]), int(test_frame.shape[0])
center_x, center_y = width // 2, height // 2

logging.info("🚀 V2 LLMエージェント統括型 能動知覚追跡システム稼働開始！")

TRACK_INTERVAL = 0.45
last_move_time = 0.0
last_agent_trigger_time = 0.0

try:
    while True:
        # 🔒 安全ガードレール: フレーム時間監視
        last_frame_t = video_reader.get_last_frame_time()
        frame_health = guard.check_frame_health(last_frame_t)
        
        if frame_health == FrameStatus.TIMEOUT:
            logging.warning("⚠️ [Safety Guard] RTSP feed lost. Pausing control...")
            pid.reset()
            time.sleep(0.5)
            continue

        ret, frame = video_reader.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # 🔒 安全ガードレール: 夜間自動検知
        is_night = guard.check_night_mode(frame)
        pid.dead_zone = 0.05 if is_night else 0.10

        # 物体検出 & 追跡
        detections = detector.detect(frame)
        sane_detections = [d for d in detections if guard.is_bbox_sane(d.bbox, frame.shape)]
        tracked_objects = tracker.update(sane_detections)

        # 知覚バッファの更新
        perception_buffer.update(tracked_objects, frame.shape)

        # 映像アノテーション & 描画
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 1)
        
        # 不感帯描画
        cv2.rectangle(
            frame, 
            (center_x - int(width * pid.dead_zone), center_y - int(height * pid.dead_zone)), 
            (center_x + int(width * pid.dead_zone), center_y + int(height * pid.dead_zone)), 
            (0, 255, 255), 1
        )

        info_y = 30
        if is_night:
            cv2.putText(frame, "NIGHT MODE (DeadZone: 5%)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
        if ptz.lock_on_id is not None:
            cv2.putText(frame, f"LOCKED TARGET: ID {ptz.lock_on_id}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        track_candidates = []
        detected_objects_summary = []
        for obj in tracked_objects:
            x, y, w, h = obj.bbox
            cid = obj.class_id
            conf = obj.confidence
            track_id = obj.track_id
            class_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"ID {cid}"
            
            detected_objects_summary.append(f"{class_name}#{track_id}({conf:.2f})")

            # 🎯 ロックオン対象IDは最低保証しきい値 CONF_THRESHOLD (0.45) で維持、それ以外は TARGET_CONF_THRESHOLD (0.60)
            is_locked = (ptz.lock_on_id == track_id)
            required_conf = CONF_THRESHOLD if is_locked else TARGET_CONF_THRESHOLD

            if cid == TRACK_TARGET_ID and conf >= required_conf:
                track_candidates.append(obj)
                if is_locked:
                    color = (255, 0, 255)  # ロックターゲットはピンク
                    label = f"{class_name}#{track_id}: {conf:.2f} [LOCKED]"
                else:
                    color = (0, 255, 0)    # 通常追尾はグリーン
                    label = f"{class_name}#{track_id}: {conf:.2f}"
            else:
                color = (0, 165, 255)
                label = f"{class_name}#{track_id}: {conf:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 🧠 司令塔エージェントへのプラン実行要求（非同期トリガー）
        current_time = time.monotonic()
        # 10秒に1回、かつエージェントが現在思考中でない場合のみ、バックグラウンドで起動して自律プランを検討させる
        global agent_is_thinking
        if 'agent_is_thinking' not in globals():
            agent_is_thinking = False

        if len(track_candidates) > 0 and (current_time - last_agent_trigger_time > 10.0) and not agent_is_thinking:
            agent_is_thinking = True
            last_agent_trigger_time = current_time
            # 知覚バッファ情報を引き抜く
            active_tracks_snapshot = perception_buffer.get_active_tracks_json()
            frame_snapshot = frame.copy()
            
            # エージェントステップをスレッドプールの asyncio で非同期起動
            future = asyncio.run_coroutine_threadsafe(
                agent.step(active_tracks_snapshot, frame_snapshot),
                agent_loop
            )
            
            def agent_done_callback(fut):
                global agent_is_thinking
                agent_is_thinking = False
                try:
                    res = fut.result()
                    logging.info(f"🧠 [Agent Finished] Planning Step Result: {res}")
                except Exception as ex:
                    logging.error(f"❌ [Agent Execution Error] Exception in planning thread: {ex}", exc_info=True)

            future.add_done_callback(agent_done_callback)
            logging.info("🧠 [Agent Dispatch] Dispatched situation metadata to LLM Planner.")

        # 優先ターゲットの選定（ロックオン優先）
        main_track_obj = None
        if len(track_candidates) > 0:
            if ptz.lock_on_id is not None:
                lock_on_obj = next((o for o in track_candidates if o.track_id == ptz.lock_on_id), None)
                if lock_on_obj is not None:
                    main_track_obj = lock_on_obj
                    mx, my, mw, mh = main_track_obj.bbox
                    person_cx, person_cy = mx + mw // 2, my + mh // 2
                    cv2.circle(frame, (person_cx, person_cy), 8, (255, 0, 255), -1)
                else:
                    logging.info(f"🎯 [LOCK-ON LOST] Target ID {ptz.lock_on_id} lost. Reverting to default.")
                    ptz.lock_on_id = None

            if main_track_obj is None:
                main_track_obj = max(track_candidates, key=lambda o: o.bbox[2] * o.bbox[3])
                mx, my, mw, mh = main_track_obj.bbox
                person_cx, person_cy = mx + mw // 2, my + mh // 2
                cv2.circle(frame, (person_cx, person_cy), 6, (0, 0, 255), -1)

        # --- ⏱️ モーター制御（反射ループ） ---
        dt = current_time - last_move_time

        if dt > TRACK_INTERVAL:
            if detected_objects_summary:
                logging.info(f"👁️ 画面内の追跡状況: {', '.join(detected_objects_summary)}")
                
            if main_track_obj is not None:
                mx, my, mw, mh = main_track_obj.bbox
                person_cx, person_cy = mx + mw // 2, my + mh // 2
                norm_cx = person_cx / width
                norm_cy = person_cy / height
                
                dx, dy = pid.calculate_step(norm_cx, 1.0 - norm_cy, dt)
                if dx != 0.0 or dy != 0.0:
                    ptz.safe_move(dx, dy)
            else:
                pid.reset()
            last_move_time = current_time

        cv2.imshow("Tapo ONVIF V2 Agent Surveillance System", frame)
        
        # キー操作割り込み
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # 1〜9のキー入力で、特定のIDを緊急上書きロックオン（Barge-Inのシミュレーション）
        elif ord("1") <= key <= ord("9"):
            forced_id = key - ord("0")
            # 非同期で Barge-In メソッドを実行
            asyncio.run_coroutine_threadsafe(
                agent.update_by_user_barge_in(forced_id),
                agent_loop
            )

finally:
    logging.info("🛑 システム終了処理中...")
    agent_loop.call_soon_threadsafe(agent_loop.stop)
    
    try:
        asyncio.run(vlm_client.close())
    except Exception as e:
        logging.warning(f"Failed to close VLM client safely: {e}")
    memory_store.close()
    
    video_reader.release()
    cv2.destroyAllWindows()
    ptz.shutdown()
