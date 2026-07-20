import os
import urllib.request
import cv2
import time
from dotenv import load_dotenv

from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader
from pico.onvif_client import PTZController

# 環境変数の読み込み
load_dotenv()

# --- 🛠️ 1. 設定 & ONVIF/RTSP初期化 ---
try:
    config = AppConfig()
    ptz = PTZController(
        ip=config.tapo_ip,
        user=config.tapo_user,
        password=config.tapo_pass,
        max_limit_x=config.max_limit_x,
        max_limit_y=config.max_limit_y
    )
    print("✅ ONVIF初期化成功。自動追尾の準備が整いました。")
except Exception as e:
    print(f"ONVIF初期化エラー: {e}")
    exit(1)

# --- 👤 2. 顔検出器の準備 ---
xml_path = "haarcascade_frontalface_default.xml"

# もしフォルダ内に設計図ファイルがなければ、OpenCV公式GitHubから自動ダウンロードする
if not os.path.exists(xml_path):
    print("⏳ 顔認識モデルファイル（XML）を公式からダウンロード中...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_path)
    print("✅ ダウンロードが完了しました！")

# ローカルに保存したファイルを読み込む
face_cascade = cv2.CascadeClassifier(xml_path)

# --- 📹 3. 映像ストリームの開始 ---
rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip, "stream1")
video_reader = RTSPVideoReader(rtsp_url)

# 最初のフレームから画面サイズを取得
time.sleep(1.0)
ret, test_frame = video_reader.read()
if not ret or test_frame is None:
    print("映像ストリームの開始に失敗しました。")
    video_reader.release()
    ptz.shutdown()
    exit()

width = int(test_frame.shape[1])
height = int(test_frame.shape[0])
center_x = width // 2
center_y = height // 2

# 💡【重要】デッドゾーン（不感帯）の設定
DEAD_ZONE_X = int(width * 0.15)
DEAD_ZONE_Y = int(height * 0.15)

MOVE_STEP = 0.05
last_move_time = 0
TRACK_INTERVAL = 0.4  # 追尾命令を出す最小間隔（秒）

print("\n🤖 自動顔追尾システムが稼働しました。画面の前に立ってみてください。")
print("※終了するには映像ウィンドウを選択して 'q' キーを押します。\n")

try:
    while True:
        ret, frame = video_reader.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # 処理軽量化のために白黒画像に変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 顔検出の実行
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # --- 📊 画面へのガイドライン描画（デバッグ用） ---
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 1)
        cv2.rectangle(frame, (center_x - DEAD_ZONE_X, center_y - DEAD_ZONE_Y), 
                      (center_x + DEAD_ZONE_X, center_y + DEAD_ZONE_Y), (0, 255, 255), 1)

        # --- 🎯 追尾ロジック ---
        if len(faces) > 0:
            # 複数人が映った場合、一番面積が大きい顔をターゲットにする
            main_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = main_face
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_cx = x + w // 2
            face_cy = y + h // 2
            cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)

            error_x = face_cx - center_x
            error_y = center_y - face_cy  # ONVIFは上がプラス、画面座標は下がプラスのため反転

            current_time = time.time()

            # 🔒 モーター保護インターバルチェック
            if current_time - last_move_time > TRACK_INTERVAL:
                move_x, move_y = 0.0, 0.0

                # 左右の追尾判定
                if abs(error_x) > DEAD_ZONE_X:
                    move_x = MOVE_STEP if error_x > 0 else -MOVE_STEP

                # 上下の追尾判定
                if abs(error_y) > DEAD_ZONE_Y:
                    move_y = MOVE_STEP if error_y > 0 else -MOVE_STEP

                if move_x != 0.0 or move_y != 0.0:
                    actual_x, actual_y = ptz.safe_move(move_x, move_y)
                    last_move_time = current_time
                    if actual_x != 0.0 or actual_y != 0.0:
                        print(f"🎯 ターゲット追尾: X_step={actual_x:+.3f}, Y_step={actual_y:+.3f} (推測位置 X:{ptz.current_x:.2f}, Y:{ptz.current_y:.2f})")

        cv2.imshow("Tapo ONVIF Face Tracking System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_reader.release()
    cv2.destroyAllWindows()
    ptz.shutdown()