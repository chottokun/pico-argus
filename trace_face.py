import os
import urllib.request
import cv2
import time
from dotenv import load_dotenv

from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader
from pico.onvif_client import PTZController
from pico.pid_controller import AdaptivePIDController

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

# 適応型PIDコントローラーの定義
pid = AdaptivePIDController(
    kp_base=0.40,
    ki=0.03,
    kd=0.005,
    dead_zone=0.12,    # 顔追尾は少し緩めに12%の不感帯
    min_speed=0.03,
    max_step=0.12,
    integral_limit=0.2
)

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

TRACK_INTERVAL = 0.40  # 追尾命令を出す最小間隔（秒）
last_move_time = 0.0

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
        cv2.rectangle(
            frame, 
            (center_x - int(width * pid.dead_zone), center_y - int(height * pid.dead_zone)), 
            (center_x + int(width * pid.dead_zone), center_y + int(height * pid.dead_zone)), 
            (0, 255, 255), 1
        )

        # --- 🎯 追尾ロジック ---
        current_time = time.monotonic()
        dt = current_time - last_move_time

        if dt > TRACK_INTERVAL:
            if len(faces) > 0:
                # 複数人が映った場合、一番面積が大きい顔をターゲットにする
                main_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = main_face
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_cx = x + w // 2
                face_cy = y + h // 2
                cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)

                # 正規化座標(0.0〜1.0)
                norm_cx = face_cx / width
                norm_cy = face_cy / height

                # ONVIFは右・上がプラス、画面は右・下がプラスのためY軸の入力を反転
                dx, dy = pid.calculate_step(norm_cx, 1.0 - norm_cy, dt)
                dx = -dx

                if dx != 0.0 or dy != 0.0:
                    actual_x, actual_y = ptz.safe_move(dx, dy)
                    if actual_x != 0.0 or actual_y != 0.0:
                        print(f"🎯 ターゲット追尾: X_step={actual_x:+.3f}, Y_step={actual_y:+.3f} (推測位置 X:{ptz.current_x:.2f}, Y:{ptz.current_y:.2f})")
            else:
                # ターゲットを失ったら積分値をリセットして暴走を防止
                pid.reset()
                
            last_move_time = current_time

        # 一時的に画面内に枠を描画するために、前回の検出矩形を枠として描画する
        for face in faces:
            fx, fy, fw, fh = face
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        cv2.imshow("Tapo ONVIF Face Tracking System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_reader.release()
    cv2.destroyAllWindows()
    ptz.shutdown()