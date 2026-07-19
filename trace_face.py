import os
import urllib.request
import cv2

import threading
import time
from dotenv import load_dotenv
from onvif import ONVIFCamera

from camera_config import build_rtsp_url, require_env

# 環境変数の読み込み
load_dotenv()
TAPO_USER = require_env("TAPO_USER")
TAPO_PASS = require_env("TAPO_PASS")
TAPO_IP = require_env("TAPO_IP")

# --- 🛠️ 1. ONVIFの初期化 ---
try:
    mycam = ONVIFCamera(TAPO_IP, 2020, TAPO_USER, TAPO_PASS)
    ptz = mycam.create_ptz_service()
    media = mycam.create_media_service()
    profile_token = media.GetProfiles()[0].token
    print("✅ ONVIF初期化成功。自動追尾の準備が整いました。")
except Exception as e:
    print(f"ONVIF初期化エラー: {e}")
    exit(1)

# --- 🔒 2. 安全対策・モーター保護の設定 ---
current_x, current_y = 0.0, 0.0
LIMIT_X = 1.0       # 左右の最大リミット
LIMIT_Y = 0.5       # 上下の最大リミット
MOVE_STEP = 0.05    # 追尾時の1歩の大きさ（速度調整）

last_move_time = 0
TRACK_INTERVAL = 0.4  # 💡【重要】追尾命令を出す間隔（秒）。短すぎるとモーターが過熱します

def async_move(x, y):
    """映像をカクつかせないための別スレッド駆動"""
    def _target():
        try:
            request = ptz.create_type('RelativeMove')
            request.ProfileToken = profile_token
            request.Translation = {'PanTilt': {'x': x, 'y': y}}
            ptz.RelativeMove(request)
        except Exception as e:
            print(f"モーター駆動エラー: {e}")
    threading.Thread(target=_target, daemon=True).start()

# --- 👤 3. 顔検出器の準備 ---
# OpenCVに標準内蔵されている軽量な顔検出ファイルを使用します

xml_path = "haarcascade_frontalface_default.xml"

# もしフォルダ内に設計図ファイルがなければ、OpenCV公式GitHubから自動ダウンロードする
if not os.path.exists(xml_path):
    print("⏳ 顔認識モデルファイル（XML）を公式からダウンロード中...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_path)
    print("✅ ダウンロードが完了しました！")

# ローカルに保存したファイルを読み込む
face_cascade = cv2.CascadeClassifier(xml_path)

# --- 📹 4. 映像ストリームの開始 ---
rtsp_url = build_rtsp_url(TAPO_USER, TAPO_PASS, TAPO_IP, "stream1")
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("映像ストリームの開始に失敗しました。")
    exit()

# 画面サイズと中心座標の計算
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = width // 2
center_y = height // 2

# 💡【重要】デッドゾーン（不感帯）の設定
# 画面の中心から「画面サイズの15%」の範囲内なら、多少顔が動いてもカメラを動かさない（ハンチング防止）
DEAD_ZONE_X = int(width * 0.15)
DEAD_ZONE_Y = int(height * 0.15)

print("\n🤖 自動顔追尾システムが稼働しました。画面の前に立ってみてください。")
print("※終了するには映像ウィンドウを選択して 'q' キーを押します。\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("映像を取得できませんでした。")
        break

    # 処理軽量化のために白黒画像に変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 顔検出の実行
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # --- 📊 画面へのガイドライン描画（デバッグ用） ---
    # 画面の中心十字（青）
    cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
    cv2.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 1)
    # デッドゾーンの枠（黄）
    cv2.rectangle(frame, (center_x - DEAD_ZONE_X, center_y - DEAD_ZONE_Y), 
                  (center_x + DEAD_ZONE_X, center_y + DEAD_ZONE_Y), (0, 255, 255), 1)

    # --- 🎯 追尾ロジック ---
    if len(faces) > 0:
        # 💡 複数人が映った場合、一番面積が大きい顔（画面の最も近くにいる人）をメインターゲットにする
        main_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = main_face
        
        # 検出した顔を緑枠で囲む
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 顔の中心座標を計算（赤点）
        face_cx = x + w // 2
        face_cy = y + h // 2
        cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)

        # 画面中心からの「ズレ（偏差）」を計算
        error_x = face_cx - center_x
        error_y = center_y - face_cy  # ONVIFは上がプラス、画面座標は下がプラスのため反転

        current_time = time.time()

        # 🔒 ハードウェア保護：前回の移動から 0.4 秒以上経過している場合のみ動かす
        if current_time - last_move_time > TRACK_INTERVAL:
            move_x, move_y = 0.0, 0.0

            # 左右の追尾判定（ズレがデッドゾーンを超えていたら）
            if abs(error_x) > DEAD_ZONE_X:
                if error_x > 0 and current_x < LIMIT_X:      # 顔が右にある ➔ カメラを右へ
                    move_x = MOVE_STEP
                    current_x += MOVE_STEP
                elif error_x < 0 and current_x > -LIMIT_X:   # 顔が左にある ➔ カメラを左へ
                    move_x = -MOVE_STEP
                    current_x -= MOVE_STEP

            # 上下の追尾判定
            if abs(error_y) > DEAD_ZONE_Y:
                if error_y > 0 and current_y < LIMIT_Y:      # 顔が上にある ➔ カメラを上へ
                    move_y = MOVE_STEP
                    current_y += MOVE_STEP
                elif error_y < 0 and current_y > -LIMIT_Y:   # 顔が下にある ➔ カメラを下へ
                    move_y = -MOVE_STEP
                    current_y -= MOVE_STEP

            # 実際に動かす命令を送信
            if move_x != 0.0 or move_y != 0.0:
                async_move(move_x, move_y)
                last_move_time = current_time
                print(f"🎯 ターゲット追尾: X_step={move_x}, Y_step={move_y} (推測位置 X:{current_x:.2f}, Y:{current_y:.2f})")

    # 映像を表示
    cv2.imshow("Tapo ONVIF Face Tracking System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()