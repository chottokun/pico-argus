import cv2
import os
from dotenv import load_dotenv

# 事前準備で確認・設定した情報
load_dotenv()

TAPO_USER = os.getenv("TAPO_USER")
TAPO_PASS = os.getenv("TAPO_PASS")
TAPO_IP = os.getenv("TAPO_IP")

# RTSPのURLを設定 
# stream1 = 高画質(2K/1080p) / stream2 = 低画質(360p)
rtsp_url = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream1"

# ビデオキャプチャの開始
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("カメラへの接続に失敗しました。URLやIP、パスワードを確認してください。")
    exit()

print("接続成功！'q' キーを押すと終了します。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("映像を取得できませんでした。")
        break

    # ウィンドウに映像を表示
    cv2.imshow("Tapo Camera Live Stream", frame)

    # 'q' キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()