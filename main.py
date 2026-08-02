import cv2
from dotenv import load_dotenv

from pico.config import AppConfig, build_rtsp_url

# 事前準備で確認・設定した情報
load_dotenv()

config = AppConfig()
TAPO_USER = config.tapo_user
TAPO_PASS = config.tapo_pass
TAPO_IP = config.tapo_ip

# RTSPのURLを設定 
# stream1 = 高画質(2K/1080p) / stream2 = 低画質(360p)
rtsp_url = build_rtsp_url(TAPO_USER, TAPO_PASS, TAPO_IP, "stream1")

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
    cv2.imshow("Pico Argus Monitor", frame)

    # 'q' キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()