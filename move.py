import cv2
from dotenv import load_dotenv
from pico.config import AppConfig, build_rtsp_url
from pico.onvif_client import PTZController

# 環境変数の読み込み
load_dotenv()

# 設定と PTZ コントローラーの初期化
try:
    config = AppConfig()
    ptz = PTZController(
        ip=config.tapo_ip,
        user=config.tapo_user,
        password=config.tapo_pass,
        max_limit_x=config.max_limit_x,
        max_limit_y=config.max_limit_y
    )
    print("✅ ONVIF 接続成功: 共通モジュール経由で制御権を取得しました。")
except Exception as e:
    print(f"❌ 初期化エラー: {e}")
    exit(1)

# 操作用定数
MOVE_STEP = 0.05  # 1回のキー入力で動く量

# RTSPストリームの開始
rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip, "stream1")
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("映像ストリームの開始に失敗しました。")
    ptz.shutdown()
    exit()

print("\n=== 🕹️ 操作方法 ===")
print("  W : 上に動かす / S : 下に動かす")
print("  A : 左に動かす / D : 右に動かす")
print("  Q : プログラム終了")
print("===================\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("映像を取得できませんでした。")
            break

        # ウィンドウに映像を表示
        cv2.imshow("Tapo Camera ONVIF Control Stream", frame)

        # キー入力を1ミリ秒待機
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        # 【上】移動
        elif key == ord('w'):
            actual_x, actual_y = ptz.safe_move(0.0, MOVE_STEP)
            if actual_x != 0.0 or actual_y != 0.0:
                print(f"【上】へ移動 (現在の推測位置 X: {ptz.current_x:.2f}, Y: {ptz.current_y:.2f})")
            else:
                print("❌【警告】これ以上、上には動かせません（リミット制限）")

        # 【下】移動
        elif key == ord('s'):
            actual_x, actual_y = ptz.safe_move(0.0, -MOVE_STEP)
            if actual_x != 0.0 or actual_y != 0.0:
                print(f"【下】へ移動 (現在の推測位置 X: {ptz.current_x:.2f}, Y: {ptz.current_y:.2f})")
            else:
                print("❌【警告】これ以上、下には動かせません（リミット制限）")

        # 【左】移動
        elif key == ord('a'):
            actual_x, actual_y = ptz.safe_move(-MOVE_STEP, 0.0)
            if actual_x != 0.0 or actual_y != 0.0:
                print(f"【左】へ移動 (現在の推測位置 X: {ptz.current_x:.2f}, Y: {ptz.current_y:.2f})")
            else:
                print("❌【警告】これ以上、左には動かせません（リミット制限）")

        # 【右】移動
        elif key == ord('d'):
            actual_x, actual_y = ptz.safe_move(MOVE_STEP, 0.0)
            if actual_x != 0.0 or actual_y != 0.0:
                print(f"【右】へ移動 (現在の推測位置 X: {ptz.current_x:.2f}, Y: {ptz.current_y:.2f})")
            else:
                print("❌【警告】これ以上、右には動かせません（リミット制限）")

finally:
    # 後片付け
    cap.release()
    cv2.destroyAllWindows()
    ptz.shutdown()