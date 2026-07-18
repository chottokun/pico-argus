import os
import cv2
import threading
from dotenv import load_dotenv
from onvif import ONVIFCamera  # pytapo から ONVIF に変更

# 環境変数の読み込み
load_dotenv()

# ONVIF経由の場合、アプリのEMAIL/PASSWORDは【完全に不要】です！
TAPO_USER = os.getenv("TAPO_USER")  # アプリの高度な設定で作ったユーザー名
TAPO_PASS = os.getenv("TAPO_PASS")  # アプリの高度な設定で作ったパスワード
TAPO_IP = os.getenv("TAPO_IP")

if not all([TAPO_IP, TAPO_USER, TAPO_PASS]):
    print("エラー: .env の TAPO_IP, TAPO_USER, TAPO_PASS が設定されていません。")
    exit(1)


def init_onvif_client():
    """ONVIF規格を使ってカメラの制御権を取得する"""
    print(f"ONVIF接続試行: {TAPO_IP}:2020 (user={TAPO_USER})")
    try:
        # TapoカメラのONVIFポートは「2020」で固定です
        mycam = ONVIFCamera(TAPO_IP, 2020, TAPO_USER, TAPO_PASS)
        
        # 必要な制御サービスを立ち上げる
        media = mycam.create_media_service()
        ptz = mycam.create_ptz_service()
        
        # カメラの制御用トークン（識別ID）を取得
        media_profile = media.GetProfiles()[0]
        profile_token = media_profile.token
        
        return ptz, profile_token
    except Exception as e:
        raise RuntimeError(f"ONVIF接続エラー: {e}\n※カメラアカウントのパスワードが正しいか、30分ロックが明けているか確認してください。")


# ONVIFの初期化
try:
    ptz, token = init_onvif_client()
    print("✅ ONVIF認証成功: 業界標準規格での制御権を取得しました。")
except Exception as e:
    print(str(e))
    exit(1)

# --- 🔒 安全対策：ソフトウェアリミットの設定（ONVIF仕様） ---
# ONVIFでは位置や移動量を -1.0 〜 1.0 の小数の範囲で制御します
current_x = 0.0
current_y = 0.0

LIMIT_X = 1.0     # 左右の最大リミット
LIMIT_Y = 0.5     # 上下の最大リミット
MOVE_STEP = 0.05  # 1回のキー入力で動く量（小数を小さくするとゆっくり動きます）

def async_move(x, y):
    """映像（メインスレッド）をカクつかせないために別スレッドで通信"""
    def _target():
        try:
            # ONVIFの「相対移動（RelativeMove）」命令を作成
            request = ptz.create_type('RelativeMove')
            request.ProfileToken = token
            request.Translation = {
                'PanTilt': {
                    'x': x,
                    'y': y
                }
            }
            ptz.RelativeMove(request)
        except Exception as e:
            print(f"モーター駆動エラー: {e}")
            
    threading.Thread(target=_target, daemon=True).start()


# RTSPストリームの開始
rtsp_url = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("映像ストリームの開始に失敗しました。")
    exit()

print("\n=== 🕹️ 操作方法 ===")
print("  W : 上に動かす / S : 下に動かす")
print("  A : 左に動かす / D : 右に動かす")
print("  Q : プログラム終了")
print("===================\n")

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
        if current_y < LIMIT_Y:
            current_y += MOVE_STEP
            async_move(0, MOVE_STEP)
            print(f"【上】へ移動 (現在の推測位置 Y: {current_y:.2f})")
        else:
            print("❌【警告】これ以上、上には動かせません（リミット制限）")

    # 【下】移動
    elif key == ord('s'):
        if current_y > -LIMIT_Y:
            current_y -= MOVE_STEP
            async_move(0, -MOVE_STEP)
            print(f"【下】へ移動 (現在の推測位置 Y: {current_y:.2f})")
        else:
            print("❌【警告】これ以上、下には動かせません（リミット制限）")

    # 【左】移動
    elif key == ord('a'):
        if current_x > -LIMIT_X:
            current_x -= MOVE_STEP
            # 💡 もしキーの操作と実際の画面の左右が逆に見える場合は、符号を反転（MOVE_STEP）にしてください
            async_move(-MOVE_STEP, 0)
            print(f"【左】へ移動 (現在の推測位置 X: {current_x:.2f})")
        else:
            print("❌【警告】これ以上、左には動かせません（リミット制限）")

    # 【右】移動
    elif key == ord('d'):
        if current_x < LIMIT_X:
            current_x += MOVE_STEP
            # 💡 もしキーの操作と実際の画面の左右が逆に見える場合は、符号を反転（-MOVE_STEP）にしてください
            async_move(MOVE_STEP, 0)
            print(f"【右】へ移動 (現在の推測位置 X: {current_x:.2f})")
        else:
            print("❌【警告】これ以上、右には動かせません（リミット制限）")

# 後片付け
cap.release()
cv2.destroyAllWindows()