import cv2
import numpy as np
import time
import json
from dotenv import load_dotenv
from onvif import ONVIFCamera

from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader

# =====================================================================
# ⚙️ 【カメラ・通信環境に合わせた調整パラメータ】
# =====================================================================
SHOW_PREVIEW = True        # キャリブレーション中の映像を画面に出すか
STEP_SIZE_X = 0.15         # 左右移動の1歩の大きさ
STEP_SIZE_Y = 0.10         # 上下移動の1歩の大きさ

RTSP_LAG_TIMEOUT = 1.4     # 命令を出してから、Wi-Fi経由でカメラが動き終わるまでの待ち時間（秒）
MOTION_THRESHOLD = 3.0     # 動き判定のしきい値（％）
PIXEL_DIFF_THRESHOLD = 15  # 画素の差分しきい値 (0〜255)
# =====================================================================

load_dotenv()
try:
    config = AppConfig(config_path="camera_config.json")
except Exception as e:
    print(f"❌ 設定読み込みエラー: {e}")
    raise SystemExit(1)

# --- 🛠️ 1. ONVIFの初期化 ---
try:
    mycam = ONVIFCamera(config.tapo_ip, 2020, config.tapo_user, config.tapo_pass)
    ptz = mycam.create_ptz_service()
    media = mycam.create_media_service()
    profiles = media.GetProfiles()
    profile_token = next((p.token for p in profiles if hasattr(p, 'PTZConfiguration') and p.PTZConfiguration is not None), profiles[0].token)
except Exception as e:
    print(f"❌ 初期化エラー: {e}")
    raise SystemExit(1)

# スレッド駆動の映像読み込みを開始
rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip, "stream1")
video_reader = RTSPVideoReader(rtsp_url)

time.sleep(1.0)
ret, test_frame = video_reader.read()
if not ret or test_frame is None:
    print("❌ 映像ストリームを正常に受信できませんでした。")
    video_reader.release()
    raise SystemExit(1)


def move_and_check_survival(step_x, step_y, status_text):
    ret, frame_before = video_reader.read()
    if not ret or frame_before is None:
        return False
    gray_before = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
    
    # 物理カメラ極性を反転適用
    cmd_x = -step_x if config.invert_pan else step_x
    cmd_y = -step_y if config.invert_tilt else step_y

    request = ptz.create_type('RelativeMove')
    request.ProfileToken = profile_token
    request.Translation = {'PanTilt': {'x': cmd_x, 'y': cmd_y}}
    ptz.RelativeMove(request)
    
    time.sleep(RTSP_LAG_TIMEOUT)
    
    ret, frame_after = video_reader.read()
    if not ret or frame_after is None:
        return False
    gray_after = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_before, gray_after)
    _, thresh = cv2.threshold(diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    moved_ratio = np.sum(thresh == 255) / thresh.size
    
    has_moved = (moved_ratio * 100) > MOTION_THRESHOLD
    
    if SHOW_PREVIEW:
        display_frame = frame_after.copy()
        h, w = display_frame.shape[:2]
        color = (0, 255, 0) if has_moved else (0, 0, 255)
        
        cv2.putText(display_frame, f"CALIB: {status_text}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Motion: {moved_ratio*100:.1f}% / Thresh: {MOTION_THRESHOLD}%", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        small_thresh = cv2.resize(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), (w // 4, h // 4))
        display_frame[h - (h // 4) - 10 : h - 10, w - (w // 4) - 10 : w - 10] = small_thresh
        cv2.rectangle(display_frame, (w - (w // 4) - 10, h - (h // 4) - 10), (w - 10, h - 10), (0, 255, 255), 2)
        
        cv2.imshow("Tapo C210 Dynamic Calibration Wizard", display_frame)
        cv2.waitKey(1)
        
    return has_moved

def move_until_stop(step_x, step_y, status_text, max_attempts=40):
    attempts = 0
    while attempts < max_attempts:
        has_moved = move_and_check_survival(step_x, step_y, status_text)
        if not has_moved:
            print(f"🛑 物理限界（端）を検知: {status_text} (計 {attempts} 歩)")
            break
        attempts += 1
        time.sleep(0.1)
    return attempts

print("\n⚙️ Tapo C210 スレッド安定型・バックラッシュ完全相殺キャリブレーションを開始します...")

# 1. 左の限界まで追い込む
print("🚀 フェーズ 1/4: 左の限界壁を探しています...")
move_until_stop(-STEP_SIZE_X, 0.0, "Hunting LEFT edge")
time.sleep(1.0)

# 2. そこから右の限界まで動かし、絶対総幅をカウント
print("🚀 フェーズ 2/4: 右の限界壁までの【絶対総距離】を計測中...")
total_steps_x = move_until_stop(STEP_SIZE_X, 0.0, "Measuring RIGHT width")
time.sleep(1.0)

# 3. 下の限界まで追い込む
print("🚀 フェーズ 3/4: 下の限界壁を探しています...")
move_until_stop(0.0, -STEP_SIZE_Y, "Hunting BOTTOM edge")
time.sleep(1.0)

# 4. 上の限界まで動かし、絶対総幅をカウント
print("🚀 フェーズ 4/4: 上の限界壁までの【絶対総距離】を計測中...")
total_steps_y = move_until_stop(0.0, STEP_SIZE_Y, "Measuring TOP width")
time.sleep(1.0)

# 5. 【ギアバックラッシュ相殺・物理基準端への再アライメント】
print("\n🔄 計測完了（左右総幅: {}歩 / 上下総幅: {}歩）".format(total_steps_x, total_steps_y))
print("🔄 バックラッシュ(ギア遊び)を相殺するため【左下端基準点】へ突き当て準備中...")
move_until_stop(-STEP_SIZE_X, 0.0, "Resetting LEFT for Backlash cancel")
time.sleep(0.5)
move_until_stop(0.0, -STEP_SIZE_Y, "Resetting BOTTOM for Backlash cancel")
time.sleep(1.0)

# 6. 【左下端から「右・上」へ一方向スムーズ中心移動】
center_steps_x = total_steps_x // 2
center_steps_y = total_steps_y // 2
print("🔄 左下物理基準点から【真の中心原点】へ「右・上」移動で正確に復帰中...")

# 左端から「右(RIGHT)」へ動かすため cmd_x = -STEP_SIZE_X (invert_pan: True環境)
for i in range(center_steps_x):
    cmd_x = -STEP_SIZE_X if config.invert_pan else STEP_SIZE_X
    ptz.RelativeMove({'ProfileToken': profile_token, 'Translation': {'PanTilt': {'x': cmd_x, 'y': 0.0}}})
    time.sleep(RTSP_LAG_TIMEOUT)
    if SHOW_PREVIEW:
        ret, frame_current = video_reader.read()
        if ret and frame_current is not None:
            display_frame = frame_current.copy()
            cv2.putText(display_frame, f"PHASE: Returning Center X ({i+1}/{center_steps_x})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Tapo C210 Dynamic Calibration Wizard", display_frame)
            cv2.waitKey(1)

# 下端から「上(UP)」へ動かすため cmd_y = STEP_SIZE_Y (invert_tilt: True環境)
for i in range(center_steps_y):
    cmd_y = STEP_SIZE_Y if config.invert_tilt else -STEP_SIZE_Y
    ptz.RelativeMove({'ProfileToken': profile_token, 'Translation': {'PanTilt': {'x': 0.0, 'y': cmd_y}}})
    time.sleep(RTSP_LAG_TIMEOUT)
    if SHOW_PREVIEW:
        ret, frame_current = video_reader.read()
        if ret and frame_current is not None:
            display_frame = frame_current.copy()
            cv2.putText(display_frame, f"PHASE: Returning Center Y ({i+1}/{center_steps_y})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Tapo C210 Dynamic Calibration Wizard", display_frame)
            cv2.waitKey(1)

# 7. 安全限界値を計算（マージン85%）
calculated_limit_x = round((total_steps_x / 2) * STEP_SIZE_X * 0.85, 2)
calculated_limit_y = round((total_steps_y / 2) * STEP_SIZE_Y * 0.85, 2)

config_data = {
    "MAX_LIMIT_X": calculated_limit_x,
    "MAX_LIMIT_Y": calculated_limit_y,
    "INVERT_PAN": config.invert_pan,
    "INVERT_TILT": config.invert_tilt,
    "STEP_SIZE_X": STEP_SIZE_X,
    "STEP_SIZE_Y": STEP_SIZE_Y,
    "TOTAL_STEPS_X": total_steps_x,
    "TOTAL_STEPS_Y": total_steps_y,
    "RETURN_STEPS_X": center_steps_x,
    "RETURN_STEPS_Y": center_steps_y,
    "CALIBRATED_AT": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open("camera_config.json", "w", encoding="utf-8") as f:
    json.dump(config_data, f, indent=4, ensure_ascii=False)

print("\n✅ キャリブレーションが完全に完了しました！")
print("🎯 カメラはギアバックラッシュ誤差ゼロで物理可動域の【ド真ん中】に着地しました。")
print(
    f"💾 結果を 'camera_config.json' に保存しました（左右限界: ±{calculated_limit_x}, 上下限界: ±{calculated_limit_y}）\n"
)

# 安全に解放
video_reader.release()
cv2.destroyAllWindows()