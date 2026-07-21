import sys
import time
import logging
from pico.config import AppConfig, build_rtsp_url
from pico.video_reader import RTSPVideoReader
from pico.cli.ptz import PTZActuator
from pico.cli.perception import OnDemandPerceptionCLI

# 標準エラーに出力する
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    print("==========================================================")
    print("🚀 コグニティブ・フォーカストラッキング・エッジシステム 起動")
    print("==========================================================")
    print("- 追従対象クラス: person (人物)")
    print("- モニター画面: OpenCV デスクトップモニター (自動起動)")
    print("※ 終了するには、OpenCV ウィンドウ上で 'ESC' キーを押すか、")
    print("   ターミナル上で 'Ctrl+C' を押してください。")
    print("----------------------------------------------------------")

    try:
        config = AppConfig()
    except Exception as e:
        logger.error(f"Configuration load failed: {e}")
        sys.exit(1)

    # 1. 共有 RTSP リーダーの作成 (システム唯一)
    rtsp_url = build_rtsp_url(config.tapo_user, config.tapo_pass, config.tapo_ip)
    logger.info(f"Connecting to RTSP stream: {config.tapo_ip}...")
    shared_reader = RTSPVideoReader(rtsp_url)
    time.sleep(1.5)  # ストリームの初期化・バッファリング待ち

    # 2. 知覚（YOLO/モニター）モジュールの初期化 (共有リーダーを適用)
    logger.info("Initializing perception module (YOLO/Monitor Window)...")
    perception = OnDemandPerceptionCLI(config, shared_reader=shared_reader)
    
    # 3. PTZ制御モジュールの初期化
    logger.info("Initializing PTZ controller...")
    ptz = PTZActuator(config)

    # 4. モジュール間の連携 (YOLO検出スレッドとPTZ追跡ターゲット情報の紐付け)
    perception.set_ptz_actuator(ptz)
    
    # 5. モニターウィンドウの自動起動
    logger.info("Starting OpenCV Live Monitor...")
    perception.start_monitor()

    # 6. 自律自動追尾ループの実行 (ユーザーが終了するまで無限ループ)
    try:
        logger.info("Starting autonomous tracking loop for class: 'person'...")
        # 共有リーダーと追従対象クラスを渡してPIDサーボループを走らせる
        ptz.lockon(shared_reader, class_filter="person")
    except KeyboardInterrupt:
        logger.info("Tracking loop stopped by user interrupt.")
    finally:
        logger.info("Shutting down and releasing resources...")
        perception.close()
        ptz.shutdown()
        # 共有リーダーを安全に解放
        shared_reader.release()
        logger.info("System shut down successfully.")

if __name__ == "__main__":
    main()
