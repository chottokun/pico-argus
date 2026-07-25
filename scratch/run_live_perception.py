import time
import logging
from pico.config import AppConfig
from pico.cli.perception import OnDemandPerceptionCLI
from pico.cli.ptz import PTZActuator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveServerRunner")

def main():
    logger.info("🚀 【常時知覚サーバー ＆ 自動人物物理PTZ追尾起動】...")
    config = AppConfig()
    logger.info(f"📋 設定読み込み完了: SHOW_MONITOR={config.show_monitor}")

    cli = OnDemandPerceptionCLI(config)
    ptz = PTZActuator(config)
    cli.set_ptz_actuator(ptz)

    logger.info("📺 モニターウィンドウ (Cognitive Surveillance Monitor) を起動しました。")
    logger.info("🎯 人物 (class_filter='person') に対する自動物理PTZロックオン追尾を開始します。")

    # 人物自動物理ロックオン追尾の開始 (常時知覚ループ内で安全に滑らか駆動)
    ptz.start_lockon(class_filter="person")

    try:
        count = 0
        while True:
            time.sleep(2.0)
            count += 1
            status = cli.get_perception_status_data()
            tracks = status.get("active_tracks", [])
            events = status.get("recent_events", [])
            lockon_info = f"[Target ID: {ptz.lockon_target_id}, Class: {ptz.lockon_class_name}]" if ptz.lockon_active else "[IDLE]"
            evt_summary = f" | 最新イベント: {events[0]['event_type']} Track {events[0]['track_id']} ({events[0]['class_name']})" if events else ""
            logger.info(f"⏱️ 稼働中 [{count*2}s経過] | LOCKON: {lockon_info} | FPS: {status['fps']} | 検出数: {len(tracks)}{evt_summary}")

    except KeyboardInterrupt:
        logger.info("👋 ユーザー操作により停止シグナルを受信しました。")
    finally:
        ptz.stop_lockon()
        cli.close()
        logger.info("🛑 常時知覚サーバーおよびPTZ追尾を安全に停止しました。")

if __name__ == "__main__":
    main()
