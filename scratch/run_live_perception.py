import time
import logging
from pico.config import AppConfig
from pico.cli.perception import OnDemandPerceptionCLI
from pico.cli.ptz import PTZActuator

import os
import signal

os.makedirs("logs", exist_ok=True)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# コンソールハンドラー
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# ファイルハンドラー (logs/perception_live.log へ常時書き込み)
file_handler = logging.FileHandler("logs/perception_live.log", mode="w", encoding="utf-8")
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger("LiveServerRunner")

def emergency_shutdown(signum, frame):
    logger.info("👋 ユーザー操作 (Ctrl+C) を検知しました。即時シャットダウンします。")
    os._exit(0)

signal.signal(signal.SIGINT, emergency_shutdown)

def main():
    logger.info("🚀 【常時知覚サーバー ＆ 自動人物物理PTZ追尾起動】...")
    config = AppConfig()
    logger.info(f"📋 設定読み込み完了: SHOW_MONITOR={config.show_monitor}")

    cli = OnDemandPerceptionCLI(config)
    ptz = PTZActuator(config)
    cli.set_ptz_actuator(ptz)

    # MCP サーバーモジュールとのインスタンス共有（二重生成・位置乖離防止）
    from pico.mcp import server as mcp_server
    mcp_server.ptz = ptz
    mcp_server.perception = cli
    mcp_server.shared_reader = cli.reader

    if config.align_to_home:
        logger.info("🎯 起動時カメラ物理アライメント（原点校正・中心復帰）を実行しています...")
        ptz._init_ptz(video_reader=cli.reader, align=True)

    logger.info("📺 モニターウィンドウ (Cognitive Surveillance Monitor) を起動しました。")
    logger.info("🎯 人物 (class_filter='person') に対する自動物理PTZロックオン追尾を開始します。")

    # 起動時の原点アライメント完了後、人物(person)の自動物理ロックオン追尾を自動ON
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
        logger.info("👋 ユーザー操作 (Ctrl+C) により停止シグナルを受信しました。")
    finally:
        try:
            ptz.stop_lockon()
            cli.close()
        except Exception:
            pass
        logger.info("🛑 常時知覚サーバーおよびPTZ追尾を安全に停止しました。")
        os._exit(0)

if __name__ == "__main__":
    main()
