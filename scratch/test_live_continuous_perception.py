import time
import json
import logging
from pico.config import AppConfig
from pico.cli.perception import OnDemandPerceptionCLI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LivePerceptionTest")

def main():
    logger.info("🎬 【たっぷり実地テスト開始】実カメラ接続 ＆ 常時知覚エンジンの長期連続稼働テスト...")
    config = AppConfig()
    cli = OnDemandPerceptionCLI(config)

    try:
        # モニターウィンドウを起動
        cli.start_monitor()
        logger.info("📺 モニターウィンドウを自動起動しました。")

        # フェーズ 1: 10秒間の連続常時スキャン (ウォームアップ ＆ 初期ステータス取得)
        logger.info("⏳ [フェーズ 1] 10秒間の常時知覚ループ＆リアルタイム描画のモニタリング...")
        for i in range(10):
            time.sleep(1.0)
            status = cli.get_perception_status_data()
            logger.info(f"  └─ Tick {i+1}/10 | Status: {status['engine_status']} | FPS: {status['fps']} | Tracks: {status['active_track_count']} | Events: {len(status['recent_events'])}")

        # フェーズ 2: 動的イベントフィルターの設定変更 (クールダウン: 2秒, 対象クラス制限なし)
        logger.info("⚙️ [フェーズ 2] イベントフィルター設定の動的変更 (cooldown_sec=2.0)...")
        updated_status = cli.configure_event_filter_data(cooldown_sec=2.0, allowed_classes=None)
        logger.info(f"  └─ 設定変更後のステータス: Cooldown={updated_status['cooldown_sec']}s")

        # フェーズ 3: 15秒間のイベント発火・検出モニタリング
        logger.info("⏳ [フェーズ 3] 15秒間の能動的イベント発火・デバウンス動作テスト (画面内を動かしてみてください)...")
        for i in range(15):
            time.sleep(1.0)
            status = cli.get_perception_status_data()
            recent_events = status["recent_events"]
            event_msg = f" (最新イベント: {recent_events[0]['event_type']} -> ID {recent_events[0]['track_id']})" if recent_events else ""
            logger.info(f"  └─ Tick {i+1}/15 | Tracks: {status['active_track_count']}{event_msg}")
            if status["active_tracks"]:
                logger.info(f"      Active Tracks: {status['active_tracks']}")

        # フェーズ 4: オンデマンドVLMおよびスナップショット並列実行テスト
        logger.info("📸 [フェーズ 4] 常時知覚動作中におけるライブスナップショット ＆ VLM解析の並列テスト...")
        snap_res = cli.get_live_snapshot_data()
        logger.info(f"  └─ スナップショット保存結果: {snap_res}")

        tracks = cli.get_tracks_data()
        if tracks:
            target_id = tracks[0]["track_id"]
            logger.info(f"  └─ Track ID {target_id} に対してオンデマンド VLM クロップ解析を実行中...")
            crop_res = cli.analyze_crop_data(track_id=target_id, query="画像内の様子を簡潔に説明してください。")
            logger.info(f"  └─ VLM 解析結果: {crop_res.get('response', crop_res.get('error'))}")
        else:
            logger.info("  └─ 現在特定の検出トラックがないため、フレーム全体の VLM 解析を実行中...")
            crop_res = cli.analyze_crop_data(query="カメラ全体の様子を簡潔に説明してください。")
            logger.info(f"  └─ VLM 解析結果: {crop_res.get('response', crop_res.get('error'))}")

        # フェーズ 5: 最終安定性チェック
        logger.info("⏳ [フェーズ 5] 最終5秒間の安定駆動チェック...")
        time.sleep(5.0)
        final_status = cli.get_perception_status_data()
        logger.info(f"✅ 【実地テスト完了】最終ステータス: FPS={final_status['fps']}, 累積イベント数={len(final_status['recent_events'])}")
        print("\n--- 最終照会サマリー ---")
        print(json.dumps(final_status, indent=2, ensure_ascii=False))

    finally:
        cli.close()
        logger.info("🛑 CLIおよび常時知覚ループを安全にクローズしました。")

if __name__ == "__main__":
    main()
