---
type: Log
title: Knowledge Update Log
---

# Knowledge Update Log

## 2026-08-03

* **Refactoring**: 堅牢性向上のための包括的なリファクタリングを実施しました。
  * **ContinuousMove 暴走防止用の安全装置 (Watchdog) の導入**: `pico/onvif_client.py` の `PTZController` に `move_continuous` コマンドと、Python 層の `threading.Timer` バックアップタイマー（および ONVIF 標準の Timeout パラメータ）による自動停止 Watchdog 機構を実装。
  * **RTSP 接続・読み込みタイムアウト処理の追加**: `pico/video_reader.py` の `RTSPVideoReader` の接続時に OpenCV の `CAP_PROP_OPEN_TIMEOUT_MSEC` と `CAP_PROP_READ_TIMEOUT_MSEC`（5秒）を設定し、WiFi の瞬断やフリーズによるブロッキングを排除。
  * **アトミック書き込み・ファイルロックによるマルチタスク・プロセス間排他制御**: `pico/cli/memory.py` の `write_knowledge_data` に `tempfile.NamedTemporaryFile` + `os.replace` によるアトミック置換、および Python 標準ライブラリ `fcntl.flock`（Windows用のフォールバック付）によるファイル排他ロック機構を実装。さらに、マルチスレッド環境での SQLite 書き込み衝突を防ぐために `pico/memory.py` の `MemoryStore` を `threading.RLock()` を用いて完全スレッド安全化。
  * **MCP エラーレスポンスの構造化・標準化**: `pico/mcp/server.py` の `handle_call_tool` にて例外発生時に `{"status": "error", "error_type": "...", "message": "...", "details": "..."}` 形式の JSON エラーを返すように正規化。

## 2026-08-01

* **Creation**: 既存の `docs/` 内仕様書を一次情報源（`Docs/raw/`）とし、OKF形式のナレッジベースを初期化・再構築しました。
