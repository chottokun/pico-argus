---
type: Log
title: Knowledge Update Log
---

# Knowledge Update Log

## 2026-08-01

* **Update**: リポジトリ調査結果に基づき、OKF（Obsidian Knowledge Format）形式の概念ドキュメントを大幅に強化・整合化しました。
  * **Architecture / Camera Agent (`Docs/architecture/camera_agent.md`)**:
    * 能動的知覚 (Active Perception) パイプライン、LangGraph 1.0動的ステート定義、各種ガードレール（Epoch-Based Guard, Dwell Time, Slew Rate制限, RTSPシングルトン, MonitorWindow, RPMLimiter）、AIエージェント行動規範、物理運動極性（相対移動方向・極性整合テーブル）などの仕様を詳細に記載。
  * **Architecture / MCP Specification (`Docs/architecture/mcp_specification.md`)**:
    * クライアント接続設定、前提環境変数、全11種類の提供ツール（`get_active_tracks`, `move_camera`, `calibrate_home`, `conduct_room_survey`, `analyze_crop_image`, `set_tracking_target`, `get_live_snapshot`, `search_wiki`, `write_wiki`, `get_perception_status`, `configure_event_filter`）の入出力パラメータ、レスポンス形式、推奨オーケストレーションフローを網羅。
  * **Domain / MCP Usecases (`Docs/domain/mcp_usecases.md`)**:
    * 室内全方位スキャン・目録構築、物理PTZ自動追従ロックオン、高解像度クロップVLM精査、記憶想起と環境変化検知、ギア遊び補正など5大シナリオのシーケンスと期待される効果をドキュメント化。
  * **Domain / Memory CLI (`Docs/domain/memory_cli.md`)**:
    * 記憶蓄積・更新CLIの挙動（既存ファイルの上書き追記機能等）、Trigram FTS5 SQLite 内部スキーマ、4テーブル（`wiki_metadata`, `wiki_fts`, `wiki_links`, `wiki_aliases`）、2段階検索想起フォールバック、および Python API 使用例を詳細化。
  * **Infrastructure / Tapo Configuration (`Docs/infrastructure/tapo_config.md`)**:
    * 新規にインフラ概念ページを構築。実機キャリブレーション設定（`MAX_LIMIT`等）や極性フラグ（`INVERT_PAN`/`INVERT_TILT`）の標準設定スキーマを定義。
* **Creation**: 既存の `docs/` 内仕様書を一次情報源（`Docs/raw/`）とし、OKF形式のナレッジベースを初期化・再構築しました。
