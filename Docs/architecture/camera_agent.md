---
type: Concept
title: Camera Agent System
description: Tapoカメラ制御、顔追跡、YOLO物体検出、ガードレールを統合したリアルタイムビジョンエージェントのアーキテクチャ
status: active
timestamp: 2026-08-01T10:30:00+09:00
tags:
  - camera
  - tracking
  - yolo
  - tapo
  - architecture
sources:
  - id: camera_agent_doc
    resource: /Docs/raw/camera_agent.md
    title: Camera Agent Specification
---

# Camera Agent System

## 1. 概要

Camera Agent System は、Tapo パンチルトカメラを用いたリアルタイム映像ストリーミング、物体検出（YOLOv8/Haar Cascade）、動体追跡、カメラ首振り制御を行う統合エージェントシステムです。
本システムでは、LLM（Gemma 4 等）を司令塔とし、「物理PTZサーボ」「超軽量な知覚バッファとしてのYOLO」「オンデマンドなVLM解釈」「SQLite 長期記憶 Wiki」を統合し、エッジPC環境における高精度なカメラ制御と視覚解析を実現します。

## 2. 詳細仕様・構造

### 2.1 常時画像監視を廃止した「時間分割・テキストメタデータ知覚」
常時重いVLMで画像変化を監視する代わりに、軽量な YOLO + ByteTrack によるテキスト化メタデータを常時稼働バッファとし、詳細な意味理解が必要な瞬間のみスポットVLMを呼び出す構成としています。
1. **内省思考（Thinking Mode）による仮説生成**: YOLOメタデータから「低確信度の物体が存在する」などの仮説を生成。
2. **物理ツールコール（物理視線シフト）**: カメラを指定ターゲットに向けて旋回し、自動光学ズームで物理的にクローズアップ（画角面積比20%程度）。
3. **スポットVLM解釈（能動的知覚の開眼）**: 高解像度フレームから対象領域だけをクロップしてVLMに入力・解析。
4. **自律Wikiコンパイル（OKF）**: VLMの結果から、自動で長期記憶（OKF形式）に蓄積。

### 2.2 LangGraph 1.0 動的ステート＆グラフ設計
LLMを司令塔とするため、LangGraphのグラフノードは「状態評価 ➔ ツール実行 ➔ 状態変異 ➔ 再帰プラン」を自律決定できるルーティングとします。

#### AgentState スキーマ定義
- `active_tracks`: YOLO/ByteTrackのリアルタイムメタデータ
- `current_frame_path`: ズーム・スポットクロップ画像の保存先パス
- `lockon_mode`: 自律/指定などのロックモード
- `target_track_id`: ロックオン対象の tracker_id
- `ptz_dynamic_gain`: 適応型PID比例ゲイン
- `agent_goal`: 自律タスク目標
- `recalled_knowledge`: SQLite FTS5 から想起した過去のルール
- `conversation_history`: 対話ログ
- `state_epoch`: 割り込み競合を防止するインクリメンタルエポック

### 2.3 物理＆論理ガードレール
1. **エポックベース状態変異（Epoch-Based Guard）**: ユーザー割り込み時に動作中だったタスクをキャンセルし、`state_epoch` をインクリメント。遅れて返ってきた古い結果（古いエポック）を自動破棄して先祖返りを100%防止。
2. **適応型PID制御 ＋ 衝突防止（Dwell Time）＆スルーレート制限**:
   - 連動コマンドロストによる激突融損を防止するため、350msの自動自律停止（`Dwell Time`）をバインドして送信。限界角度での連続駆動5秒超で強制 Stop。
   - 急制動を避けるため、1ステップあたりの速度変化を `max_acceleration`（デフォルト: `0.03`）以内に制限する **`Slew Rate制限`** を適用。
   - 安全限界クランプ（`MAX_LIMIT_X`, `MAX_LIMIT_Y`）による自動位置保護。
3. **共有ビデオリーダー（RTSP シングルトン）**:IPカメラの接続セッション数制限（上限約2セッション）をクリアするため、システム全体で唯一の共有 `RTSPVideoReader` インスタンスをライフサイクル管理。
4. **自動起動モニター（MonitorWindow）**: OpenCV ウィンドウを別スレッドで自動起動し、BBox（緑枠）、ロックオン対象（赤枠）、PID偏差ラインを 30 FPS でリアルタイムにオーバーレイ描画。
5. **可変レートリミッター（RPMLimiter）**: 環境変数 `OLLAMA_MAX_RPM` を介して動的に最大 RPM（デフォルト: 12）を設定可能なリミッター。

## 3. 制約・注意事項

### 3.1 AIエージェント運用・行動規範 (Agent Operational Rules)
- **直接的な画像ファイルアクセスの禁止 (No Image Direct Access)**: `get_live_snapshot` などのツールから返された画像パスに対して、`view_file` 等のファイル読み込みツールを使用して直接ロード・自己ビジョン解釈を行うことを**厳重に禁止**します。画像は人間のユーザーに提示するためにのみ使用します。
- **能動的知覚パイプラインの順守**: 状況把握にはまず `get_active_tracks` からテキストメタデータを取得し、詳細が必要な場合は `analyze_crop_image` でVLM（Ollama）に解析を代行させます。

### 3.2 物理カメラ運動方向とONVIF相対移動の符号・極性反転
Tapo C210 などを**正立設置**して使用する場合、正フィードバックを合致させるため、`INVERT_PAN: true` および `INVERT_TILT: true` の両方を `true` に設定することが必須です。

| 移動目的 (物理方向) | 論理移動量 (dx, dy) | ONVIF Raw コマンド | 物理モーター運動 |
|---|---|---|---|
| **物理左 (LEFT)** | `dx < 0` | `cmd_x = -dx` (プラス) | 反時計回り（左）へ旋回 |
| **物理右 (RIGHT)** | `dx > 0` | `cmd_x = -dx` (マイナス) | 時計回り（右）へ旋回 |
| **物理下 (BOTTOM)** | `dy < 0` | `cmd_y = dy` (マイナス) | 下方へ俯く |
| **物理上 (TOP)** | `dy > 0` | `cmd_y = dy` (プラス) | 上方へ仰ぎ見る |

## 4. 関連概念

* [MCP Specification](./mcp_specification.md) - カメラ操作を標準インターフェース化するMCP層
* [MCP Usecases](../domain/mcp_usecases.md) - 監視カメラシステムの実際の運用ユースケース
* [Tapo Config](../infrastructure/tapo_config.md) - 安全クランプ限界値と運動極性に関する構成情報
