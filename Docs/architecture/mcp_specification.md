---
type: Concept
title: Model Context Protocol (MCP) Specification
description: Cognitive Surveillance ツールおよび外部AIクライアント間を連携する MCP サーバー仕様と全ツール定義
status: active
timestamp: 2026-08-01T10:30:00+09:00
tags:
  - mcp
  - protocol
  - surveillance
  - tools
  - api
sources:
  - id: mcp_spec_doc
    resource: /Docs/raw/mcp_specification.md
    title: MCP Specification Document
---

# Model Context Protocol (MCP) Specification

## 1. 概要

Model Context Protocol (MCP) は、AIモデル（Claude DesktopやClaude Code等）が監視カメラシステム `Pico-Argus` の各種機能（物理PTZ操作、YOLOリアルタイム物体認識、VLMスポット解釈、SQLite長期記憶Wiki等）を安全かつ標準化されたツールとして呼び出すためのプロトコル実装です。

## 2. クライアント設定 & 認証情報

### 2.1 接続設定例 (Claude Code / Claude Desktop)
`mcpServers` 設定に、以下のように登録します：
```json
{
  "mcpServers": {
    "cognitive-surveillance": {
      "command": "uv",
      "args": ["run", "python", "-m", "pico.mcp.server"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

### 2.2 前提環境変数 (`.env`)
```ini
TAPO_USER=your_tapo_app_username    # Tapo ONVIFユーザー名
TAPO_PASS=your_tapo_app_password    # Tapo ONVIFパスワード
TAPO_IP=192.168.0.10               # TapoカメラIP
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b            # 使用するVLM名
OLLAMA_MAX_RPM=12                  # 1分間あたりの最大APIリクエスト許容数
```

## 3. 提供ツール仕様一覧 (Cognitive Surveillance)

### 3.1 `get_active_tracks`
常時更新されている YOLO + ByteTrack 追跡オブジェクトのテキスト化メタデータ（JSON情報）のみを高速に一括取得します。画像処理のオーバーヘッドがないため極めて軽量です。
- **引数**: なし
- **レスポンス例**:
  ```json
  [
    {
      "track_id": 3,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.82,
      "bbox": [840, 210, 310, 800],
      "normalized_center": [0.433, 0.471],
      "normalized_area": 0.0828,
      "position_label": "middle-center",
      "warning_zone_triggered": true
    }
  ]
  ```

### 3.2 `move_camera`
カメラを任意のアングル方向へ指定した角度（Pan / Tilt 相対移動量）で移動・旋回させます。`pan=0, tilt=0` を指定した場合は正面中心原点(0,0)へ戻ります。
- **引数**:
  - `pan` (number, **必須**): 水平旋回量 (-0.96 ～ 0.96。正: 右、負: 左)。
  - `tilt` (number, **必須**): 垂直旋回量 (-0.89 ～ 0.89。正: 上、負: 下)。

### 3.3 `calibrate_home`
カメラの物理ゼロ点補正（ホームアライメント）を実行します。物理限界へのブラインド突き当て動作により正確な中心原点(0,0)を再校正・再確立します。
- **引数**: なし
- **レスポンス例**: 「Success: カメラの物理ゼロ点補正（ホームアライメント）が完了しました。中心原点 (0.00, 0.00) に再校正完了。」

### 3.4 `conduct_room_survey`
カメラを全方位（左、中央、右、上）へ自律的に順次旋回させて室内をマルチアングル知覚し、部屋の状況や検出オブジェクトを解析して Obsidian Long-Term Wiki ページ `[[部屋の全方位環境調査記録_20260725]]` に自動記録します。
- **引数**: なし

### 3.5 `analyze_crop_image`
指定されたオブジェクトのバウンディングボックス領域を切り出して高解像度化し、Ollama VLM を用いたオンデマンド画像解析を行います。
- **引数**:
  - `query` (string, **必須**): VLMに解析させたい具体的な指示や質問（例: 「この人物は何を持っているか詳細に答えてください」）。
  - `track_id` (integer, オプション): 解析対象の特定のYOLO追跡ID。
  - `class_filter` (string, オプション): 解析対象のオブジェクトクラス名。IDが不明な場合に使用します。
- **レスポンス例**: 「提供された画像によると、人物は右手で黒いスマートフォンを操作しながらリラックスした様子で立っています...」

### 3.6 `set_tracking_target`
物理 PID サーボループを介して、カメラが自動追従して常に画面中央に捉え続けるべき優先ロックオンターゲットを設定または解除します。
- **引数**:
  - `track_id` (integer, オプション): ロックオン追従する対象のトラックID。
  - `class_filter` (string, オプション): 自動追尾捕捉を開始するオブジェクトクラス名。見失っても画面内に出現した同クラスのオブジェクトを自動再捕捉します。

### 3.7 `get_live_snapshot`
カメラの現在のライブ映像フレームをキャプチャし、その画像をチャットインターフェース上に Markdown 画像リンクとして展開表示します。
- **引数**: なし
- **レスポンス例**:
  > Success: 現在のカメラフレームをキャプチャしました。
  > ![Live Snapshot](file:///app/wiki/snapshots/live_172023901.jpg)

### 3.8 `search_wiki`
SQLite FTS5 Trigram 検索を用い、過去の観測思い出や環境固有のルールに関する Wiki ナレッジを想起します。
- **引数**:
  - `query` (string, **必須**): 検索・想起をかけるための日本語キーワード。

### 3.9 `write_wiki`
新しい観測事実、ユーザー指定ルール、会話インサイト、外部検索結果を OKF (Obsidian Knowledge Format) 形式 Markdown に書き込み、SQLite FTS5 インデックス、WikiLinks (`[[...]]`)、およびエイリアス（名寄せ）テーブルを同期更新します。
- **引数**:
  - `filepath` (string, **必須**): 保存先の Markdown ファイルパス (例: `'wiki/known_objects_tama.md'`)。
  - `title` (string, **必須**): 記憶・ナレッジのタイトル。
  - `content` (string, **必須**): 記録する観察内容・ルール本文 (本文内の `[[項目名]]` は相互リンクとして自動抽出)。
  - `tags` (string, オプション): スペース区切りの分類用タグ文字列。
  - `aliases` (array or string, オプション): 表記ゆれ名寄せ用の別名リスト。

### 3.10 `get_perception_status`
常時知覚エンジンの現在の稼働状況（FPS）、リアルタイム追跡リスト、ロックオン設定、および直近で発火した能動イベント履歴を一括照会・問い合わせします。
- **引数**: なし

### 3.11 `configure_event_filter`
能動的イベント発火の過剰抑止・抑制ルール（同一オブジェクト再発火防止クールダウン秒数、発火対象クラスの制限）をカスタマイズ設定します。
- **引数**:
  - `cooldown_sec` (number, オプション): 同一イベント・IDに対する再発火抑制秒数。
  - `allowed_classes` (array of string, オプション): 監視発火対象とするクラス名リスト。

## 4. 推奨される協調動作（オーケストレーション）フロー

LLMクライアント（Claude Code 等）から本サーバーを使用する際は、無制限なリソース浪費を防ぐために以下の「能動的知覚ステップ」を順守することを推奨します。

```
[ 1. get_active_tracks で周囲を定期スキャン ]
                     │
                     ▼
  【警告ゾーン内に人/オブジェクトを検知？】
   ├── NO  ──► 処理をスリープ（または待機）
   └── YES ──► [ 2. set_tracking_target でカメラの中心に捉える ]
                     │
                     ▼ （PTZ旋回とピント安定まで1秒待機）
               [ 3. analyze_crop_image でVLMによるスポット詳細観察 ]
                     │
                     ▼
               [ 4. search_wiki で過去の履歴を想起確認 ]
                     │
                     ▼
               [ 5. 新情報があれば write_wiki で直接長期記憶へコミット ]
```

## 5. 関連概念

* [Camera Agent System](./camera_agent.md) - MCPのバックエンドとなる追跡・ビジョンエンジン
* [MCP Usecases](../domain/mcp_usecases.md) - MCPツールの具象的な活用手順
