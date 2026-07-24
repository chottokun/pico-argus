# コグニティブ・フォーカストラッキング MCP サーバー仕様書
## Claude Code / Claude Desktop 連携および外部 LLM プラットフォーム対応

本仕様書は、Tapo カメラの物理制御（PTZ）、YOLOによるリアルタイム物体認識、Ollama を用いたオンデマンド VLM 画像解析、および SQLite 長期記憶 Wiki を統合した **MCP（Model Context Protocol）サーバー** の接続方法およびツール仕様を定義したドキュメントである。

本仕様に沿って設定を行うことで、**Claude Code** や **Claude Desktop** などの外部 LLM クライアントから、物理カメラデバイスと長期記憶ベースを自律的に操作・制御させることが可能になる。

---

## 🚀 1. クライアント設定（接続・マウント方法）

### A. Claude Code の設定
Claude Code で本 MCP サーバーを読み込むには、プロジェクトルートの `.claudeco/config.json`（存在しない場合は新規作成）に以下の接続定義を追加するか、Claude Code 起動時にコマンドラインから登録します。

**設定ファイルへの追加例:**
```json
{
  "mcpServers": {
    "cognitive-surveillance": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "pico.mcp.server"
      ],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

### B. Claude Desktop の設定
Claude Desktop アプリでマウントして使用する場合は、以下の設定ファイル（OS別）に記述を追加します。

*   **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
*   **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**設定内容:**
```json
{
  "mcpServers": {
    "cognitive-surveillance": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "pico.mcp.server"
      ],
      "env": {
        "PYTHONPATH": "E:/Python Scripts/Pico"
      },
      "cwd": "E:/Python Scripts/Pico"
    }
  }
}
```
*(注: `PYTHONPATH` および `cwd` の絶対パスは、実際のプロジェクトディレクトリの環境に合わせて適宜書き換えてください)*

---

## 🔑 2. 前提環境と認証情報
MCP サーバーは起動時にプロジェクトの `.env` ファイルに記述された認証情報を読み込みます。起動前に以下の環境変数が設定されていることを確認してください。

```ini
TAPO_USER=your_tapo_app_username    # TapoのONVIF高度設定で作成したユーザー名
TAPO_PASS=your_tapo_app_password    # TapoのONVIF高度設定で作成したパスワード
TAPO_IP=10.3.100.176               # TapoカメラのローカルIPアドレス
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b            # 使用するVLM/LLMモデル名
OLLAMA_MAX_RPM=12                  # 1分間あたりの最大APIリクエスト許容数
```

---

## 🛠️ 3. 提供ツール仕様一覧

### 3.1 `get_active_tracks`
常時更新されている YOLO + ByteTrack 追跡オブジェクトのテキスト化メタデータ（JSON情報）のみを高速に一括取得します。画像処理のオーバーヘッドがないため極めて軽量です。

*   **引数:** なし
*   **レスポンス例:**
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

### 3.2 `analyze_crop_image`
指定されたオブジェクトのバウンディングボックス領域を切り出して高解像度化し、Ollama VLM を用いたオンデマンド画像解析を行います。

*   **引数:**
    - `query` (string, **必須**): VLMに解析させたい具体的な指示や質問（例: 「この人物は何を持っているか詳細に答えてください」）。
    - `track_id` (integer, オプション): 解析対象の特定のYOLO追跡ID。
    - `class_filter` (string, オプション): 解析対象のオブジェクトクラス名（例: `'person'`, `'suitcase'`）。IDが不明な場合に使用します。
*   **レスポンス例:**
    > 「提供された画像によると、人物は右手で黒いスマートフォンを操作しながらリラックスした様子で立っています。眼鏡を着用しており、周囲に怪しいアクションは見られません。」

### 3.3 `set_tracking_target`
物理 PID サーボループを介して、カメラが自動追従して常に画面中央に捉え続けるべき優先ロックオンターゲットを設定または解除します。

*   **引数:**
    - `track_id` (integer, オプション): ロックオン追従する対象のトラックID。
    - `class_filter` (string, オプション): 自動追尾捕捉を開始するオブジェクトクラス名（例: `'person'`）。見失っても画面内に出現した同クラスのオブジェクトを自動再捕捉します。
*   *※注: 両方のパラメータを省略するか null を指定した場合は、追従ターゲットが解除されます。*
*   **レスポンス例:**
    > 「Success: 追跡ターゲットを ID: 3 に設定し、自律自動追尾を開始しました。」

### 3.4 `get_live_snapshot`
カメラの現在のライブ映像フレームをキャプチャし、その画像をチャットインターフェース上に Markdown 画像リンクとして展開表示します。

*   **引数:** なし
*   **レスポンス例:**
    > 「Success: 現在のカメラフレームをキャプチャしました。
    > ![Live Snapshot](file:///e:/Python%20Scripts/Pico/wiki/snapshots/live_172023901.jpg)」

### 3.5 `search_wiki`
SQLite FTS5 Trigram 検索を用い、過去の観測思い出や環境固有のルールに関する Wiki ナレッジを想起します。

*   **引数:**
    - `query` (string, **必須**): 検索・想起をかけるための日本語キーワード（例: `'家主'`, `'タマちゃん'`）。
*   **レスポンス例:**
    ```json
    [
      {
        "filepath": "wiki/rules.md",
        "title": "エッジ状況分析基本指示",
        "tags": "status person action",
        "content": "人(person)やオブジェクトを検知した場合、その状態や行動・状況を客観的に観察し分析すること。"
      }
    ]
    ```

### 3.6 `write_wiki`
新しい観測事実、ユーザー指定ルール、会話インサイト、外部検索結果を OKF (Obsidian Knowledge Format) 形式 Markdown に書き込み、SQLite FTS5 インデックス、WikiLinks (`[[...]]`)、およびエイリアス（名寄せ）テーブルを同期更新します。

*   **引数:**
    - `filepath` (string, **必須**): 保存先の Markdown ファイルパス (例: `'wiki/known_objects_tama.md'`)。
    - `title` (string, **必須**): 記憶・ナレッジのタイトル。
    - `content` (string, **必須**): 記録する観察内容・ルール本文 (本文内の `[[項目名]]` は相互リンクとして自動抽出)。
    - `tags` (string, オプション): スペース区切りの分類用タグ文字列 (例: `'pet cat profile'`)。
    - `aliases` (array or string, オプション): 表記ゆれ名寄せ用の別名リスト (例: `["タマ", "猫のタマ"]`)。
*   **レスポンス例:**
    > 「Success: Knowledge saved to wiki/known_objects_tama.md. Result: {"status": "success", "filepath": "wiki/known_objects_tama.md"}」

---

## 🧠 4. 推奨される協調動作（オーケストレーション）フロー
LLMクライアント（Claude Code 等）から本サーバーを使用する際は、無制限なリソース浪費を防ぐために以下の**「能動的知覚ステップ」**を順守することを推奨します。

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
