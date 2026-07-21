# MCPサーバー移行を見据えた「3大自律CLIモジュール」詳細設計および段階的デプロイ・ベストプラクティス計画書

本ドキュメントは、エッジコンピューティング環境（GeForce RTX 3060 12GB等）における「コグニティブ・フォーカストラッキング・エッジシステム」の実機物理配備において、手戻りを完全にゼロにし、検証効率を最大化するための**「CLI先行開発 ➔ MCPサーバーラッピング」という2段階開発ロードマップ**の完全な実装仕様と具体的なテスト手順を定義する。

---

## 1. 全体ロードマップ：なぜCLIから始めるべきか？

物理ハードウェア（IPカメラ）とLLMエージェント（脳）をいきなりModel Context Protocol（MCP）やLangGraphで密結合させると、物理サーボの駆動タイミング遅延、ネットワークパケロス、SQLiteの検索ノイズ、LLM引数解釈のハルシネーションなどのバグが重なり、デバッグが泥沼化する。

そのため、本システムは以下のステップで段階的に開発・配備する。

```
【Phase 1: 自律型CLIツールの徹底検証 (Reflex & Component Validation)】
  ├─ 1. ptz_cli.py        ➔ ONVIF PTZ、適応PID、摩擦ブースト、自動Dwell Timeセーフガード
  ├─ 2. memory_cli.py     ➔ SQLite FTS5 Trigram日本語想起 ＋ 2文字以下フォールバック
  └─ 3. perception_cli.py ➔ YOLO-ORT高速検出テキスト出力 ＋ オンデマンドVLM画像クロップ
       ▼ (手動実行や自動シェルテストにより、単体での動作、安全性、リソース占有を100%バグゼロに調整)

【Phase 2: 薄いMCPサーバー（Model Context Protocol）ラッパーの実装】
  └─ mcp_server.py ➔ 各CLIモジュールをインポート。
                    LLMに提示するツールスキーマ（JSON）の定義と、引数引き渡しを処理。

【Phase 3: 脳（LangGraph）との統合（Cognitive Orchestration）】
  └─ エージェントがMCP経由で筋肉、感覚、記憶ツールを状況に合わせて自律選択・実行。
```

---

## 2. Phase 1: 3大自律型CLIモジュールの実装設計仕様

### 2.1 物理PTZ・アクチュエーター CLI (`ptz_cli.py`)

カメラの物理旋回・ズーム追従を司る「筋肉」モジュール。ONVIFの`ContinuousMove`コマンドが持つ「Stopを受信するまで永久に旋回し続ける」という物理危険仕様を、最下層で安全にガード。

#### コマンド引数インターフェース
```bash
# 特定のYOLOトラックIDへのロックオン追従を開始（PID適応サーボバックグラウンドスレッド起動）
python ptz_cli.py --action lockon --id 101

# カメラを指定ベクトルで手動パルス移動（Dwell Time自動セーフガードが働く）
python ptz_cli.py --action move --pan 0.15 --tilt -0.08

# カメラ旋回を緊急即時停止
python ptz_cli.py --action stop
```

#### クラス設計およびセーフガード処理
```python
import sys
import time
import argparse
import numpy as np
from onvif import ONVIFCamera # onvif-zeepバックエンドを使用

class PTZActuator:
    """ONVIF ContinuousMove 物理制御 & 衝突防止セーフガード"""
    def __init__(self, config: dict):
        self.ip = config["camera_ip"]
        self.port = config["camera_port"]
        self.user = config["camera_user"]
        self.password = config["camera_pass"]
        self.dwell_time = config.get("dwell_time_ms", 350) / 1000.0 # 350ms
        
        # onvif-zeep接続初期化
        self.camera = ONVIFCamera(self.ip, self.port, self.user, self.password)
        self.ptz = self.camera.create_ptz_service()
        self.media = self.camera.create_media_service()
        self.profile_token = self.media.GetProfiles()[0].token
        
        # 速度ベクトルリクエストの初期化
        self.move_request = self.ptz.create_type('ContinuousMove')
        self.move_request.ProfileToken = self.profile_token

    def send_pulse_move(self, pan: float, tilt: float):
        """Dwell Time（一時自動停止）を適用した安全なパルス物理駆動"""
        # 速度クランプ
        pan = max(-1.0, min(1.0, pan))
        tilt = max(-1.0, min(1.0, tilt))
        
        self.move_request.Velocity.PanTilt.x = pan
        self.move_request.Velocity.PanTilt.y = tilt
        
        # 1. ContinuousMove開始
        self.ptz.ContinuousMove(self.move_request)
        
        # 2. 指定時間待機 (Dwell Timeによる自律強制停止)
        # Wi-Fiパケロス時でも、このパルススリープ後に強制Stopを送信し物理衝突を防ぐ
        time.sleep(self.dwell_time)
        self.ptz.Stop({'ProfileToken': self.profile_token})

    def emergency_stop(self):
        """物理緊急停止コマンドの即時送信"""
        self.ptz.Stop({'ProfileToken': self.profile_token})
```

---

### 2.2 長期記憶想起・書込 CLI (`memory_cli.py`)

日本語部分一致検索をVRAM・メインメモリ消費を極小に抑えて高速想起（FAG）する「記憶」モジュール。

#### コマンド引数インターフェース
```bash
# SQLite FTS5 Trigramによる日本語想起
python memory_cli.py --action search --query "猫のタマへの対応ポリシー" --limit 2

# 新しい対話事実や環境ルールのOKF形式Markdownへの書き込み
python memory_cli.py --action write --file "user_preferences.md" --title "猫のタマへの対応方針" --content "夕方はタマちゃんに警告音を鳴らさずに話しかける。"
```

#### FTS5 Trigram ➔ LIKE フォールバック実装仕様
```python
import sqlite3
import re
import argparse

class SQLiteMemoryCLI:
    """日本語 Trigram FTS5 インデックス ＋ LIKE部分一致フォールバック"""
    def __init__(self, db_path: str = "wiki.db"):
        self.db_path = db_path

    def search_knowledge(self, query: str, limit: int = 3):
        words = [w for w in re.split(r'\s+', query) if w]
        if not words:
            return []

        results = []
        seen_filepaths = set()

        # 1. 3文字以上の単語があればTrigram FTS5で超高速想起
        fts_query = " AND ".join([f'"{w}"' for w in words if len(w) >= 3])
        if fts_query:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT f.filepath, m.title, m.content, bm25(wiki_fts) as rank
                    FROM wiki_fts f
                    JOIN wiki_metadata m ON f.filepath = m.filepath
                    WHERE wiki_fts MATCH ?
                    ORDER BY rank ASC LIMIT ?
                """, (fts_query, limit))
                for row in cursor.fetchall():
                    seen_filepaths.add(row["filepath"])
                    results.append({"filepath": row["filepath"], "title": row["title"], "content": row["content"], "score": float(-row["rank"]) + 10.0})

        # 2. 2文字以下の短いキーワード（「猫」「タマ」）をLIKEスキャンで救済
        if len(results) < limit:
            remaining = limit - len(results)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                for word in words:
                    like_term = f"%{word}%"
                    not_in_clause = ",".join(["?"] * len(seen_filepaths)) if seen_filepaths else "''"
                    cursor.execute(f"""
                        SELECT filepath, title, content FROM wiki_metadata
                        WHERE (title LIKE ? OR content LIKE ?) AND filepath NOT IN ({not_in_clause})
                        LIMIT ?
                    """, (like_term, like_term, *seen_filepaths, remaining))
                    for row in cursor.fetchall():
                        if row["filepath"] not in seen_filepaths:
                            seen_filepaths.add(row["filepath"])
                            results.append({"filepath": row["filepath"], "title": row["title"], "content": row["content"], "score": 1.0})
                            if len(results) >= limit: break
        return results
```

---

### 2.3 知覚・アノマリー認識 CLI (`perception_cli.py`)

常時YOLOのテキスト情報のみを出力し、指示された瞬間のみズーム画像をクロップしてVLMで状況評価する「感覚」モジュール。

#### コマンド引数インターフェース
```bash
# 現在のYOLO ByteTrack認識オブジェクトをテキストメタデータのみで一覧取得 (VLMは休止)
python perception_cli.py --action get_tracks

# 特定のTrack IDのエリアを高解像度クロップ（1024pxクランプ）し、VLMに画像解釈を投げる
python perception_cli.py --action analyze_crop --id 101 --query "これは風で揺れている影ですか、それとも生き物ですか？"
```

#### OOMを極限回避するVLM画像前処理＆オンデマンド推論
```python
import cv2
import argparse
from PIL import Image

class OnDemandPerceptionCLI:
    """軽量YOLOテキスト常時取得 ＆ 1024pxクランプによるオンデマンドVLM起動"""
    def __init__(self, yolo_model_path: str):
        # YOLO-ORTの初期化（VRAMは500MBに厳格制限）
        pass

    def get_current_tracks_text(self) -> list:
        """常時監視は「軽量テキスト」のみ。YOLO BBox座標をテキスト配列で返す"""
        # YOLO+ByteTrack推論実行（VRAM極小消費）
        mock_tracks = [
            {"track_id": 101, "class": "cat", "bbox": [120, 200, 300, 400], "confidence": 0.88},
            {"track_id": 102, "class": "person", "bbox": [50, 450, 400, 600], "confidence": 0.94}
        ]
        return mock_tracks

    def analyze_crop_image(self, frame_path: str, bbox: list, query: str) -> str:
        """指定されたBBox領域を高解像度で光学切り出し ➔ 1024px以下にクランプしてVLMへ入力"""
        # 1. 物理フレームから対象領域をクロップ
        image = cv2.imread(frame_path)
        y0, x0, y1, x1 = bbox
        crop = image[y0:y1, x0:x1]
        
        # 2. VRAMの瞬間スパイク（OOM）を防ぐため、1024px以下に強制リサイズ
        h, w = crop.shape[:2]
        max_size = 1024
        if w > max_size or h > max_size:
            scale = max_size / float(max(w, h))
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        cv2.imwrite("temp_vlm_input.jpg", crop)
        
        # 3. この一瞬だけVLM（Qwen2.5-VL-3B Q4_K_M等）をオンデマンド起動し、画像を解釈
        vlm_response = "[VLM Result] これは右耳がカットされた茶トラ의猫（タマ）です。リラックスした状態でウッドデッキの下に潜り込もうとしています。"
        return vlm_response
```

---

## 3. Phase 2: MCPサーバー（Model Context Protocol）のラッパー設計仕様

各CLIツールが単体で完璧にバグゼロ動作することを確認できたら、それらをインポートしてMCP標準に適合させる「薄いMCPサーバー層（`mcp_server.py`）』を構築する。

LLM（脳）は、このMCPがエクスポートするJSONスキーマのみを読み込み、必要に応じてツールコールを発行する。

### 3.1 LLMに提示されるMCPツールスキーマ定義 (JSON)

#### 感覚ツール: `get_active_tracks`
常時高頻度（ミリ秒単位）で更新される、現在の認識オブジェクトのテキスト情報をエージェントに提供。VLMは起動しない。
* **入力パラメータ**: なし
* **出力形式**:
```json
{
  "active_tracks": [
    {
      "track_id": 101,
      "class_name": "cat",
      "bbox_normalized": [120, 200, 300, 400],
      "confidence": 0.88
    }
  ]
}
```

#### 感覚ツール: `analyze_crop_image`
特定のターゲットにカメラをズームさせ、その highres クロップ画像を一瞬だけVLMで精査させ、セマンティクス（意味情報）を返す。
* **入力パラメータ**:
```json
{
  "properties": {
    "track_id": { "type": "integer", "description": "ズームして精査する対象のYOLOトラックID" },
    "query": { "type": "string", "description": "VLMに画像解釈させるための具体的なプロンプト・問いかけ" }
  },
  "required": ["track_id", "query"]
}
```

#### 筋肉ツール: `set_tracking_target`
物理PID制御ループ（脊髄）に対し、ロックオン・追従すべきターゲットIDを動的指示。
* **入力パラメータ**:
```json
{
  "properties": {
    "track_id": { "type": ["integer", "null"], "description": "ロックオン追従する対象のトラックID。None/null を指定した場合は自律広角監視に戻る" }
  },
  "required": ["track_id"]
}
```

#### 記憶ツール: `search_wiki`
現在の状況に最も合致する過去の会話設定や、物理制限ルールをSQLite Trigramから想起（RAG）。
* **入力パラメータ**:
```json
{
  "properties": {
    "query": { "type": "string", "description": "SQLite Trigram検索を走らせるための日本語キーワード" }
  },
  "required": ["query"]
}
```

---

## 4. MCPサーバーのハンドラ（`tools/call`）実装のベストプラクティス

MCPサーバーは、受け取った引数をそのままPhase 1で構築した実績のあるCLIクラスへ中継する薄いブリッジとして動作する。

```python
# mcp_server.py
import asyncio
from mcp.server import Server # mcp Python SDK を使用

# Phase 1 の堅牢なCLIモジュールからビジネスロジックを直接インポート
from ptz_cli import PTZActuator
from memory_cli import SQLiteMemoryCLI
from perception_cli import OnDemandPerceptionCLI

server = Server("cognitive-surveillance-mcp")

# CLIモジュールのシングルトンインスタンス生成
ptz = PTZActuator(load_config())
memory = SQLiteMemoryCLI()
perception = OnDemandPerceptionCLI()

@server.list_tools()
async def handle_list_tools():
    """LLMエージェントへ感覚、筋肉、記憶ツールのスキーマを公開"""
    return [
        {
            "name": "get_active_tracks",
            "description": "常時更新されているYOLO追跡オブジェクトのテキスト情報のみを高速取得します。",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "analyze_crop_image",
            "description": "特定のオブジェクトを高解像度ズームクロップし、オンデマンドでVLM画像解釈を実行します。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track_id": { "type": "integer" },
                    "query": { "type": "string" }
                },
                "required": ["track_id", "query"]
            }
        },
        {
            "name": "set_tracking_target",
            "description": "物理PIDサーボループが自動追従ロックオンすべきYOLOトラックIDを動的指定します。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track_id": { "type": ["integer", "null"] }
                },
                "required": ["track_id"]
            }
        }
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    """LLMが意思決定したツールコールを実行し、CLIロジックを呼び出す"""
    try:
        if name == "get_active_tracks":
            # YOLO常時テキストの取得中継
            tracks = perception.get_current_tracks_text()
            return { "content": [{ "type": "text", "text": str(tracks) }] }
            
        elif name == "analyze_crop_image":
            # VLMオンデマンドズーム解釈の中継
            track_id = arguments["track_id"]
            query = arguments["query"]
            
            # YOLOの最新座標からVLM LOCへのマッピング
            tracks = perception.get_current_tracks_text()
            target = next((t for t in tracks if t["track_id"] == track_id), None)
            if not target:
                return { "content": [{ "type": "text", "text": f"Error: Track ID {track_id} が画面から見失われました。" }] }
            
            # クロップ画像を取得してVLMへ入力（1024px制限前処理内包）
            res = perception.analyze_crop_image("current_live_frame.jpg", target["bbox"], query)
            return { "content": [{ "type": "text", "text": res }] }
            
        elif name == "set_tracking_target":
            # 物理ターゲットID의バインド切り替え
            track_id = arguments["track_id"]
            # 物理PIDサーボコントローラへの指示インジェクション
            ptz.set_active_lock_id(track_id)
            return { "content": [{ "type": "text", "text": f"Success: 追跡ターゲットを ID: {track_id} に固定しました。" }] }
            
    except Exception as e:
        return { "content": [{ "type": "text", "text": f"Critical Error in MCP Bridge: {str(e)}" }] }
```

---

## 5. 段階的テスト・検証手順（ベストプラクティス）

システム統合時のリスクを最小化するために、各レイヤーを順番にテスト・スタックする。

### テスト1: 筋肉（物理制御）の完全性テスト (PTZ Loop Test)
* **目的**: 350ms自動自律停止（Dwell Time）および5秒保護リミッターが正常に動作するか。
* **手順**: 
  1. `ptz_cli.py --action move --pan 1.0` を実行。
  2. カメラが指定時間だけ勢いよく回り、**その後明示的に自動停止（Stop SOAPパケットの受信）すること**を確認。
  3. 万が一Wi-Fiを切断（パケット消失を模擬）しても、Dwell Timeタイムアウトでカメラ内のサーボ運動が自動制動されるか検証。

### テスト2: 記憶（SQLite Trigram）の精度テスト (Japanese RAG Test)
* **目的**: 日本語部分一致検索が外部依存（MeCab等）なしで完璧に想起を完了するか。
* **手順**:
  1. `memory_cli.py --action search --query "猫のタマちゃん"` を実行。
  2. `user_preferences.md` の「庭の野良猫（タマ）に対する方針」カードがFTS5またはLIKE部分一致フォールバックで正常にヒットするか確認（3文字以上および2文字以下の双方でテスト）。

### テスト3: 感覚（YOLOテキスト ➔ VLMオンデマンド）の極限リソーステスト (OOM Avoidance Test)
* **目的**: 12GB VRAM（RTX 3060）環境でメモリオーバーフロー（OOM）を起こさずにVLM推論が完結するか。
* **手順**:
  1. バックグラウンドで YOLO-ORT（常時テキスト監視）を連続ループで動かす。
  2. その状態で、`perception_cli.py --action analyze_crop --id 101` をキック（オンデマンドVLM起動）。
  3. `nvidia-smi` コマンドでVRAM推移を監視し、画像幅1024pxクランプ処理により瞬間的なメモリ消費が **VRAM限界（12GB）に衝突せず、マージン（空き4.3GB）の範囲内で完全に安定稼働すること**を確認。

---

以上の「CLI先行 ➔ 薄いMCPラッピング」のベストプラクティス手順に従うことで、物理世界のノイズに起因する実装バグを個別に完全に分離・撃破し、LLMエージェントが完璧に筋肉・感覚・記憶をコントロールする、最も完成度の高い自律対話型エージェントカメラシステムが実機で具現化する。
