# 能動知覚追跡システム V2.2 実装計画書
## MCPサーバー連携とクリーン画像解析・Wikiコンソリデーションの安定化

本ドキュメントは、LangGraphによる緊密な結合を排除し、感覚・筋肉・記憶をそれぞれ「CLI / MCP ツール」として再構築した V2.1 新アーキテクチャを受け、次のフェーズとして実装する **V2.2 (MCP自律協調動作の最適化および堅牢化) の実装計画** をまとめたものである。

---

## 📋 V2.2 で解決する課題

### 1. YOLO/アノテーション描画による VLM 誤認の完全排除
- **課題:** カメラモニター画面に YOLO の検出枠（グリーン矩形）やラベル、および PID 用のマーカー線を描画した後のフレームを VLM に送ると、VLMが「画像に緑の枠線や測定用の線が描かれており、サイズ測定プロセスが実行されている」と誤認識する。
- **対策:** YOLOのループ上部で描画が加わる前の「完全なオリジナルクリーンフレーム」をメモリ上（または共有バッファ）に確実にコピー・退避し、VLMの画像クロップ（感覚CLI）にはこの汚染されていないクリーンフレームのみを供給する。

### 2. Ollama への並列リクエストに伴うハング・ReadTimeout の完全防止
- **課題:** 外部 LLM から MCP サーバー経由で複数のツール（状況認識やプラン決定のための LLM テキスト推論、および VLM クロップ画像解析）がほぼ同時に呼び出された際、Ollama のリソース確保が衝突してフリーズし、ReadTimeout が発生する。
- **対策:** 
  - MCP サーバーのツールハンドラ層（または Ollama クライアントの内部）に**セマフォまたはスレッドロックによる直列化ロック（Serialization Lock）**を実装する。
  - 重い VLM 画像解析（`analyze_crop_image`）を実行した直後には、Ollama が VRAM 領域のクリーンアップを行えるよう、**強制的なクールダウン待機時間（1.5秒〜2.0秒）**を非同期でバインドする。

### 3. 一時的なトラッカーIDによる Wiki ファイルの無限増殖防止とログ化
- **課題:** 人物が通りかかるたびに `auto_person_16.md` や `auto_person_18.md` のような一時的トラッカーIDに基づいた Wiki ファイルが生成され、`wiki/` フォルダがゴミファイルで無限に増殖してしまう。
- **対策:**
  - `memory-cli` および MCP ツール `store_memory` (または `write_wiki`) は、一時的な ID 単独のタイトルでの新規作成を禁止する。
  - 代わりに、`observed_people` (観測された人物) や `observed_objects` (観測された物体) などの**カテゴリごとに統合された永続ファイル名**のみを受け入れ、既存ファイルが存在する場合は Frontmatter を維持したまま末尾にタイムスタンプ（`### 🕒 観測記録: YYYY-MM-DD HH:MM:SS`）付きで時系列ログとして自動追記するコンソリデーション機構を標準化する。

---

## 🛠️ 具体的な変更コードとデータフロー設計

### A. クリーン画像とVLMのデータフロー設計
`scratch/run_live_system.py` などのライブ追従ループから、YOLOの描画が加わる前の `frame_clean` を安全に感覚モジュール（`OnDemandPerceptionCLI`）へ引き渡す。

```
[ カメラ RTSP 映像受信 ]
         │
         ├──► 【オリジナルクリーンフレーム】 ──► [感覚モジュール (Perception) / クロップ用]
         │                                              │ (YOLO描画のないクリーンな人物切出)
         ▼                                              ▼
[ YOLO検出・ByteTrack ]                             [ Ollama VLM への送信 ]
         │
         ▼
[ BBox・PIDライン描画 ]
         │
         ▼
[ OpenCV モニター画面表示 ]
```

### B. MCP サーバー側での排他制御（Ollama Lock）
MCP サーバー内の呼び出しに `asyncio.Lock` をバインドし、LLM推論要求とVLM解析要求が並列で実行されるのをサーバー側で防ぐ。

```python
# pico/mcp/server.py (またはそれに準ずるモジュール) でのロック実装イメージ
import asyncio

ollama_lock = asyncio.Lock()

async def handle_analyze_crop(track_id: int, query: str):
    async with ollama_lock:
        # 1. 画像切り出しと VLM 呼び出し
        result = await perception.analyze_crop(track_id, query)
        # 2. クールダウン待機
        await asyncio.sleep(1.5)
        return result
```

### C. Wiki カテゴリ統合追記 (Consolidation) ロジックの組み込み
`memory-cli` の書き込み機能、および MCP の `search_wiki` / `write_wiki` において、Frontmatter（メタデータ）の YAML 構造を破壊せずに本文末尾に追記するパーサーを実装する。

---

## 🧪 検証計画（Definition of Done）

1. **自動テストの確認**:
   - `uv run pytest` を実行し、既存のテストケース（検出器、PTZ、PID等）がすべてパスすること。
2. **クリーン画像の確認**:
   - `wiki/auto_observed_people.md` に自動記録された VLM の解析内容に「緑の枠線」や「測定マーキング」といったアノテーションへの言及が一切含まれず、人物の実際の衣服や特徴、動作だけが正確に記述されていること。
3. **Wiki の肥大化テスト**:
   - 人物がカメラの前を繰り返し通過し（異なるIDが割り振られた場合でも）、`wiki/` 内に新規のファイルが作られず、`wiki/auto_observed_people.md` という単一ファイルに時系列で観測情報が自動蓄積されていくこと。
4. **Ollama 負荷テスト**:
   - MCPツール経由での LLM/VLM 解析を連続で5回以上実行した際、`ReadTimeout` によるエラー終了が発生しないこと。
