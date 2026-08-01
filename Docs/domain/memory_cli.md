---
type: Concept
title: Memory CLI Specification
description: Pico-Argus システムにおける長期記憶(FTS5 Trigram, WikiLinks)・SQLiteスキーマおよびCLI・Python API仕様
status: active
timestamp: 2026-08-01T10:30:00+09:00
tags:
  - memory
  - cli
  - sqlite
  - knowledge
  - fts5
sources:
  - id: memory_cli_doc
    resource: /Docs/raw/memory_cli_specification.md
    title: Memory CLI Specification Document
---

# Memory CLI Specification

## 1. 概要とシステム位置づけ

`memory` モジュールは、追加VRAM「ゼロ」の長期記憶基盤として SQLite 3.34+ の **FTS5 Trigram トークナイザー (`tokenize='trigram'`)** を採用し、人間・AI共用フォーマットとして **OKF (Obsidian Knowledge Format)** YAML Frontmatter付き Markdown ファイル（`wiki/*.md`）を用いて永続化・高速日本語想起を実現します。

## 2. 2層記憶アーキテクチャ（短期知覚と長期記憶の分離）

YOLO/ByteTrackが発行する `track_id` は頻繁にロスト・変化するため、**数秒〜数十秒間の「局所的な使い捨てセッションポインタ」**として割り切り、長期記憶では以下の 3 大アンカーを組み合わせて実世界の同一対象を認識（Re-ID）します。
1. **物理絶対座標 ＋ 空間ゾーン (Spatial Anchors)**: ONVIF PTZ 角度から導出した領域（例: `Zone B (庭)`）。
2. **意味的視覚特徴 (Semantic Anchors)**: VLMが解釈した詳細テキスト（例: `白茶トラ、赤首輪`）。
3. **人間・システムによる固定名 (Persistent Identity)**: 正規のファイル名・タイトル（例: `known_objects_tama.md`）。

## 3. 記憶・記録すべき 6 大カテゴリ

1. **物体・場所の意味同定 (Identity & Location)**: 既知物体の領域・特徴保存。
2. **時間・環境ノイズ・コンテキストルール (Context & Noise Rules)**: 「西日による誤認影」などの誤検知低減パターン。
3. **人物・ペットのプロファイル・指示 (Entity Profiles & Directives)**: 飼い猫や特定の登場人物、ユーザーからの指示。
4. **物理制御・空間制限の自己学習 (Physical & Spatial Constraints)**: カメラの構造的死角、限界角度。
5. **ユーザー対話・会話インサイト (Conversational Insights)**: 会話履歴から抽出したユーザーの好み、帰宅時間、永続スケジュール等のパーソナライズ文脈。
6. **外部検索・調査知見 (External Research Synthesis)**: Web検索やマニュアル照会等でLLMが自己解決した外部知識（重複検索のAPIコスト削減）。

## 4. SQLite データ構造 (`wiki.db`)

内部では以下の 4 つのテーブルを利用します：

### 本体テーブル (`wiki_metadata`)
```sql
CREATE TABLE IF NOT EXISTS wiki_metadata (
    filepath TEXT PRIMARY KEY,
    doc_type TEXT,
    title TEXT,
    tags TEXT,
    content TEXT,
    last_reviewed TEXT,
    provenance_source TEXT,
    provenance_confidence TEXT
);
```

### FTS5 仮想テーブル (`wiki_fts`)
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    filepath, doc_type, title, tags, content, tokenize='trigram'
);
```

### 相互リンク・リレーションテーブル (`wiki_links`)
本文内の `[[項目名]]`（WikiLink）を自動抽出してバックリンク（被参照）関係を構成します。
```sql
CREATE TABLE IF NOT EXISTS wiki_links (
    source_filepath TEXT,
    target_title TEXT,
    relation_type TEXT DEFAULT 'wikilink',
    PRIMARY KEY (source_filepath, target_title)
);
```

### エイリアス・名寄せテーブル (`wiki_aliases`)
```sql
CREATE TABLE IF NOT EXISTS wiki_aliases (
    alias_name TEXT PRIMARY KEY,
    target_filepath TEXT
);
```

### 2段階フォールバック想起ロジック (Search Algorithm)
1. **Phase 1: Trigram FTS5 検索**: 3文字以上の単語を抽出し `MATCH` 検索。BM25 スコアでソート。
2. **Phase 2: LIKE 句部分一致フォールバック**: 2文字以下の単語（「猫」「影」）や FTS5 検索 0 件時に LIKE 句検索へ非同期フォールバック。

## 5. CLI および Python API インターフェース仕様

### 5.1 CLI コマンド仕様

#### 記憶の検索 (Search)
```powershell
uv run memory-cli --action search --query "猫のタマちゃん" --limit 3
```

#### 記憶の書き込み・更新 (Write)
```powershell
uv run memory-cli --action write \
  --file "wiki/known_objects_tama.md" \
  --title "飼い猫タマちゃんの識別と追従ルール" \
  --content "夕方は [[タマちゃん]] に警告音を鳴らさずに話しかける。" \
  --tags "pet cat profile"
```
※ファイルが既に存在する場合は、既存の Frontmatter を維持しながら末尾に自動でタイムスタンプ付きで観測セクションを追記します。

### 5.2 Python API 使用例 (`pico.memory.MemoryStore`)
```python
from pico.memory import MemoryStore

store = MemoryStore(db_path="wiki.db")
# 記憶の追加
store.add_entry(
    filepath="wiki/environmental_rules.md",
    doc_type="knowledge",
    title="西日の影ルール",
    tags="environment noise rules",
    content="15:00-17:00は窓際に強い影ができやすい。"
)
# 記憶の想起検索
results = store.search(query="西日の影", limit=3)
store.close()
```

## 6. 自律更新トリガーとライフサイクル設計

LLMエージェント（Ollama等）は以下のタイミングで `memory-cli` を自律コールします。
- **仮説検証完了時**: 低確信度物体 ➔ 物理追従ズーム ➔ VLM解釈 ➔ 自動保存
- **ユーザー対話時**: ユーザーからの「これは〜だよ」などの指摘・指示 ➔ 会話ノードから保存
- **誤検知確定時**: 誤検知パターンの蓄積

## 7. 関連概念

* [MCP Usecases](./mcp_usecases.md) - メモリ検索・保存を活用するユースケース
