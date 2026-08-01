# pico.cli.memory (memory-cli) 詳細設計仕様書 & 運用設計指針

本仕様書は、エッジAIカメラシステム（Pico）における長期記憶システム **`pico.cli.memory (memory-cli)`** および中核モジュール **`pico.memory.MemoryStore`** の設計理念、データ構造、CLI/API仕様、および記憶蓄積・更新に関する運用指針を定義するドキュメントである。

---

## 1. 概要とシステム位置づけ

### 1.1 エッジAIにおける記憶の課題と本システムの役割
一般的なマルチモーダルAI/LLMエージェントシステムでは、過去の観察事実や知識を保存するために重いベクトルデータベース（ChromaDB, Qdrant等）や常時稼働する埋め込みモデル（Embedding Model）が用いられる。
しかし、エッジPC環境（VRAM 8GB〜12GB）においてベクトルDBを常駐させることは、VRAM枯渇（OOM）やレスポンス遅延の原因となる。

本システムでは以下の設計アプローチを採用する：
1. **追加VRAMゼロの長期記憶基盤**: SQLite 3.34+ に標準搭載されている **FTS5 Trigram トークナイザー (`tokenize='trigram'`)** を採用し、追加VRAM「ゼロ」、メモリ消費数MB以下で高速な日本語全文検索・想起を実現する。
2. **可読性の高い人間・AI共用フォーマット**: 記憶の永続化ファイルとして **OKF (Obsidian Knowledge Format / YAML Frontmatter付き Markdown)** を採用。Gitでの差分管理や直接編集が容易であり、Obsidian等の知識管理ツールのナレッジベースとしても利用できる。

```
                       ┌──────────────────────────────────────────────┐
                       │        LLMエージェント（LangGraph 1.0）       │
                       └──────────────┬──────────────┬────────────────┘
                                      │              │
         ┌────────────────────────────┘              └────────────────────────────┐
         ▼ (Tool: search_wiki)                                                    ▼ (Tool: write_wiki)
┌─────────────────────────────────┐                                      ┌─────────────────────────────────┐
│ pico.cli.memory (memory-cli)    │                                      │ pico.cli.memory (memory-cli)    │
│ --action search                 │                                      │ --action write                  │
└────────────────┬────────────────┘                                      └────────────────┬────────────────┘
                 │ (2段階検索)                                                            │ (ファイル生成 & DB同期)
                 ▼                                                                        ▼
┌─────────────────────────────────┐                                      ┌─────────────────────────────────┐
│ SQLite 3.34+ (wiki.db)          │                                      │ Markdown ファイル (wiki/*.md)   │
│  - wiki_fts (FTS5 Trigram)      │                                      │  - YAML Frontmatter (OKF)       │
│  - wiki_metadata (本体)         │                                      │  - 観測記録セクション (追記)     │
└─────────────────────────────────┘                                      └─────────────────────────────────┘
```

---

## 2. 2層記憶アーキテクチャ（短期知覚と長期記憶の分離）

本システムでは、リアルタイム知覚の「使い捨てID」と、長期的・持続的な「実世界の識別子」を明確に分離して管理する。

### 2.1 短期的知覚バッファ (`track_id`) の限界と割り切り
YOLOおよびByteTrack/IoU Trackerが発行する `track_id`（例: `105`, `106`）は、カメラの旋回・ズーム、物陰への遮蔽、一時的な検出ロストによって頻繁に変化（カウントアップ）する。
したがって、**`track_id` は数秒間〜数十秒間の「局所的な使い捨てセッションポインタ（画面上の指定用ハンドル）」**としてのみ使用し、長期記憶の主キーとしては一切使用しない。

### 2.2 長期記憶 (OKF Wiki) における3大永続キー
`track_id` が変化しても、エージェントが「過去に見た同じ物体・場所・人物」と認識（Re-ID）できるように、長期記憶では以下のアンカーを組み合わせて識別する：

1. **物理絶対座標 ＋ 空間ゾーン (Spatial Anchors)**
   - カメラの ONVIF PTZ 物理角度（Pan / Tilt / Zoom）から導出される空間ゾーン（例: `Zone A (玄関アプローチ)`、`Zone B (庭)`）。
   - 「Pan +0.5 / Tilt -0.2 に存在する物体」として特定。
2. **意味的視覚特徴 (Semantic Anchors)**
   - VLMが解釈したテキスト特徴（例: `黄色い充電バッテリー`, `赤首輪の白茶トラ`）。
   - 単に `cat` や `package` というYOLOラベルだけでなく、VLMによる視覚的詳細記述と照合。
3. **人間・システムによる固定名 (Persistent Identity)**
   - OKFファイル名やタイトルとして定着した識別名（例: `wiki/known_objects_tama.md`, `wiki/zone_a_package.md`）。

---

## 3. 記憶・記録すべき4大対象カテゴリ

カメラエージェントが自律的かつ適応的に動作するために、`memory-cli` を通じて以下の4つのカテゴリの知識を蓄積する。

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       記憶・記録すべき 6 大カテゴリ                                      │
├──────────────────────────┬──────────────────────────┬──────────────────────────┬───────────────────────┤
│ 1. 物体・場所の意味同定  │ 2. 時間・環境ノイズ・    │ 3. 人物・ペットの        │ 4. 物理制御・空間制限 │
│    (Identity & Location) │    コンテキストルール    │    プロファイル・指示    │    の自己学習 (Limits)│
├──────────────────────────┴──────────────────────────┼──────────────────────────┴───────────────────────┤
│ 5. ユーザー対話・会話インサイト                     │ 6. 外部検索・調査知見 (External Research Synthesis)│
│    (Conversational Insights & Standing Orders)      │    (Web検索 / ドキュメント照会 / 技術ナレッジ)       │
└─────────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

### 3.1 物体・場所の意味同定 (Static & Semi-Static Identity)
- **対象**: 一時的ID (`track_id`) を超えて実世界に存在する特定物体や所有物、空間領域。
- **具体例**:
  - 「駐車場Aにある黒のプリウス」
  - 「玄関前に置かれた黄色の工具バッテリー」
- **効果**: 次回以降、同領域で検出された際に無駄なVLM起動をスキップし、YOLO＋位置記憶想起だけで「既知の物体」と高速同定できる。

### 3.2 時間・環境ノイズ・コンテキストルール (Environmental Context & Noise Rules)
- **対象**: 逆光、影、風による誤検知のパターンや、時間帯ごとの行動ルール。
- **具体例**:
  - 「15:00〜17:00 は西日により窓際に強い影ができ、YOLOが `person` と誤認しやすい」
  - 「強風時は庭の木が揺れ、誤検知が発生する」
  - 「夜間 22:00〜07:00 は警戒モード。Zone A での検知は即座にPTZ追跡とアラート発信」
- **効果**: LLMエージェントが「環境の癖」を理解し、誤アラートを大幅に低減する。

### 3.3 人物・ペットのプロファイル・指示 (Entity Profiles & Directives)
- **対象**: 登場人物や動物ごとの個別ルール、ユーザーからの明示的な指示。
- **具体例**:
  - 「飼い猫のタマちゃん（白茶トラ、赤首輪）：追従時はカメラ移動速度を緩やか (`0.05`) にし、ライト・音での威嚇は禁止」
  - 「宅配業者：玄関前に10秒以上滞在しても警告しない」
- **効果**: 対象の性格や役割に応じた安全で適切な振る舞い（トーン＆マナー）を実現。

### 3.4 物理制御・空間制限の自己学習 (Physical & Spatial Constraints)
- **対象**: カメラの構造的死角、限界角度、PIDゲインの個別特性。
- **具体例**:
  - 「Pan > +1.40 以上の範囲は壁になり視界が遮られる」
  - 「Tilt < -0.85 付近はカメラ台座自身が映り込むためズーム対象外とする」
- **効果**: サーボモーターの空回り・壁への衝突摩耗を防ぎ、ハードウェア寿命を延ばす。

### 3.5 ユーザー対話・会話インサイト (Conversational Insights & Standing Orders)
- **対象**: チャットや音声対話の中でユーザーが発言した雑談・好み・スケジュール・家族構成・定常的な命令。
- **具体例**:
  - 「たかしさんは毎週木曜日の21:00に帰宅する」
  - 「置き配のダンボールは開けずに玄関横に置いておいてほしい」
  - 「来週まで庭で工事が行われており、見慣れない人物が行き来する予定である」
- **効果**: 会話履歴（Context Window）が流れて忘却されるのを防ぎ、エージェントが永続的なパーソナライズ文脈を保持できる。
- **Provenance指定**: `provenance_source: user_dialogue` / `doc_type: conversation_insight`

### 3.6 外部検索・調査知見 (External Research Synthesis)
- **対象**: LLMがWeb検索ツールやドキュメント参照ツール、Wikipedia等を用いて調査・解決した外部知識や機器マニュアル情報。
- **具体例**:
  - 「Tapo C210 の ONVIF 応答ポートは 2020、RTSP パスは `/stream1` である」
  - 「特定製品の警告マークの意味や取扱注意点」
- **効果**: 同一の外部検索を何度も実行するAPIコストや通信遅延を削り、ローカルWikiから即座に想起できるようにする。
- **Provenance指定**: `provenance_source: web_search` または `doc_type: external_knowledge`

---

## 4. データ構造と OKF (Obsidian Knowledge Format) 仕様

### 4.1 SQLite データベース構造 (`wiki.db`)

`pico.memory.MemoryStore` は内部で 2 つの SQLite テーブルを使用する。

#### 本体テーブル (`wiki_metadata`)
ドキュメントの全内容および管理属性を格納する。

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

#### FTS5 仮想テーブル (`wiki_fts`)
C言語レベルの高速 Trigram インデックス。

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    filepath,
    doc_type,
    title,
    tags,
    content,
    tokenize='trigram'
);
```

#### 相互リンク・リレーションテーブル (`wiki_links`)
本文から抽出された Obsidian 形式 WikiLink (`[[...]]`) の項目間リンクを保持する。

```sql
CREATE TABLE IF NOT EXISTS wiki_links (
    source_filepath TEXT,
    target_title TEXT,
    relation_type TEXT DEFAULT 'wikilink',
    PRIMARY KEY (source_filepath, target_title)
);
```

#### エイリアス・名寄せテーブル (`wiki_aliases`)
表記の揺らぎ（同義語・別名）を吸収し、正規のファイルパスへ一括変換する。

```sql
CREATE TABLE IF NOT EXISTS wiki_aliases (
    alias_name TEXT PRIMARY KEY,
    target_filepath TEXT
);
```

### 4.2 2段階フォールバック想起ロジック (Search Algorithm)

`MemoryStore.search(query, limit)` では、日本語の形態素・単語分割に依存せず確実な想起を行うため、2段階の検索アルゴリズムを実装している。

1. **Phase 1: FTS5 Trigram 高速インデックス検索**
   - クエリ文字列から3文字以上の単語を抽出し、`wiki_fts MATCH '..."word"...'` を実行。
   - `bm25(wiki_fts)` スコアに基づいて結果を昇順（関連度の高い順）にソート。
2. **Phase 2: LIKE 句部分一致フォールバック**
   - 2文字以下の単語（例: 「猫」「影」）または FTS5 検索で結果が 0 件だった場合、`LIKE '%query%'` 検索へ自動的にフォールバック。

### 4.3 OKF (Obsidian Knowledge Format) Markdown 仕様

`write` アクションによって出力・更新される Markdown ファイルの標準フォーマット。

```markdown
---
title: "飼い猫タマちゃんの識別と追従ルール"
tags: "pet cat profile rule"
doc_type: "knowledge"
provenance_source: "CLI memory write"
provenance_confidence: "High"
---

# 飼い猫：タマちゃん (Tama)

### 📌 基本識別特徴
- **外観**: 白茶トラの猫（赤色首輪あり）
- **主な出現エリア**: 庭 (Zone B)、テラス

### ⚙️ 行動ルール・ポリシー
- **警戒モード**: **無効 (Do Not Alarm)**
- **PTZ制御指針**: 追従速度を低速 (`0.05`) に制限。威嚇は絶対禁止。

### 🕒 観測記録: 2026-07-24 17:35:00
夕方の庭巡回パターンを確認。VLM解析により「タマちゃん」と同定。
```

---

## 5. CLI および Python API インターフェース仕様

### 5.1 CLI コマンド仕様 (`pico.cli.memory` / `memory-cli`)

#### 記憶の検索 (Search)
```powershell
# uv 経由で実行する場合
uv run memory-cli --action search --query "猫のタマちゃん" --limit 3

# モジュールとして実行する場合
python -m pico.cli.memory --action search --query "西日 影"
```
- **引数**:
  - `--action`: `search` (必須)
  - `--query`: 検索クエリ文字列 (必須)
  - `--limit`: 最大取得件数 (デフォルト: 3)
  - `--db`: SQLite DB ファイルパス (デフォルト: `wiki.db`)
- **出力**: JSON 形式の想起結果リスト

#### 記憶の書き込み・更新 (Write)
```powershell
uv run memory-cli --action write `
  --file "wiki/known_objects_tama.md" `
  --title "飼い猫タマちゃんの識別と追従ルール" `
  --content "夕方はタマちゃんに警告音を鳴らさずに話しかける。" `
  --tags "pet cat profile"
```
- **引数**:
  - `--action`: `write` (必須)
  - `--file`: 書き込み対象の Markdown ファイルパス (必須)
  - `--title`: 記憶のタイトル (必須)
  - `--content`: 記録する本文・観測内容 (必須)
  - `--tags`: 空白区切りのタグ文字列 (任意)
- **動作**:
  1. ファイルが存在しない場合: 指定された OKF Frontmatter を持つ新しい Markdown ファイルを生成。
  2. ファイルが既に存在する場合: 既存の Frontmatter を維持しつつ、末尾に `### 🕒 観測記録: <タイムスタンプ>` セクションを自動追記。
  3. `wiki_metadata` および `wiki_fts` のインデックスを自動更新。

### 5.2 Python API 使用例 (`pico.memory.MemoryStore`)

```python
from pico.memory import MemoryStore

# データベースの初期化・接続
store = MemoryStore(db_path="wiki.db")

# 1. 記憶の書き込み・インデックス登録
store.add_entry(
    filepath="wiki/environmental_rules.md",
    doc_type="knowledge",
    title="西日による誤検知ルール",
    tags="environment noise rules",
    content="15:00-17:00は窓際に強い影ができ人影と誤認しやすい。",
    provenance_source="VLM_analysis",
    provenance_confidence="high"
)

# 2. 記憶の想起検索
results = store.search(query="西日の影", limit=3)
for r in results:
    print(f"Title: {r['title']}, File: {r['filepath']}, Score: {r['score']}")

# 3. 終了処理
store.close()
```

---

## 6. 自律更新トリガーとライフサイクル設計

LLMエージェントが運用の中で自律的に `memory-cli` を呼び出すタイミングと更新フローを以下のように設計する。

```
[ イベント発生 ]
   │
   ├── (1) 仮説検証完了時 ───► YOLO低確信度 ➔ PTZズーム ➔ VLM解釈 ➔ 【自動 write】
   │
   ├── (2) ユーザー対話時 ───► 「夕方は静かにして」「あれはタマちゃんだよ」 ➔ 【会話ノードから write】
   │
   └── (3) 誤検知確定時   ───► 人間/VLMが誤判定を確認 ➔ 環境ルールファイルへ 【追記 write】
```

### 6.1 実 LLM (Ollama gemma4:e2b) による自律更新・想起の実証実績

本システムでは、実際のローカル LLM (Ollama `gemma4:e2b`) を用いて、マルチターン対話・カメライベント発生時の自律ツール呼び出し（Function Calling）および相互リンク生成の E2E 実動テストを実施し、以下の自律動作を実証済みである (`scratch/test_real_llm_agent_loop.py`)：

1. **ユーザー指示からの自律 write_wiki 発行**:
   - ユーザー発言「夕方18時前後に庭（Zone B）にやってくる猫はタマちゃんだよ」に対し、LLM が自律的に `write_wiki` を判定・実行し、`wiki/known_objects_tama.md` に OKF ナレッジとして保存。
2. **観察イベントからの自律 search_wiki 想起**:
   - カメラからの検出通知（17:50 庭で猫検出）に対し、LLM が自発的に `search_wiki(query="タマちゃんに関するルール")` をキックして過去の規則をミリ秒想起。
3. **WikiLinks `[[...]]` による相互結合の自動生成**:
   - 観察ログ記録時に LLM が `[[飼い猫タマちゃん]]` の相互リンクを記述し、ナレッジグラフ上のバックリンク関係を完全接続。

---

## 7. まとめ

`pico.cli.memory (memory-cli)` は、軽量かつVRAM消費ゼロの SQLite FTS5 Trigram と可読性に優れた OKF Markdown、および WikiLinks (`[[...]]`)・バックリンク（被参照）自動追跡構造を融合させることで、エッジ環境における効率的な長期記憶ナレッジグラフを実現している。

`track_id` などの一時的知覚情報と、空間座標・視覚意味特徴に基づく永続的な記憶情報を明確に分離して運用することで、IDが頻繁に変わる状況下でもブレない知識ベースを維持・成長させることが可能となる。
