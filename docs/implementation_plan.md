# 詳細実装計画: カメラ自律追跡エージェント

> **前提**: [camera_agent.md](camera_agent.md) はコンセプトレベルの設計文書。本計画は現実のコードベース・環境に基づき、批判的に実現可能性を評価した上での段階的実装ロードマップです。

---

## 0. 実行環境の事実整理

| 項目 | 実態 | 設計書との差異 |
|:---|:---|:---|
| **OS** | Windows (開発) + WSL2 (Ollama) | 設計書は Ubuntu 24.04 前提 |
| **VLM推論** | WSL上の Ollama (`http://localhost:11434`) | 設計書は vLLM/llama.cpp を直接想定 |
| **カメラ** | Tapo C210 (ONVIF port 2020) | 一致 |
| **PTZ制御** | `RelativeMove` で動作確認済み | 設計書は `ContinuousMove` 前提だが、**ファームウェアにより未対応の可能性大** |
| **検出エンジン** | `cv2.dnn.readNetFromONNX` + yolov8s.onnx | 設計書は ORT + IO Binding |
| **Python** | 3.13+ (pyproject.toml) | 設計書は 3.12+ |
| **追跡** | 最大面積ヒューリスティック（IDなし） | 設計書は ByteTrack |
| **PID** | 線形P制御のみ | 設計書は VOR規範型適応PID |
| **キャリブレーション** | 実機済み（±1.08, ±0.85） | 設計書に言及なし |

> **ContinuousMove について**: 調査の結果、Tapo C210 のファームウェア 1.5.x 以降で `ContinuousMove` が動作しないケースが複数報告されている。**本計画では `RelativeMove` を継続使用**し、設計書の ContinuousMove は将来のカメラ変更時の選択肢として留保する。

> **Ollama 接続**: WSL2 の Ollama を Windows ホストから `http://localhost:11434` で呼び出す。WSL2 の最新版では localhost が自動ブリッジされるが、動作しない場合は `OLLAMA_HOST=0.0.0.0:11434` の設定が必要。`.env` に `OLLAMA_BASE_URL` を追加して設定を外出しにする。

---

## 1. 現在のアーキテクチャ（As-Is）

```
[Tapo C210 RTSP stream1]
        │
        ▼ cv2.VideoCapture
[RTSPVideoReader スレッド駆動]
        │
        ▼ frame
[cv2.dnn YOLOv8s.onnx]
        │
        ▼ BBox list
[最大面積選択 (ID追跡なし)]
        │
        ▼ error_x, error_y
[線形P制御 KP_X=0.18, KP_Y=0.16]
        │
        ▼ move_x, move_y
[RelativeMove キュー経由]
        │
        ▼ SOAP
[Tapo C210]
```

**問題点**:
- 全て同期ループ（`while True` + `cv2.waitKey`）
- 検出・追跡・制御が1つのスクリプトに密結合
- RTSPVideoReader が2ファイルに重複コピー
- ONVIF初期化コードが3ファイルに重複

---

## 2. 目標アーキテクチャ（To-Be）

```
┌─────────────────────────────────────────────────────────┐
│ 反射ループ (Reflex) — メインスレッド                       │
│                                                         │
│ [Tapo C210 RTSP stream2: 640x480]                       │
│         │ VideoCaptureThread                            │
│         ▼                                               │
│ [YoloDetector ORT / cv2.dnn]                            │
│         │ detections                                    │
│         ▼                                               │
│ [SimpleTracker IoU追跡]                                  │
│         │ track_id, cx, cy                              │
│         ▼                                               │
│ [AdaptivePIDController 非線形ゲイン]                      │
│         │ move_x, move_y                                │
│         ▼                                               │
│ [PTZController RelativeMove + 安全制御]                   │
│         │ SOAP                                          │
│         ▼                                               │
│ [Tapo C210]                                             │
└────────────────────┬────────────────────────────────────┘
                     │ イベント: 新規検出
                     ▼
┌────────────────────┴────────────────────────────────────┐
│ 認知ループ (Cognition) — asyncioタスク                    │
│                                                         │
│ [CognitionEngine]                                       │
│     │               │                                   │
│     ▼               ▼                                   │
│ [MemoryStore]   [Ollama WSL2 localhost:11434]            │
│ [SQLite FTS5]   [Qwen2.5-VL-3B]                         │
│     │               │                                   │
│     └───────┬───────┘                                   │
│             ▼                                           │
│ [ロックオン指示 / シーン要約ログ]                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 安全ガードレール (常時)                                   │
│ [GuardRails] ← フレーム監視 / 駆動時間監視                │
└─────────────────────────────────────────────────────────┘
```

---

## フェーズ 1: コードベース整理（リファクタリング）

### 目的
散在するスクリプトの重複コードをモジュール化し、テスト可能な構造を作る。**機能追加ゼロ**、既存動作を壊さない。

### 批判的リスク評価

| リスク | 深刻度 | 緩和策 |
|:---|:---|:---|
| リファクタで既存の動作が壊れる | 中 | 各ステップで `uv run pytest` を実行。手動で `tapo_yolo_tracking.py` の動作確認 |
| モジュール分割の設計が後工程と合わない | 低 | インターフェースを最小にし、後で変更しやすくする |
| `onvif-zeep` の import パスが環境依存 | 低 | テストでは ONVIF 呼び出しを Mock 化 |

### ファイル変更計画

#### [NEW] `src/pico/__init__.py`
- 空ファイル（パッケージ化）

#### [NEW] `src/pico/video_reader.py`
- `tapo_yolo_tracking.py` L136-L165 の `RTSPVideoReader` を移動
- 改善点:
  - `maxsize` 付きキューで最新フレームだけ保持（設計書 §1.2 の `maxsize=3` を参考に、ただし `maxsize=1` で最新のみ）
  - `threading.Event` による安全な停止シグナル
  - フレームタイムスタンプの記録（後でガードレールに使う）

#### [NEW] `src/pico/onvif_client.py`
- ONVIF 初期化ロジックの統一（3ファイルに散在するコードを集約）
- `PTZController` クラス:
  - `relative_move(x, y)` — キュー経由の非同期送信（現在の `send_move_command` 相当）
  - `safe_move(x, y, current_pos, limits)` — 安全制限付き移動
  - `stop()` — 緊急停止

#### [NEW] `src/pico/config.py`
- 既存の `camera_config.py` を移動
- `.env` + `tapo_config.json` の読み込みを統合
- 新規: `OLLAMA_BASE_URL` の定義

#### [MODIFY] `tapo_yolo_tracking.py`
- 上記モジュールを import して使うように書き換え
- 既存の動作ロジックは維持（P制御のまま）

#### [NEW] `tests/test_video_reader.py`
- フレーム取得のモックテスト

#### [NEW] `tests/test_onvif_client.py`
- `safe_move` の境界値テスト（可動限界超過時のクランプ動作）

### 完了判定
- [ ] `uv run pytest` 全テスト通過
- [ ] `uv run ruff check .` エラーゼロ
- [ ] `tapo_yolo_tracking.py` が新モジュールを使って既存と同じ動作をする
- [ ] `RTSPVideoReader` の重複が解消されている

---

## フェーズ 2: 追従制御の改善（適応PID）

### 目的
線形P制御を適応PIDに置き換え、追従精度と安定性を向上させる。

### 批判的リスク評価

| リスク | 深刻度 | 詳細 | 緩和策 |
|:---|:---|:---|:---|
| **PIDパラメータの初期値が不適切** | 高 | 設計書の `kp_base=0.5, ki=0.05, kd=0.01` は理論値であり、Tapo C210 の物理特性（SOAP通信遅延 100-200ms、モーターの慣性）と合わない可能性がある | Phase 2 完了後に実機チューニング用スクリプトを作成。まず P 制御のみで動かし、I/D を段階的に追加 |
| **積分ワインドアップ** | 高 | ターゲットロスト中に積分値が蓄積→再検知時にカメラ暴走 | アンチワインドアップ（クランプ + ロスト時リセット）を**初期実装に必須で含める** |
| **RelativeMove と PID の相性** | 中 | `RelativeMove` は「相対位置」指令であり `ContinuousMove`（速度指令）とは制御モデルが異なる。設計書のPID出力は速度 `v(t)` だが、RelativeMove に渡すのは位置差分 `Δx` | PID出力を「移動量」として解釈し、出力をステップ幅として使う。微分項はステップ間の誤差変化率で近似 |
| **不感帯の設定ミス** | 低 | 小さすぎるとハンチング、大きすぎると追従しない | 設定ファイル化して実機で調整可能に |

> **RelativeMove と ContinuousMove の本質的差異**: 設計書の PID は速度制御（ContinuousMove用）だが、実際に使うのは位置制御（RelativeMove）。これは**根本的な制御モデルの違い**であり、設計書のコードをそのまま使えない。
>
> **対応**: PID出力を「1ステップあたりの移動量」として解釈する。比例ゲインの非線形化（VOR規範）は有効だが、積分項・微分項の時定数は RelativeMove の送信間隔（0.45秒）に合わせて再設計が必要。

### ファイル変更計画

#### [NEW] `src/pico/pid_controller.py`

```python
class AdaptivePIDController:
    """VOR規範型非線形ゲインを持つ適応PID制御器。

    設計書§4のコンセプトをRelativeMove向けに再解釈。
    - 出力は速度ではなく「1ステップの移動量」
    - 実測dtを使用（固定tsではない）
    - アンチワインドアップ標準装備
    """
```

主要メソッド:
- `calculate_step(target_cx, target_cy, timestamp) -> (dx, dy)` — 正規化座標(0-1)から移動量を算出
- `reset()` — 積分値・前回誤差のリセット（ターゲットロスト時）
- `update_params(kp_base, ki, kd, ...)` — パラメータ動的変更

#### [NEW] `tests/test_pid_controller.py`
- 不感帯内→出力ゼロ
- 画面端→ゲイン最大化
- 積分項の蓄積制限
- ターゲットロスト→リセット動作
- 出力クリッピング
- 静止摩擦突破（最小速度ブースト）

#### [MODIFY] `tapo_yolo_tracking.py`
- `AdaptivePIDController` を使用するように置換
- 既存の `KP_X, KP_Y, MAX_STEP` を PID パラメータに統合

### 完了判定
- [ ] 単体テスト全パス
- [ ] 実機で追従動作が改善（主観的に滑らかさ向上）
- [ ] ハンチングが発生しないこと（不感帯内で静止）
- [ ] ターゲットロスト→再検知時にカメラが暴走しないこと

---

## フェーズ 3: 検出エンジンの強化

### 目的
1. `cv2.dnn` → `onnxruntime` への移行（推論速度向上）
2. 簡易IDトラッキングの導入（ByteTrackの完全実装ではなく、IoUベースの最小実装）

### 批判的リスク評価

| リスク | 深刻度 | 詳細 | 緩和策 |
|:---|:---|:---|:---|
| **ORT導入で依存が増える** | 中 | `onnxruntime` は大きなパッケージ。`onnxruntime-gpu` は CUDA バージョンとの互換性問題が頻発 | まず `onnxruntime`（CPU版）で移行。GPU版は後回し |
| **E2E NMS統合モデルの問題** | 中 | 設計書はE2E NMS前提だが、現在の `yolov8s.onnx` は標準エクスポート（NMS非統合）の可能性大。出力テンソルの形状が異なる | 現在のモデルの出力形状を確認してから実装方針を決定。NMS非統合なら後処理を Python で実装（現在の `cv2.dnn.NMSBoxes` と同等） |
| **ByteTrack の完全実装は過剰** | 高 | 設計書はByteTrackだが、現段階では「1人の人物を追い続ける」だけでよい。カルマンフィルタ＋ハンガリアンアルゴリズムは複雑すぎる | **IoUベースの簡易トラッカーを自前実装**。同一対象かどうかをフレーム間のBBox重複率で判定。IDはインクリメンタルカウンタで付与 |
| **推論速度がCPUで十分か** | 中 | ORT CPUでも `cv2.dnn` より高速だが、640x640入力で30ms以下を保証できるか不明 | ベンチマーク計測スクリプトを用意。必要なら yolov8n.onnx（nano版、既にリポジトリに存在）に切り替え |

### ファイル変更計画

#### [NEW] `src/pico/detector.py`

```python
class YoloDetector:
    """YOLOv8 ONNX推論エンジン。

    ORT / cv2.dnn を切り替え可能なアダプタパターン。
    設計書§2のコンセプトを参考にしつつ、実態に合わせて簡素化。
    - IO Binding は GPU 環境でのみ有効化（オプション）
    - 事前加温（pre-warm）は ORT 使用時のみ
    """
    def __init__(self, model_path: str, backend: str = "ort"):
        ...

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """フレームから検出結果を返す。"""
        ...
```

#### [NEW] `src/pico/tracker.py`

```python
class SimpleIoUTracker:
    """IoUベースの簡易オブジェクトトラッカー。

    設計書§2.2のByteTrackの代わりに、最小限のID追跡を実現。
    - フレーム間のBBox IoU > 閾値 → 同一ID
    - max_lost_frames を超えたら ID 解放
    - GMC補正なし（Tapo C210 の旋回速度では不要と判断）
    """
```

#### [MODIFY] `pyproject.toml`
- `onnxruntime>=1.24` を追加

#### [NEW] `tests/test_detector.py`
- ダミー入力での推論テスト（出力形状の確認）

#### [NEW] `tests/test_tracker.py`
- IoU計算の正確性
- ID付与と解放のロジック

#### [NEW] `scripts/benchmark_detector.py`
- ORT vs cv2.dnn の推論速度比較ベンチマーク

### 完了判定
- [ ] ORT推論が cv2.dnn と同等以上の速度
- [ ] 簡易トラッカーで同一人物にIDが付与される
- [ ] 人物がフレームアウト→再出現時に新IDが振られる
- [ ] 単体テスト全パス

---

## フェーズ 4: 認知ループ（Ollama VLM統合 + 記憶）

### 目的
反射ループ（YOLO+PID）と独立した非同期認知ループを構築。WSL上のOllamaを呼び出してシーン理解を行う。

### 批判的リスク評価

| リスク | 深刻度 | 詳細 | 緩和策 |
|:---|:---|:---|:---|
| **Ollama API 呼び出しのレイテンシ** | 高 | VLM推論は画像入力で2-5秒かかる。この間に反射ループが止まってはならない | **完全に非同期（asyncio + httpx）**。反射ループはメインスレッドの同期ループのまま、認知ループは別スレッドの asyncio イベントループで実行 |
| **WSL2 ネットワークの不安定性** | 中 | WSL2 の localhost ブリッジは不安定な場合あり。Ollama への接続がタイムアウトする可能性 | 接続テスト用のヘルスチェック関数を用意。タイムアウト5秒、3回リトライ。失敗時は認知ループを無効化し、反射ループだけで動作継続 |
| **画像のbase64エンコードコスト** | 低 | 640x480の JPEG を base64 変換する時間は数ms | JPEG品質を75に下げてサイズ削減 |
| **Epoch ステートマシンの実装複雑度** | 中 | asyncio.Lock + エポックカウンタの正しい実装は、デバッグが難しい | 設計書のコンセプトを参考にしつつ、状態変数は dataclass で管理。ログを詳細に出力して競合をトレース可能にする |
| **SQLite FTS5 Trigram の日本語精度** | 中 | Trigram は3文字単位なので、1-2文字の検索語には LIKE フォールバックが必要（設計書通り）。しかし LIKE は全件スキャンでデータ量次第で遅い | 初期データ量が少ないため問題にならない。1000件超えたら性能測定して対策 |
| **VLM のハルシネーション** | 高 | VLM が存在しない物体を「検出」したり、座標をでたらめに返す可能性。特に低照度・動体ブレ時 | VLM の出力は「提案」として扱い、YOLO の検出結果と照合。YOLO で検出されていない物体への追従指示は拒否する |

> **最大のリスク: 反射ループと認知ループの結合タイミング**
>
> 認知ループの VLM 推論は 2-5 秒かかる。この間にターゲットの状況は大きく変わる。設計書のEpoch機構はこの問題を「古い結論の破棄」で解決するが、**そもそも 2-5 秒前のフレームに基づく判断がどれだけ有用か**を実証する必要がある。
>
> **対策**: フェーズ4の初期段階では、VLM にリアルタイム判断を求めず、**「シーンの要約ログを残す」だけの受動的モード**から始める。有用性が確認できたらロックオン指示を段階的に有効化する。

### ファイル変更計画

#### [NEW] `src/pico/ollama_client.py`

```python
class OllamaVisionClient:
    """WSL2 上の Ollama REST API クライアント。

    - /api/chat エンドポイントを使用
    - 画像は base64 エンコードで送信
    - httpx.AsyncClient による非同期通信
    - タイムアウト・リトライ内蔵
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5-vl:3b"):
        ...

    async def analyze_scene(self, frame: np.ndarray, prompt: str) -> str:
        """フレーム画像をVLMに送信し、テキスト応答を返す。"""
        ...

    async def health_check(self) -> bool:
        """Ollama サーバーの接続確認。"""
        ...
```

#### [NEW] `src/pico/memory.py`

```python
class MemoryStore:
    """SQLite FTS5 Trigram による軽量記憶エンジン。

    設計書§5のコンセプトをベースに簡素化。
    - wiki_metadata + wiki_fts の2テーブル構成
    - search(): FTS5 → LIKE フォールバック
    - add_entry(): 新規記憶の追加
    """
```

#### [NEW] `src/pico/cognition.py`

```python
class CognitionEngine:
    """非同期認知ループエンジン。

    設計書§6のEpochステートマシンを簡素化して実装。
    - asyncio.Queue でイベント受信
    - エポックベースの競合排除
    - フェーズ4初期は「受動的ログモード」のみ
    """
    async def run(self):
        """認知ループのメインコルーチン。"""
        ...
```

#### [MODIFY] `.env.example`
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-vl:3b
```

#### [MODIFY] `pyproject.toml`
- `httpx>=0.28` を追加（非同期HTTP）

#### [NEW] `tests/test_ollama_client.py`
- モック API サーバーでの応答テスト
- タイムアウト時のフォールバック動作

#### [NEW] `tests/test_memory.py`
- FTS5 Trigram 検索の動作
- LIKE フォールバックの動作
- 日本語キーワードでの検索精度

#### [NEW] `tests/test_cognition.py`
- エポック整合性チェック（古い結果の破棄）
- Barge-In（割り込み）時のタスクキャンセル

### 完了判定
- [ ] Ollama ヘルスチェックが通る
- [ ] VLM にフレーム画像を送信→テキスト応答を受信できる
- [ ] 認知ループが反射ループをブロックしない（並行動作確認）
- [ ] 受動的ログモード: シーン要約がログに出力される
- [ ] 単体テスト全パス

---

## フェーズ 5: 安全ガードレール

### 目的
自律エージェントの誤動作による物理破壊を防止する、意思決定層から独立したルールベース安全層。

### 批判的リスク評価

| リスク | 深刻度 | 詳細 | 緩和策 |
|:---|:---|:---|:---|
| **RTSP断検知の誤判定** | 中 | ネットワーク一時的ゆらぎで断判定→不要な停止が頻発 | 3秒以上のタイムアウトを条件とし、一時的なフレームドロップは許容 |
| **夜間モード判定の精度** | 低 | 彩度分散チェックだけでは照明色温度の影響を受ける | 最初はシンプルな輝度閾値で実装。精度問題が出たら改善 |
| **ガードレールがパフォーマンスに影響** | 低 | フレームごとのチェックが推論ループを遅延させる | チェックは軽量な条件分岐のみ。重い処理（輝度統計等）はNフレームに1回 |

### ファイル変更計画

#### [NEW] `src/pico/guardrails.py`

```python
class GuardRails:
    """物理安全ガードレール。設計書§7の実装。

    ルールベースの安全チェック。LLM/VLMの判断に依存しない。
    """
    def check_frame_health(self, frame, last_frame_time) -> FrameStatus:
        """RTSP断 / フリーズ検知"""
        ...

    def check_bbox_sanity(self, bbox, frame_shape) -> bool:
        """異常BBox（画面の75%超）の除外"""
        ...

    def check_drive_duration(self, drive_start_time) -> bool:
        """連続駆動時間の制限（5秒上限）"""
        ...

    def check_night_mode(self, frame) -> bool:
        """夜間低照度の判定"""
        ...
```

#### [NEW] `tests/test_guardrails.py`
- フレームタイムアウト判定
- 異常BBox検知
- 連続駆動制限

### 完了判定
- [ ] RTSP断時にPTZ停止→リトライが動作する
- [ ] 異常BBoxが除外される
- [ ] 5秒以上の連続駆動が制限される
- [ ] 単体テスト全パス

---

## 全体のデータフロー（最終形）

```
反射ループ (~30ms周期)
============================================================
[Tapo C210] --RTSP--> [VideoReader]
    [VideoReader] --> [GuardRails: フレームヘルスチェック]
    [VideoReader] --> [YoloDetector] --> [SimpleTracker]
    [SimpleTracker] --> [AdaptivePID]
    [AdaptivePID] --> [GuardRails: 駆動時間チェック]
    [AdaptivePID] --> [PTZController] --SOAP--> [Tapo C210]

認知ループ (~5-10s周期, 非同期)
============================================================
[SimpleTracker] --新規検出イベント--> [CognitionEngine]
    [CognitionEngine] --> [MemoryStore: search(class_name)]
    [CognitionEngine] --> [Ollama WSL2: analyze_scene (2-5秒)]
    [CognitionEngine] --> [Epoch整合性チェック]
    [CognitionEngine] --ロックオン指示 or 破棄--> [SimpleTracker]
```

---

## パッケージ構成（最終形）

```
e:\Python Scripts\Pico\
├── src/
│   └── pico/
│       ├── __init__.py
│       ├── config.py          ← camera_config.py を移動・拡張
│       ├── video_reader.py    ← RTSPVideoReader 統合
│       ├── onvif_client.py    ← ONVIF初期化・PTZ制御
│       ├── detector.py        ← YOLO推論エンジン
│       ├── tracker.py         ← 簡易IoUトラッカー
│       ├── pid_controller.py  ← 適応PID制御
│       ├── ollama_client.py   ← Ollama REST API
│       ├── memory.py          ← SQLite FTS5 記憶
│       ├── cognition.py       ← 認知ループエンジン
│       └── guardrails.py      ← 物理安全ガードレール
├── tests/
│   ├── test_camera_config.py  ← 既存（パス更新）
│   ├── test_video_reader.py
│   ├── test_onvif_client.py
│   ├── test_pid_controller.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_ollama_client.py
│   ├── test_memory.py
│   ├── test_cognition.py
│   └── test_guardrails.py
├── scripts/                   ← 既存スクリプトの移動先
│   ├── run_tracking.py        ← tapo_yolo_tracking.py の後継
│   ├── calibrate.py           ← calibrate_tapo.py の後継
│   ├── manual_control.py      ← move.py の後継
│   └── benchmark_detector.py  ← 性能ベンチマーク
├── docs/
│   ├── camera_agent.md        ← コンセプト設計書（参照用、変更なし）
│   └── implementation_plan.md ← 本計画書
├── pyproject.toml
└── README.md
```

> **`src/` レイアウトの導入判断**: `pyproject.toml` の `requires-python = ">=3.13"` で `src/` レイアウトを使う場合、パッケージ設定が必要。`uv` はこれを自動解決するが、既存の `from camera_config import ...` パスが変わるため、全スクリプトの import 修正が必要。
>
> **代替案**: 複雑性を避けるなら、`src/` ではなくプロジェクト直下に `pico/` パッケージを作る案もある。

---

## 依存関係の追加（全フェーズ通じて）

| パッケージ | フェーズ | 理由 | AGENTS.md の「標準ライブラリ優先」との整合性 |
|:---|:---|:---|:---|
| `onnxruntime>=1.24` | 3 | YOLO推論の高速化 | 現在の cv2.dnn から正当な移行。推論エンジンとして必要 |
| `httpx>=0.28` | 4 | Ollama APIの非同期呼び出し | `urllib.request` では非同期不可。asyncio 対応HTTP として最小限 |

> `numpy`, `opencv-python`, `onvif-zeep`, `python-dotenv` は既存依存。追加は上記2つのみに抑える。ByteTrack用の外部ライブラリは**使わない**（自前IoUトラッカーで代替）。

---

## 確認事項

以下の判断が実装方針に影響する。

1. **パッケージ構成**: `src/pico/` レイアウト vs プロジェクト直下 `pico/` のどちらを採用するか？既存スクリプトの import パスに影響する。

2. **フェーズ1の着手範囲**: フェーズ1（リファクタ）でパッケージ構成の変更まで含めるか？それとも、まず重複コードの抽出だけ行い、パッケージ化は後回しにするか？

3. **既存スクリプトの扱い**: `tapo_yolo_tracking.py` / `move.py` / `calibrate_tapo.py` / `trace_face.py` は `scripts/` に移動して名前を変更するか？それとも互換性のため現在の場所に残すか？

4. **フェーズ4 の認知ループ初期モード**: 「受動的ログモード（シーン要約のみ）」から始める方針でよいか？それとも最初からロックオン指示機能まで実装するか？

5. **Ollama のモデル**: `qwen2.5-vl:3b` で確定か？他のVLMモデル（例: `llava`, `minicpm-v`）の選択肢はあるか？

---

## 未解決の技術的疑問

- **yolov8s.onnx の出力形状**: 現在のモデルは E2E NMS 統合版か？ `cv2.dnn` で動いている＝おそらく標準エクスポートだが、確認が必要。フェーズ3の ORT 移行時の後処理設計に影響する。
- **Tapo C210 の stream2 解像度**: 設計書は `640x480` と記載しているが、実際の stream2 解像度は確認済みか？検出精度に直接影響する。

---

## 検証計画

### 自動テスト
```powershell
uv run pytest                    # 各フェーズで全テスト通過を確認
uv run ruff check .              # Lint エラーゼロ
uv audit                         # 依存関係の脆弱性チェック
```

### 手動検証
- **フェーズ1**: 既存 `tapo_yolo_tracking.py` の動作が変わらないこと
- **フェーズ2**: 実機でPID追従の改善を確認（ハンチング解消、滑らか追従）
- **フェーズ3**: ORT推論速度のベンチマーク計測
- **フェーズ4**: WSL Ollama への接続確認、認知ループが反射ループをブロックしないこと
- **フェーズ5**: RTSP切断シミュレーション（LANケーブル抜き）での緊急停止動作
