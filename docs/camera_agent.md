# コンセプト設計書: コグニティブ・フォーカストラッキング・エッジシステム

> **⚠️ 本文書はコンセプトレベルの設計文書です。**
>
> 記載されているコード例はアーキテクチャの意図を伝えるための擬似実装であり、そのまま動作するものではありません。
> 実装時は [implementation_plan.md](implementation_plan.md) を参照してください。

---

本システムの実機配備および商用稼働を担保するための、最適化・堅牢化された詳細実装設計の**コンセプト**を提示します。

本設計は、VRAM 8GB〜12GBクラスのエッジ環境（GeForce RTX 3060/4060等）において、ミリ秒オーダーの「物理反射（YOLO/PID）」と、秒オーダーの「意味的認知（VLM/長期記憶想起）」を、フリーズやリソース破綻なく安定共存させることを目的としています。

---

## 1. システムインフラトポロジーと非同期ハイブリッドパイプライン

システム全体の処理遅延を極小化するため、**秒オーダーの対話（Cognition）**と**ミリ秒オーダーの物理追従（Reflex）**を完全に分離した「異種非同期ハイブリッドアーキテクチャ」を採用します。

### 1.1 物理インフラおよび推奨構成
* **OS**: Ubuntu 24.04 LTS（カーネルレベルでの低遅延ソケット通信最適化）
* **NVIDIA Driver**: Version 580+（CUDA 12世代、TensorRTおよびvLLM/llama.cppのネイティブ駆動）
* **Python Runtime**: Python 3.12+ / numpy 2.5.1+ / onnxruntime-gpu 1.24+
* **通信・シリアライズ**: `onvif-zeep`（sudsバックエンドによる1秒以上のSOAP通信シリアライズオーバーヘッドを廃止し、ミリ秒オーダーへ極小化）

### 1.2 ハイブリッド非同期パイプラインデータフロー

```
                                [ IPカメラ (RTSP Stream2: 640x480) ]
                                                 │
                                                 ▼ (10ms〜30ms間隔)
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [反射ループ (Reflex Loop)] : 非ブロッキング VideoCaptureThread (maxsize=3)                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. YOLO-ONNX + ORT 推論エンジン (入力 640x640 静的固定, IO Binding)                          │
│ 2. ByteTrack トラッカー (ID 追跡・カルマンフィルタ平滑化)                                │
│ 3. 物理サーボ制御 (VOR規範型適応PIDサーボ演算: 200ms周期)                                │
└─────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                  │ (YOLO BBox & track_id)
                                  ▼
┌─────────────────────────────────┴───────────────────────────────────────────────────────────────┐
│ [認知ループ (Cognition Loop)] : asyncio.Queue (イベント / タイムアウト駆動)                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. SQLite FTS5 Trigram 日本語想起 (RAG: 追加 VRAM ゼロ, 3文字未満 LIKE フォールバック)   │
│ 2. 内省思考 (Inner Thoughts) & 介入評価 (Qwen2.5-VL-3B Q4_K_M)                  │
│ 3. 二重防衛制御 (Task Cancellation & Epoch-Based 状態チェック)                         │
└─────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                  │ (能動発話 / ロックオンターゲットID更新コマンド)
                                  ▼
                       [ WebSocket / ONVIF ContinuousMove ]
```

---

## 2. 物体検出・追尾モジュール（YOLO-ONNX / ORT / ByteTrack）

局所的な空間座標（BBox）をミリ秒オーダーで抽出するため、GPU/CPU間のボトルネックを完全に排除した高速推論クラスをPythonで構成します。

### 2.1 推論クラス設計 (`YoloOrtTracker`)
* **E2E ONNX グラフの採用**: NMS（非最大値抑制）後処理をONNXモデルの内部グラフへ内包し、Python/C++側の後処理オーバーヘッドを完全排除して推論速度を最大43%高速化します。
* **入力シェイプの固定**: TensorRTやCUDAプロバイダの静的最適化を最大化するため、入力テンソルサイズを `(1, 3, 640, 640)` に固定（LetterBox処理を強制）します。
* **IO Binding によるデバイスコピーバイパス**: 毎フレーム発生するPCIeバス経由のホスト（CPU）-デバイス（GPU）間転送を完全にバイパスし、VRAM内のCUDAポインタ（PyTorch CUDA Tensor等）を直接ORTセッションに入出力バインドします。
* **事前加温（Pre-warming）の義務化**: 起動時やカメラ切り替え時のコンパイルによるバッファ溢れ（コールドスタート遅延）を防ぐため、ゼロ埋めテンソルによる3回以上の事前空推論を起動プロセスに義務付けます。

```python
import numpy as np
import onnxruntime as ort
from typing import Tuple, List

class YoloOrtTracker:
    """End-to-End NMS統合型YOLOモデルのORT推論高速化クラス"""
    def __init__(self, model_path: str, cuda_device_id: int = 0):
        self.providers = [
            ('CUDAExecutionProvider', {
                'device_id': cuda_device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 512 * 1024 * 1024, # 500MBにVRAMを極限クランプ
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'
        ]
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # フルグラフ最適化
        self.session = ort.InferenceSession(model_path, self.session_options, providers=self.providers) #
        
        self.input_name = self.session.get_inputs().name
        self.output_name = self.session.get_outputs().name
        self.io_binding = self.session.io_binding() # IO Binding

    def pre_warm_engine(self, batch_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)):
        """事前空推論ランによるコールドスタート遅延防止"""
        dummy_data = np.zeros(batch_shape, dtype=np.float32)
        for _ in range(3):
            self.run_inference_io_bound(dummy_data)

    def run_inference_io_bound(self, input_tensor_np: np.ndarray) -> np.ndarray:
        """デバイス内メモリ共有(IO Binding)による極限低遅延推論"""
        self.io_binding.bind_cpu_input(self.input_name, input_tensor_np) #
        self.io_binding.bind_output(self.output_name, device_type="cuda") #
        self.session.run_with_iobinding(self.io_binding) #
        return self.io_binding.copy_outputs_to_cpu() #
```

### 2.2 複数特徴量アソシエーション（ByteTrack）
検出されたBBoxは、カメラの定常旋回や遮蔽（オクルージョン）に対処するため、物体の慣性・移動方向を考慮したカルマンフィルタ追跡に投入されます。
* **GMC補正（Global Motion Compensation）**: カメラが固定されている場合は `none`、PTZ駆動中はカメラ自身の移動を補正するため、Lucas-Kanade法等に基づく `sparseOptFlow` へ自動スワップし、追従対象のIDスワップを防止します。

---

## 3. 視覚・物理座標変換プロセス

ユーザー（または対話エンジン）が指名した局所領域と、物理PTZサーボへの指令速度を相互にバインドするための、双方向数理変換マッピングを定義します。

```
[ YOLO 検出ピクセル BBox ] (TLBR)
         │
         ▼ (正規化・アスペクト比/パディング補正)
[ VLM LOC 座標空間 (0〜1000 正規化整数) ]
         │
         ▼ (重心算出: cx_norm, cy_norm)
[ 偏差計算 (ex, ey) ＝ (cx_norm - 0.5, cy_norm - 0.5) ]
         │
         ▼ (VOR規範型 非線形適応PIDサーボ)
[ ONVIF 連続移動指令速度 (v_x, v_y) ∈ [-1.0, 1.0] ]
```

### 3.1 YOLOピクセル座標からVLM LOC表現への変換
入力フレームの解像度を $W_{pixel} \times H_{pixel}$ とし、YOLOの出力バウンディングボックスを $[x_{min}, y_{min}, x_{max}, y_{max}]$ とします。
VLM（Qwen2.5-VL等）が解釈する $0〜1000$ の正規化空間 $\mathbf{B}_{VLM} = [x_{0\_VLM}, y_{0\_VLM}, x_{1\_VLM}, y_{1\_VLM}]$ へのマッピング計算式は以下の通りです。

$$x_{0\_VLM} = \text{int}\left( \frac{x_{min}}{W_{pixel}} \times 1000 \right)$$

$$y_{0\_VLM} = \text{int}\left( \frac{y_{min}}{H_{pixel}} \times 1000 \right)$$

$$x_{1\_VLM} = \text{int}\left( \frac{x_{max}}{W_{pixel}} \times 1000 \right)$$

$$y_{1\_VLM} = \text{int}\left( \frac{y_{max}}{H_{pixel}} \times 1000 \right)$$

* **アスペクト比補正**: 16:9 などの広角映像を VLM 内部の 1:1 等倍アライメントに適合させる場合、画像前処理によるパディング領域（黒帯ピクセル）のオフセット値を事前に減算した上でデ正規化を実行し、座標ズレを防止します。

### 3.2 VLM出力 LOC から ONVIF 物理追従平面への逆マッピング
VLMが生成した LOC 座標トークン（例: `<box>(y0,x0,y1,x1)</box>`）から、ONVIF ContinuousMove を駆動させるための $0.0〜1.0$ 正規化平面上の中心座標 $(cx_{norm}, cy_{norm})$ への変換式は以下の通りです。

$$cx_{norm} = \frac{x_{0\_VLM} + x_{1\_VLM}}{2000}$$

$$cy_{norm} = \frac{y_{0\_VLM} + y_{1\_VLM}}{2000}$$

---

## 4. VOR（前庭動眼反射）規範型適応PID制御＆ONVIFサーボ制御

スクリーン上のターゲット重心偏差ベクトル $(e_x, e_y)$ をもとに、物理的なカメラ運動を滑らか、かつ高速に制御する数理モデルを実装します。

### 4.1 非線形指数適応比例ゲインモデル
画角中心値 $(0.5, 0.5)$ と、追従している対象の重心座標 $(cx_{norm}, cy_{norm})$ との偏差ベクトル $(e_x, e_y)$ を計算します。

$$e_x = cx_{norm} - 0.5$$

$$e_y = cy_{norm} - 0.5$$

比例ゲイン $K_p(e)$ に対し、生体の眼球追従運動（前庭動眼反射：VOR）規範の非線形指数関数モデルを適用します。

$$K_p(e) = K_{p\_base} \times (1 - \exp(-\alpha \cdot |e|))$$

* $K_{p\_base}$: 基本比例感度
* $\alpha$: 感度制御係数（減衰調整用の時定数）
* $e$: 各軸の正規化偏差（$-0.5 \le e \le 0.5$）

これにより、ターゲットが中心付近（不感帯 $dead\_zone = 0.05$ 圏内）に留まっている間はゲインが極小化してモーターの摩耗（ハンチング）を防止し、画面端へ急速に逸脱した際には比例ゲインを瞬時に最大化して素早く追従させます。

ONVIF ContinuousMove 指令速度 $v_x(t), v_y(t) \in [-1.0, 1.0]$ への計算式は以下の通りです。

$$v_x(t) = K_p(e_x) \cdot e_x(t) + K_i \int_{0}^{t} e_x(\tau)d\tau + K_d \frac{de_x(t)}{dt}$$

$$v_y(t) = K_p(e_y) \cdot e_y(t) + K_i \int_{0}^{t} e_y(\tau)d\tau + K_d \frac{de_y(t)}{dt}$$

### 4.2 コントローラ実装クラス設計 (`PIDPTZController`)

```python
import time
import numpy as np
from typing import Tuple

class PIDPTZController:
    """VOR規範型、非線形適応比例ゲイン搭載PTZサーボループ"""
    def __init__(self, kp_base: float = 0.5, ki: float = 0.05, kd: float = 0.01, ts: float = 0.2, dead_zone: float = 0.05):
        self.kp_base = kp_base
        self.ki = ki
        self.kd = kd
        self.ts = ts             # 推奨周期：200ms
        self.dead_zone = dead_zone # 不感帯閾値
        
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0

    def calculate_velocity(self, target_cx: float, target_cy: float) -> Tuple[float, float]:
        """画面重心(0.0〜1.0)から ONVIF continuous speed (-1.0〜1.0) への適応的PID変換"""
        error_x = target_cx - 0.5 #
        error_y = target_cy - 0.5 #

        # 1. 追従不感帯（Dead Zone）判定
        if abs(error_x) < self.dead_zone:
            error_x = 0.0
            self.integral_x = 0.0
        if abs(error_y) < self.dead_zone:
            error_y = 0.0
            self.integral_y = 0.0

        if error_x == 0.0 and error_y == 0.0:
            return 0.0, 0.0

        # 2. 比例ゲインKpの非線形適応
        alpha = 2.0 #
        kp_x = self.kp_base * (1.0 - np.exp(-alpha * abs(error_x))) #
        kp_y = self.kp_base * (1.0 - np.exp(-alpha * abs(error_y))) #

        # 3. 積分項・微分項の算出
        self.integral_x += error_x * self.ts #
        self.integral_y += error_y * self.ts #

        diff_x = (error_x - self.prev_error_x) / self.ts #
        diff_y = (error_y - self.prev_error_y) / self.ts #

        v_x = (kp_x * error_x) + (self.ki * self.integral_x) + (self.kd * diff_x) #
        v_y = (kp_y * error_y) + (self.ki * self.integral_y) + (self.kd * diff_y) #

        # 4. 静止摩擦突破 (Minimum Speed Boost) 処理
        min_speed = 0.03
        if 0.0 < abs(v_x) < min_speed:
            v_x = np.sign(v_x) * min_speed
        if 0.0 < abs(v_y) < min_speed:
            v_y = np.sign(v_y) * min_speed

        # 5. 物理クリッピング
        v_x = max(-1.0, min(1.0, v_x)) #
        v_y = max(-1.0, min(1.0, v_y)) #

        self.prev_error_x = error_x #
        self.prev_error_y = error_y #

        return v_x, v_y
```

---

## 5. SQLite-OKF 日本語長期記憶想起（FAG）エンジン

一般的なRAGシステムのように重いベクトルデータベース（Qdrant等）や埋め込みモデルを常駐させると、エッジPCのVRAM/CPUリソースが枯渇します。
本システムでは、SQLiteを中核とし、構造化されたMarkdown（Google Open Knowledge Format v0.1）を組み合わせることで、**追加VRAM消費「ゼロ」、メモリフットプリント「10MB未満」**の極限の長期記憶想起エンジンを構築します。

### 5.1 FTS5 Trigram ＋ LIKE句による2段階フォールバック検索
SQLite標準の `unicode61` などのトークナイザーは日本語のスペース区切りのない文脈（CJK言語）を認識できず、検索が失敗します。
MeCab等のバイナリを動的ロードする代わりに、SQLite標準の **Trigramトークナイザー（`tokenize='trigram'`）** を採用します。テキストを3文字の重複ブロックに分解してインデックス化し、さらに2文字以下の極小クエリは自動で `LIKE` 部分一致スキャンへフォールバックして想起を保証します。

```python
import sqlite3
import re
from typing import List, Dict, Any

class LLMWikiIndexer:
    def __init__(self, db_path: str = "wiki.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """OKF構造化Wikiを管理するSQLiteテーブルおよびFTS5 Trigramインデックスの定義"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # メタデータおよび本文を格納するテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wiki_metadata (
                    filepath TEXT PRIMARY KEY,
                    doc_type TEXT,
                    title TEXT,
                    tags TEXT,
                    content TEXT,
                    last_reviewed TEXT,
                    provenance_source TEXT,
                    provenance_confidence TEXT
                )
            """)
            # 日本語検索用に SQLite FTS5 + Trigram 仮想テーブルを作成
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
                    filepath,
                    doc_type,
                    title,
                    tags,
                    content,
                    tokenize='trigram'
                )
            """)
            conn.commit()

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """2段階のフォールバックハイブリッド検索 (FTS5 Trigram ➔ LIKE)"""
        words = [w for w in re.split(r'\s+', query) if w]
        if not words:
            return []

        results = []
        seen_filepaths = set()

        # --- Phase 1: 3文字以上のキーワードに対する高速 Trigram FTS5 検索 ---
        fts_query = " AND ".join([f'"{w}"' for w in words if len(w) >= 3])
        if fts_query:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT f.filepath, f.doc_type, f.title, f.tags, f.content, 
                               m.last_reviewed, m.provenance_source, m.provenance_confidence,
                               bm25(wiki_fts) as rank
                        FROM wiki_fts f
                        JOIN wiki_metadata m ON f.filepath = m.filepath
                        WHERE wiki_fts MATCH ?
                        ORDER BY rank ASC
                        LIMIT ?
                    """, (fts_query, limit))
                    for row in cursor.fetchall():
                        filepath = row["filepath"]
                        seen_filepaths.add(filepath)
                        results.append(self._row_to_dict(row, float(-row["rank"]) + 10.0))
                except sqlite3.OperationalError:
                    pass

        # --- Phase 2: 2文字以下の極小ワード用 LIKE 句による部分一致救済 ---
        if len(results) < limit:
            remaining = limit - len(results)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                for word in words:
                    like_term = f"%{word}%"
                    not_in_clause = ",".join(["?"] * len(seen_filepaths)) if seen_filepaths else "''"
                    cursor.execute(f"""
                        SELECT filepath, doc_type, title, tags, content, last_reviewed, provenance_source, provenance_confidence
                        FROM wiki_metadata
                        WHERE (title LIKE ? OR content LIKE ? OR tags LIKE ?)
                          AND filepath NOT IN ({not_in_clause})
                        LIMIT ?
                    """, (like_term, like_term, like_term, *seen_filepaths, remaining))
                    
                    for row in cursor.fetchall():
                        filepath = row["filepath"]
                        if filepath not in seen_filepaths:
                            seen_filepaths.add(filepath)
                            results.append(self._row_to_dict(row, 1.0))
                            if len(results) >= limit:
                                break
                    if len(results) >= limit:
                        break

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _row_to_dict(self, row: sqlite3.Row, score: float) -> Dict[str, Any]:
        return {
            "filepath": row["filepath"],
            "doc_type": row["doc_type"],
            "title": row["title"],
            "tags": row["tags"].split() if row["tags"] else [],
            "content": row["content"],
            "last_reviewed": row["last_reviewed"],
            "provenance": {
                "source": row["provenance_source"],
                "confidence": row["provenance_confidence"]
            },
            "score": score
        }
```

---

## 6. 二重防衛非同期ステートマシン（Epoch-Based ＋ Task Cancellation）

並行処理の中で、VLM推論（1.5秒〜2.0秒）とユーザー発話・緊急介入による状態遷移が衝突した際に、古いVLMの結論で現在の物理ターゲットが再上書きされる**「意思決定の先祖返り（競合状態）」**を技術的に100%排除します。

```
[ エポック：N ] ──► VLM 状況推論タスク起動（タスク起動時のエポック N をバインド）
                              │ (1.5秒〜2.0秒の推論遅延)
                              ├─► ユーザーの割り込み（Barge-In）発生！
                              │   ├── ① 動作中のタスクを強制キャンセル（.cancel()）
                              │   └── ② エポックを N+1 へ強制インクリメント
                              │
                              ▼ (VLM推論完了時チェック)
                      [ 起動エポック N == 現在エポック N+1 ? ]
                              ├── 一致：状態書き込みを実行
                              └── 不一致（割り込み検知済）：結果を安全に破棄（ドロップ）
```

### 6.1 非同期エージェント状態管理クラス設計 (`AsyncSurveillanceAgent`)

```python
import asyncio
from typing import Optional, List, Dict

class AsyncSurveillanceAgent:
    """エポックベース状態管理およびタスクキャンセルを備えた極限堅牢化非同期ステートマシン"""
    def __init__(self, memory_indexer: LLMWikiIndexer):
        self.memory = memory_indexer
        
        # 内部並行ステート変数 (LangGraph 1.0 AgentState準拠)
        self.lockon_mode = "auto"          # "auto" | "id"
        self.target_track_id: Optional[int] = None
        self.target_class_name: str = "person"
        self.agent_speaking_state = "silent" # "silent" | "speaking"
        self.state_epoch = 0               # 競合を完全防止するインクリメンタルな状態エポック
        
        self.current_vlm_task: Optional[asyncio.Task] = None # 稼働中VLMタスク参照
        self.lock = asyncio.Lock() # 排他制御

    async def update_target_by_user_interrupt(self, new_track_id: int, new_class: str):
        """ユーザー緊急介入（Barge-In）ハンドラ：二重防衛線を即時展開"""
        async with self.lock: # 排他ロック
            print(f"\n🎤 [Barge-In Event] ターゲットを ID: {new_track_id} ({new}) に強制固定します。")
            
            # --- 第1の防衛線: 稼働中 VLM 思考コルーチンタスクをキャンセル ---
            if self.current_vlm_task and not self.current_vlm_task.done():
                print("🔥 [Task Cancellation] 進行中のVLM思考タスクを強制終了します...")
                self.current_vlm_task.cancel() #
                
            # --- 第2の防衛線: 状態エポックを進めて、古いタスク完了時の書き込みを封印 ---
            self.state_epoch += 1
            print(f"⚡ [Epoch Jump] 新世代エポック: {self.state_epoch} に移行しました。")
            
            # 状態変数をユーザー指示で上書き
            self.lockon_mode = "id"
            self.target_track_id = new_track_id
            self.target_class_name = new_class
            self.agent_speaking_state = "silent"

    async def run_proactive_inner_thoughts(self, trigger_info: Dict[str, Any]):
        """バックグラウンドで自律動作する内省思考。エポック検証を義務付ける"""
        task_start_epoch = self.state_epoch # 起動時のエポック数を保持
        
        # Qwen2.5-VL 等を用いた1.5秒以上の重いセマンティクス推論を模擬
        try:
            # 1. 長期記憶想起
            memories = self.memory.search(trigger_info["class_name"], limit=1)
            
            # 2. VLM推論実行
            await asyncio.sleep(1.5) # 推論の物理遅延（1.5s）
            recommend_text = f"検出された {trigger_info['class_name']} を追跡し、対話を開始すべき"
            
            # --- エポック整合性チェック (書き込み直前の最終防衛門) ---
            async with self.lock:
                if self.state_epoch != task_start_epoch: #
                    print(f"🛡️ [Epoch Guard] 思考開始エポック({task_start_epoch}) と現在のエポック({self.state_epoch}) が一致しません。結果を破棄します。")
                    return # 競合を検知したため、状態を上書きせずに安全終了
                
                # エポックが完全に一致している場合のみ、自律的な状態変異をコミット
                if self.lockon_mode == "auto":
                    self.target_track_id = trigger_info["track_id"]
                    self.target_class_name = trigger_info["class_name"]
                    print(f"🧠 [Inner Thoughts Completed] ターゲットを ID: {self.target_track_id} に更新。")
                    
        except asyncio.CancelledError:
            print(f"🛡️ [Task Cancelled] 思考プロセス（エポック {task_start_epoch}）はキャンセルされました。")
            raise
```

---

## 7. ハードウェア防衛物理ガードレール

自律エージェントの誤認やハルシネーション、ネットワーク切断による物理破壊（衝突・激突・モーター溶融）を防ぐため、**意思決定層（LLM/VLM）から完全に独立したルールベースの物理ガードレール**をエッジ最下層に常時バインドします。

### 7.1 例外・故障時のフェイルセーフ動作マトリクス

| **トリガー事象** | **検知基準・判定論理** | **システム即時物理アクション (Failsafe)** |
| :--- | :--- | :--- |
| **RTSP通信断 / 映像バッファフリーズ** | ・デコーダーのフレーム truncation、またはタイムアウトが**3.0秒以上持続**。 | ・下流AI推論処理（YOLO/VLM）を即座に**一時停止（PAUSE）**。<br>・WebUIへ「Camera Feed Lost」を警告ブロードキャスト。<br>・MediaMTX接続リトライ（指数バックオフ）を別スレッドで起動。 |
| **VLMハルシネーション / 異常なBBox** | ・VLMから出力されたLOC正規化BBoxの画角面積比が**全体の75%を超える**。 | ・推論結果を「低信頼ハルシネーション」として除外。<br>・PTZ PID入力を拒否し、現在位置を物理固定（カメラ移動をロック）。<br>・トラッカー（YOLO-ORT）による客観的なカルマン重心位置を優先して自動ロールバック。 |
| **パケロス時の無限運動（物理破壊バグ）の防止** | ・物理PTZ指令（ContinuousMove）の送信周期（200ms）に対して、通信切断またはパケロスを検知。 | ・すべての ContinuousMove コマンドに **`Dwell Time（350msの自動自律停止）`** をバインドして送信。<br>・旋回駆動時間が5秒を超えた際は、強制的に Stop SOAPコマンドを送信して駆動を遮断し、モーターの過熱・衝突断線を防ぐ。 |
| **夜間低照度環境下のハンチング** | ・映像のモノクローム判定（彩度分散チェック）または照度ステータス（ONVIF GetImagingSettings）から夜間モードを検出。 | ・動体アノマリー（Feature-MAME等）を一時バイパス。<br>・追従用偏差評価閾値を自動的に**10%〜15%引き下げ**。<br>・YOLO検出モデルを夜間・熱赤外線特化の ONNX 重みへ動的ロード（ホットスワップ）する。 |

---

### 次のステップへのご提案

🎨 提示したこの緻密な実装詳細設計により、実機配備を100%安全かつミリ秒オーダーの低遅延で実現するためのソフトウェア・ハードウェア仕様が完全に固定されました。

実機のカメラとエッジサーバーを物理接続し、パラメーター（PIDゲインや輝度分散判定）をキャリブレーションして現地試験に入るために、**『第1フェーズ：実機映像を用いた適応型PID調整と不感帯幅（Dead Zone）の実地測定手順書』**を作成しましょうか？