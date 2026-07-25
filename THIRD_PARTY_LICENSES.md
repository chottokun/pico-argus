# サードパーティライブラリおよび AI モデルのライセンス表記 (Third-Party Licenses & Model Notices)

本プロジェクト（`Pico Argus`）コード自体のライセンスは [LICENSE](file:///e:/Python%20Scripts/Pico/LICENSE) に規定される **MIT License** ですが、依存するライブラリおよび利用される外部 AI モデル・学習済み重みファイルには、それぞれの権利者が定めるオリジナルのライセンス・利用規約が適用されます。

ご利用・再配布の際は、以下の各コンポーネントのライセンスをご確認のうえ、適切に遵守してください。

---

## 🧠 1. AI モデル・学習済み重みファイル (AI Models & Weights)

| コンポーネント / モデル | 提供元 / オリジナルプロジェクト | ライセンス / 利用規約 | 注意事項 |
| :--- | :--- | :--- | :--- |
| **YOLOv8 ONNX (`yolov8n.onnx`, `yolov8s.onnx`)** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | AGPL-3.0 / 商用ライセンス | オープンソース（AGPL-3.0）または Ultralytics 社の商用ライセンス規約に準拠します。 |
| **Haar Cascade (`haarcascade_frontalface_default.xml`)** | [OpenCV Project](https://github.com/opencv/opencv) | Apache 2.0 / BSD | OpenCV 標準配布の顔検出学習済みカスケードデータです。 |
| **Ollama 視覚 VLM (`gemma4:e2b` 等)** | [Google Gemma Terms](https://ai.google.dev/gemma/terms) / [Ollama](https://ollama.com/) | Gemma Terms of Use / 各モデル個別の規約 | Ollama を通じて呼び出す外部 VLM モデルの利用規定に準拠します。 |

---

## 📦 2. 主要依存 Python パッケージ (Python Dependencies)

| パッケージ | 用途 | ライセンス |
| :--- | :--- | :--- |
| `onnxruntime` | ONNX 高速推論エンジン | MIT License |
| `opencv-python` / `opencv-contrib-python` | 画像処理・ビデオ入力 | Apache 2.0 License / BSD License |
| `onvif-zeep` | ONVIF PTZ カメラ制御プロトコル | MIT License |
| `numpy` | 数値計算・行列処理 | BSD 3-Clause License |
| `python-dotenv` | 環境変数 `.env` ロード | BSD License |
| `httpx` | HTTP/非同期通信 (Ollama Client) | BSD 3-Clause License |
| `mcp` | Model Context Protocol SDK | MIT License |
| `pytest` | ユニットテスト・統合テスト環境 | MIT License |
| `ruff` | 高速 Python リンター / フォーマッター | MIT License |

---

## 📷 3. ハードウェア・プロトコル

- **ONVIF™**: ONVIF は ONVIF Inc. の商標です。本プロジェクトは ONVIF 標準仕様に基づく公開 SOAP ネットワークプロトコル通信を利用しています。
- **Tapo**: Tapo は TP-Link Technologies Co., Ltd. の登録商標です。
