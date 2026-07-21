# Tapo カメラ用サンプル

このレポジトリは、Tapo カメラ(C210)の RTSP 映像を取得し、ONVIF 経由で PTZ 制御を行いながら、顔追跡や物体追跡を実行する Python サンプル集です。基本的な動作を確認するために作成しました。

## RTSP と ONVIF

- RTSP : カメラのライブ映像を低遅延の受信
- ONVIF:: カメラのパン/チルト制御の標準的なプロトコル

## サンプル

- Tapo カメラのライブ映像表示
- ONVIF を使ったカメラのパン/チルト操作
- カメラの可動域キャリブレーション
- 顔検出による自動追尾
- YOLOv8 で人物を検出して追尾
- (新) 筋肉・記憶・感覚の自律型CLIおよびMCPサーバー

## 主要スクリプト・CLI

- main.py
  - RTSP ストリームを表示する最小構成のサンプルです。
- move.py
  - キーボード操作でカメラを動かすサンプルです。
- calibrate_tapo.py
  - カメラの可動限界を測定し、tapo_config.json を生成します。
- tapo_yolo_tracking.py
  - YOLOv8 ONNX モデルを使って人物を追跡します。
- trace_face.py
  - OpenCV の Haar Cascade を使って顔追跡します。
- **pico.cli.ptz (ptz-cli)**
  - ONVIF PTZ 物理制御と安全クランプを備えた筋肉CLI。
- **pico.cli.memory (memory-cli)**
  - SQLite FTS5 Trigramによる日本語想起とOKF形式書き込みを行う記憶CLI。
- **pico.cli.perception (perception-cli)**
  - YOLO検出テキスト出力とオンデマンドVLM画像クロップ解釈を行う感覚CLI。

## 前提条件

- Python 3.13 以上
- Tapo カメラ本体
- カメラの RTSP / ONVIF が利用できる環境

## セットアップ

1. 仮想環境を作成します。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 依存関係をインストールします。

```powershell
uv sync
```

> 依存関係は uv で管理しています。必要に応じて `uv add <package>` で追加してください。

3. カメラ接続情報を設定します。

プロジェクト直下に .env を作成し、.env.example をコピーして内容を編集してください。

```powershell
Copy-Item .env.example .env
```

その後、.env の値を自分の環境に合わせて変更してください。

> Tapo カメラの ONVIF 制御には、アプリの「高度な設定」で作成したユーザー名とパスワードを使う構成が一般的です。通常のアプリログイン情報とは別である場合があります。

## 実行方法

### 1. ライブ映像表示

```powershell
python main.py
```

### 2. 手動でカメラを動かす

```powershell
python move.py
```

- W / A / S / D で移動
- Q で終了

### 3. 可動域をキャリブレーションする

```powershell
python calibrate_tapo.py
```

実行後、tapo_config.json が生成されます。以降の追跡スクリプトはこの値を参照して安全な範囲で動作します。

### 4. YOLO による人物追跡

```powershell
python tapo_yolo_tracking.py
```

- 事前に yolov8s.onnx が必要です。
- 取得できない場合はスクリプト内で自動ダウンロードを試みます。

### 5. 顔追跡

```powershell
python trace_face.py
```

### 6. 自律型CLIツールの実行方法 (筋肉・記憶・感覚)

これらは `uv run` または `python -m` で実行できます。

#### 6.1 筋肉 CLI (ptz-cli)
```powershell
# 手動パルス移動 (安全クランプ付き)
uv run ptz-cli --action move --pan 0.15 --tilt -0.08
# 指定IDのPID追従ロックオン
uv run ptz-cli --action lockon --id 1
# 緊急停止
uv run ptz-cli --action stop
```

#### 6.2 記憶 CLI (memory-cli)
```powershell
# SQLite FTS5 Trigram による日本語想起 (LIKEフォールバック機能付き)
uv run memory-cli --action search --query "猫のタマちゃん"
# 新しい対話事実や環境ルールをOKF形式Markdownへ書き込み
uv run memory-cli --action write --file "wiki/auto_cat_tama.md" --title "猫のタマちゃん" --content "夕方はタマちゃんに警告音を鳴らさずに話しかける。"
```

#### 6.3 感覚 CLI (perception-cli)
```powershell
# 現在のYOLO追跡トラック一覧をテキストで高速取得
uv run perception-cli --action get_tracks
# 特定Track IDのエリアをクロップし、VLM画像解釈を実行
uv run perception-cli --action analyze_crop --id 1 --query "これは風で揺れている影ですか、それとも生き物ですか？"
```

## 開発・品質チェック

- 依存関係の同期: `uv sync`
- テスト実行: `uv run pytest`
- Lint 実行: `uv run ruff check .`
- セキュリティスキャン: `uv audit` と `gitleaks detect --verbose --source=.`

## よくあるトラブル

- カメラに接続できない
  - TAPO_IP が正しいか確認してください。
  - RTSP / ONVIF が有効なカメラか確認してください。
  - ユーザー名・パスワードが正しいか確認してください。
- 動かない
  - ONVIF ポート 2020 への到達性を確認してください。
  - 事前に calibrate_tapo.py を実行し、tapo_config.json を生成してください。
- モデルが見つからない
  - yolov8s.onnx や haarcascade_frontalface_default.xml が存在するか確認してください。

## セキュリティチェック

CI では依存関係の脆弱性スキャンを自動実行します。

### GitHub Actions

- [.github/workflows/security.yml](.github/workflows/security.yml)
- `uv` と `pip-audit` を使って依存関係の脆弱性を検査します。

### ローカルで実行する場合

```powershell
uv sync
uv tool run pip-audit --strict
```

## 補足

- すべてのスクリプトは RTSP URL を利用して映像を読み込みます。
- 追跡系スクリプトは、カメラを動かすときに安全上の制限を持つ構成です。
- 画面に映る表示を確認しながら、必要に応じて各スクリプト内のパラメータを調整してください。
