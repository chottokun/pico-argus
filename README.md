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

## 主要スクリプト

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
