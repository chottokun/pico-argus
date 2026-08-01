---
type: Concept
title: Tapo Camera Infrastructure Configuration
description: Tapoカメラの安全制御クランプ限界、極性反転、およびキャリブレーション設定の物理・論理構成仕様
status: active
timestamp: 2026-08-01T10:30:00+09:00
tags:
  - tapo
  - config
  - physical
  - limits
  - calibration
sources:
  - id: camera_agent_doc
    resource: /Docs/raw/camera_agent.md
    title: Camera Agent Specification
---

# Tapo Camera Infrastructure Configuration

## 1. 概要

本コンセプトは、物理カメラ（Tapo C210等）と本システム間の通信・物理制御パラメータを規定するインフラ設定仕様です。
カメラ可動時のサーボモーターの寿命保護、過熱融損防止、安全な移動限界クランプ、および設置方向による極性（Pan/Tilt）の整合性を制御するファイル構成を定義します。

## 2. 詳細仕様・構造

### 2.1 環境設定ファイル (`camera_config.json` / `tapo_config.json`)
実機キャリブレーションスクリプト (`calibrate_tapo.py`) の実行によって物理限界値が測定され、以下のスキーマで保存されます。

```json
{
    "MAX_LIMIT_X": 0.96,
    "MAX_LIMIT_Y": 0.85,
    "INVERT_PAN": true,
    "INVERT_TILT": true,
    "STEP_SIZE_X": 0.15,
    "STEP_SIZE_Y": 0.10,
    "TOTAL_STEPS_X": 36,
    "TOTAL_STEPS_Y": 20,
    "RETURN_STEPS_X": 18,
    "RETURN_STEPS_Y": 10,
    "CALIBRATED_AT": "2026-07-25 00:00:00"
}
```

#### 各項目の論理的・物理的意味
- **`MAX_LIMIT_X` / `MAX_LIMIT_Y`**: ONVIF RelativeMove で指定可能な最大安全クランプ移動量。現在位置の論理座標がこれを超えないように自動保護クランプが作動します。
- **`INVERT_PAN` / `INVERT_TILT`**: 物理カメラの設置方向による追従方向（極性）の制御用フラグ。
  - **正立設置（通常デスクトップ等）**: `"INVERT_PAN": true`, `"INVERT_TILT": true` （両方 `true` が必須。`false` にすると逆駆動現象を起こし即座にフレームアウトします）。
  - **天井吊り下げ（逆さ設置）**: `"INVERT_PAN": false`, `"INVERT_TILT": false`。
- **`STEP_SIZE_X` / `STEP_SIZE_Y`**: 1ステップあたりの移動量。
- **`TOTAL_STEPS_X` / `TOTAL_STEPS_Y`**: 端から端までの総ステップ数。
- **`RETURN_STEPS_X` / `RETURN_STEPS_Y`**: 中心原点に戻るために必要な逆ステップ数。

### 2.2 セキュアな環境変数 `.env` の管理
ハードウェア認証鍵・IPアドレスはリポジトリにコミットせず、`.env` 経由で動的にロードされます。
- `TAPO_USER` / `TAPO_PASS`: Tapo ONVIF 用高度設定で作成した固有アカウント情報。
- `TAPO_IP`: カメラのローカル固定IP。

## 3. 制約・注意事項

- **激突融損・過負荷防止**: ONVIF の ContinuousMove コマンドはパケットロスにより Stop SOAP がロストすると壁に激突し続けてギアやモーターを破損するリスクがあります。
  - すべての物理移動命令には **`Dwell Time (350ms自動停止)`** をバインドすること。
  - 急制動を避けるための **`Slew Rate (1ステップ増減幅 ±0.03以内制限)`** を厳守すること。

## 4. 関連概念

* [Camera Agent System](../architecture/camera_agent.md) - 設定を利用して制御を行う物理PTZ・サーボコアモジュール
