---
type: Concept
title: Camera Agent System
description: Tapoカメラ制御、顔追跡、YOLO物体検出を統合したリアルタイムビジョンエージェントのアーキテクチャ
status: active
timestamp: 2026-08-01T10:30:00+09:00
tags:
  - camera
  - tracking
  - yolo
  - tapo
sources:
  - id: camera_agent_doc
    resource: /Docs/raw/camera_agent.md
    title: Camera Agent Specification
---

# Camera Agent System

## 1. 概要

Camera Agent System は、Tapo パンチルトカメラを用いたリアルタイム映像ストリーミング、物体検出（YOLOv8/Haar Cascade）、動体追跡、カメラ首振り制御を行う統合エージェントシステムです。

## 2. 詳細仕様・構造

- **主要コンポーネント**:
  - `tapo_yolo_tracking.py`: YOLOv8を用いた高精度な人物・顔追跡ループ。
  - `trace_face.py`: OpenCV Haar Cascade をベースにした軽量追跡スクリプト。
  - `calibrate_tapo.py`: カメラ画角・モータ移動量のキャリブレーション用ユーティリティ。
  - `move.py`: Tapo API 経由の首振り・プリセット移動用モジュール。

- **制御アルゴリズム**:
  - 画面中央からの検出バウンディングボックスのずれ（エラー）に基づく比例制御（P制御）。
  - 移動命令のオーバーシュート防止およびデッドゾーン（感度不感帯）処理。

## 3. 制約・注意事項

- Tapo API のネットワーク遅延があるため、追跡レスポンスには最小限の制御間隔（スリープ）を設ける必要があります。
- キャリブレーションデータは `tapo_config.json` / `camera_config.json` に保存されます。

## 4. 関連概念

* [MCP Specification](./mcp_specification.md) - カメラ操作を標準インターフェース化するMCP層
* [MCP Usecases](../domain/mcp_usecases.md) - 監視カメラシステムの実際の運用ユースケース
