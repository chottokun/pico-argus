---
type: Concept
title: Model Context Protocol (MCP) Specification
description: Cognitive Surveillance ツールおよび外部AIクライアント間を連携する MCP サーバー仕様
status: active
timestamp: 2026-08-01T10:30:00+09:00
tags:
  - mcp
  - protocol
  - surveillance
  - tools
sources:
  - id: mcp_spec_doc
    resource: /Docs/raw/mcp_specification.md
    title: MCP Specification Document
---

# Model Context Protocol (MCP) Specification

## 1. 概要

Model Context Protocol (MCP) は、AIモデル（Claude DesktopやAntigravity等）が監視カメラシステム `Pico-Argus` の各種機能（ライブカメラ画像取得、パン・チルト操作、追跡ターゲット設定など）を安全かつ標準化されたツールとして呼び出すためのプロトコル実装です。

## 2. 提供ツール一覧 (Cognitive Surveillance)

- **`get_live_snapshot`**: 最新のリアルタイム静止画を取得
- **`move_camera`**: パン・チルト方向および移動量の指定によるカメラ操作
- **`set_tracking_target`**: 追跡対象のオブジェクト/ID/クラスの設定
- **`get_active_tracks`**: 現在検知・追跡中の物体一覧を取得
- **`calibrate_home` / `conduct_room_survey`**: カメラのホーム位置較正および全方位サーベイ

## 3. 制約・注意事項

- MCPツール呼び出し時のレスポンスタイムアウトに注意。
- 資格情報（パスワード・APIキー）は `.env` 等の環境変数で管理し、リポジトリに含めないこと。

## 4. 関連概念

* [Camera Agent System](./camera_agent.md) - MCPのバックエンドとなる追跡・ビジョンエンジン
* [MCP Usecases](../domain/mcp_usecases.md) - MCPツールの具象的な活用手順
