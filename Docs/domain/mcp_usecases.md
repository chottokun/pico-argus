---
type: Concept
title: MCP Usecases
description: 監視カメラシステムにおけるAIエージェントの具体的なシナリオおよび動作フロー
status: active
timestamp: 2026-08-01T10:30:00+09:00
tags:
  - usecase
  - mcp
  - surveillance
  - automation
sources:
  - id: mcp_usecases_doc
    resource: /Docs/raw/mcp_usecases.md
    title: MCP Usecases Document
---

# MCP Usecases

## 1. 概要

AIエージェントが MCP ツールを用いて自律的または対話的に監視ミッションを実行するための代表的なシナリオ群です。

## 2. 主なユースケース

- **定期巡回（Room Survey）**:
  - `conduct_room_survey` を呼び出し、パノラマ・複数アングル画像を撮影。
  - 各アングルでの異変（不審物、ドアの開閉等）を分析・検証。

- **特定対象の自動追跡（Target Tracking）**:
  - 人物やペットが検知された際、`set_tracking_target` でターゲット指定。
  - 画角中央から外れそうになった場合、自動的に `move_camera` で調整追従。

- **トリガーイベント通知 & 記録**:
  - イベント検知時（侵入検知等）に `get_live_snapshot` で証拠画像を保存し、ナレッジDBへコンテキストを蓄積。

## 3. 制約・注意事項

- リアルタイム性が求められる連続追跡シナリオでは、ネットワーク帯域とレスポンス遅延に注意が必要。

## 4. 関連概念

* [MCP Specification](../architecture/mcp_specification.md) - 利用するMCPツールのインターフェース定義
* [Memory CLI Specification](./memory_cli.md) - イベントログやコンテキストの保存先仕様
