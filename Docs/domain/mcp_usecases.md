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

### 2.1 室内全方位パノラマ環境調査 ＆ アイテムインベントリ自動構築
部屋の東西南北（または正面・左右・上部）へカメラを自動制御し、YOLO物体検出と VLM (視覚言語モデル) を組み合わせることで、室内の配置・家具・家電・備品の自動インベントリ（目録）を Wiki データベースへ保存します。
- **使用ツール**: `conduct_room_survey` ➔ `analyze_crop_image` ➔ `write_wiki`
- **活用例**: お出かけ前の室内点検（照明、電気機器の切り忘れ、窓の閉め忘れ等）、遺失物の探索。

### 2.2 特定オブジェクト・人物の自動物理PTZロックオン追尾
特定の人物（`person`）や特定の持ち物（`suitcase`, `handbag` 等）が画面内に現れた際、PIDサーボループによってカメラレンズを物理的に駆動し、常にオブジェクトを画面中央（スクエア）へ自動追従・補正し続けます。
- **使用ツール**: `set_tracking_target` ➔ `get_active_tracks` ➔ `get_perception_status`
- **活用例**: 室内見守り・自動フォロー、特定物品の持ち去り・動体監視。

### 2.3 アクティブ・高解像度クロップ VLM 深層精査
部屋の遠くにある小さな物体や気になるエリアを対象とし、YOLO BBOX でトリミング（クロップ）した高解像度画像を VLM に入力して、「何が置いてあるか」「どのような状態か」を精密に言語解釈させます。
- **使用ツール**: `get_active_tracks` ➔ `analyze_crop_image` (特定 track_id またはクラス指定)
- **活用例**: デスク上のコップの中身や、開いたままのノートの確認など、高解像度での精密視認。

### 2.4 過去の記憶想起と環境変化の自動検知 (Memory & Event Loop)
過去に記録した Wiki 記憶を FTS5 Trigram 全文検索し、「前回の調査時と比べて何が新しく増えたか」「位置が変わったか」をエージェントが自律比較・分析します。
- **使用ツール**: `search_wiki` ➔ `configure_event_filter` ➔ `write_wiki`
- **活用例**: 不在時の防犯・変化検知（新しく持ち込まれた物の発見）、文脈を考慮した会話応答（「ハサミどこに置いた？」への回答）。

### 2.5 ギア遊び誤差ゼロの物理原点自動校正 (Self-Healing Alignment)
外部の干渉や長時間のPID追尾運動によってカメラの物理的な角度推測が狂った際、物理左下限界壁へのブラインド突き当てと一方向正確復帰を行い、真の正面ド真ん中原点 `(0, 0)` へ自律復元します。
- **使用ツール**: `calibrate_home` ➔ `move_camera` (原点復旧)
- **活用例**: 定期メンテナンスによる累積運動誤差の完全リセット。

## 3. シナリオ別推奨連携パターン

| アクションシナリオ | 主な連携 MCP ツール | 期待される効果 |
|---|---|---|
| **定期環境調査** | `conduct_room_survey` ➔ `analyze_crop_image` ➔ `write_wiki` | パノラマスキャンから視覚言語化、Wiki蓄積まで自動完遂 |
| **人物・ペット見守り** | `set_tracking_target` ➔ `get_perception_status` | リアルタイムで対象を追従し健康状態を監視 |
| **アイテム探索・想起** | `search_wiki` ➔ `get_active_tracks` ➔ `analyze_crop_image` | 過去の記憶と現在の映像を照合し発見・報告 |
| **物理軸メンテナンス** | `calibrate_home` ➔ `move_camera` | ギア遊び誤差のない正確な原点姿勢を常に維持 |

## 4. 関連概念

* [MCP Specification](../architecture/mcp_specification.md) - 利用するMCPツールのインターフェース定義
* [Memory CLI Specification](./memory_cli.md) - イベントログやコンテキストの保存先仕様
