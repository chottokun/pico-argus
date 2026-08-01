---
type: Concept
title: Memory CLI Specification
description: Pico-Argus システムにおける対話履歴・監視ナレッジの永続化およびCLI操作仕様
status: active
timestamp: 2026-08-01T10:30:00+09:00
tags:
  - memory
  - cli
  - sqlite
  - knowledge
sources:
  - id: memory_cli_doc
    resource: /Docs/raw/memory_cli_specification.md
    title: Memory CLI Specification Document
---

# Memory CLI Specification

## 1. 概要

`memory` モジュールは SQLite やベクトル検索（またはキーワード検索）を用いて、過去のカメラ検知イベント、会話コンテキスト、環境設定のナレッジを蓄積・検索するための仕様です。

## 2. コマンド構造・データ構造

- **CLIコマンド**:
  - `memory search <query>`: ナレッジベース内の関連エントリーを検索。
  - `memory add --type <type> --content <content>`: 新規ナレッジの追加。
  - `memory list`: 登録済みメモリ一覧の表示。

- **バックエンドストア**:
  - `wiki.db` (SQLiteデータベース): ナレッジエントリー、メタデータ、インデックスを管理。

## 3. 制約・注意事項

- データ削除・更新時はリファレンス整合性を保つこと。
- 個人情報や不要なバイナリデータはテキストDBに直接含めず、パスで管理すること。

## 4. 関連概念

* [MCP Usecases](./mcp_usecases.md) - メモリ検索・保存を活用するユースケース
