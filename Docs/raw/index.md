# Pico Argus ドキュメント総合インデックス

`Pico Argus`（LLM / マルチモーダル AI 統括型「能動的知覚」エッジ AI システム）の仕様書および技術ドキュメントの総合案内です。

---

## 📚 ドキュメント一覧

| ドキュメント | 概要・対象 | 主な内容 |
| :--- | :--- | :--- |
| **[camera_agent.md](file:///e:/Python%20Scripts/Pico/docs/camera_agent.md)** | 能動的知覚 (Active Perception) 詳細設計仕様書 | ONVIF PTZ 物理制御、PID サーボロックオン、安全クランプ制限、YOLO ONNX + ByteTrack テキスト化知覚バッファ設計 |
| **[mcp_specification.md](file:///e:/Python%20Scripts/Pico/docs/mcp_specification.md)** | MCP (Model Context Protocol) サーバー仕様書 | Claude Code / Claude Desktop 等と連携する全 8 種類の MCP ツール詳細と API プロトコル |
| **[mcp_usecases.md](file:///e:/Python%20Scripts/Pico/docs/mcp_usecases.md)** | MCP ユースケース・シナリオ集 | 自律認知・能動発火イベント、スポット視覚解釈、ナレッジグラフ検索の実用シナリオ |
| **[memory_cli_specification.md](file:///e:/Python%20Scripts/Pico/docs/memory_cli_specification.md)** | 長期記憶 (OKF) & ナレッジグラフ仕様書 | SQLite 3.34+ FTS5 Trigram 日本語検索、Obsidian WikiLinks (`[[...]]`) 自動相互リンク & バックリンク形成アルゴリズム |

---

## 🏗️ 全体アーキテクチャの概要

Pico Argus は 3 つの自律 CLI ツールおよび MCP サーバーで構成されています：

1. **筋肉 (`ptz-cli` / `pico.cli.ptz`)**: PTZ モーター物理制御、PID サーボ。
2. **感覚 (`perception-cli` / `pico.cli.perception`)**: YOLO 追跡 ＋ Ollama VLM クロップ解析。
3. **記憶 (`memory-cli` / `pico.cli.memory`)**: SQLite FTS5 ＋ OKF Markdown 記憶網。
4. **統括 (`pico.mcp.server`)**: Claude 等の LLM エージェントを司令塔として接続。

---

## 🔗 ルートガイドラインへのリンク

- [README.md](file:///e:/Python%20Scripts/Pico/README.md) - プロジェクトトップページ & クイックスタート
- [LICENSE](file:///e:/Python%20Scripts/Pico/LICENSE) - MIT ライセンス規定
- [THIRD_PARTY_LICENSES.md](file:///e:/Python%20Scripts/Pico/THIRD_PARTY_LICENSES.md) - 第三者ライブラリ・AIモデルライセンス一覧
- [CONTRIBUTING.md](file:///e:/Python%20Scripts/Pico/CONTRIBUTING.md) - 開発・テスト貢献ガイド
- [SECURITY.md](file:///e:/Python%20Scripts/Pico/SECURITY.md) - セキュリティ & プライバシーポリシー
