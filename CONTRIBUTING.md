# 開発・貢献ガイドライン (Contributing Guide)

`Pico Argus` プロジェクトへのご貢献に感謝いたします！
バグ報告、機能提案、ドキュメントの改善、プルリクエストなど、あらゆる形でのコミュニティ貢献を歓迎します。

---

## 🛠️ 1. 開発環境のセットアップ

本プロジェクトでは、高速な Python パッケージマネージャーである [uv](https://github.com/astral-sh/uv) を使用しています。

```powershell
# 1. リポジトリのクローン
git clone https://github.com/chottokun/pico-argus.git
cd pico-argus

# 2. 仮想環境の作成と依存関係の同期
uv sync
```

---

## 🧪 2. テストとコード品質の検証

変更を加えた際は、必ず以下の検証コマンドを実行し、すべてのチェックを通過させてください。

```powershell
# 単体テスト・結合テストの実行
uv run pytest

# 静的コード解析 (Ruff)
uv run ruff check

# コードスタイルの自動修正
uv run ruff check --fix

# 依存パッケージのセキュリティ監査
uv audit
```

### 【厳格ルール】実データ・クレデンシャルの扱い
- テストデータやログに、実際のカメラパスワードや IP アドレス、秘密鍵を含めないでください。
- テストコード内ではモック（`unittest.mock` 等）またはダミー値 (`YOUR_KEY`, `192.168.0.10`) を使用してください。

---

## 📝 3. コミットメッセージ規約 (Conventional Commits)

コミットメッセージは [Conventional Commits](https://www.conventionalcommits.org/) に準拠した明確で小さな単位に分けて作成してください。

- `feat`: 新機能の追加 (`feat: add PID slew rate control`)
- `fix`: バグ修正 (`fix: prevent memory leak in tracker buffer`)
- `docs`: ドキュメントの変更 (`docs: update MCP tool specification`)
- `style`: コードの意味に影響を与えないフォーマットの変更
- `refactor`: バグ修正や機能追加を行わないコード整理
- `test`: テストコードの追加・修正
- `chore`: ビルドプロセスや補足ツールの変更

---

## 🔀 4. プルリクエスト (PR) 提出手順

1. 本リポジトリを Fork し、作業用のトピックブランチを作成します (`git checkout -b feat/my-new-feature`)。
2. 変更を加え、ローカルでテスト（`uv run pytest`, `uv run ruff check`）を完了させます。
3. 分かりやすいコミットメッセージでコミットします。
4. Fork 先のリポジトリに Push し、`main` ブランチに向けて Pull Request を送信します。
