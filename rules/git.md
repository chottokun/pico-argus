# Git 運用ルール

## 📝 1. ブランチ運用

* `main` ブランチへ直接コミットしない
* 機能・修正ごとにブランチを作成する
* ブランチ名は内容が分かる名前を付ける

**例**

  * `feature/pid-control`
  * `fix/tracker-buffer`
  * `docs/mcp-spec`

---

## 📝 2. コミットメッセージ規約 (Conventional Commits)

コミットメッセージは **Conventional Commits** に準拠し、変更内容が分かるように記述してください。

* `feat`: 新機能の追加
* `fix`: バグ修正
* `docs`: ドキュメントの変更
* `style`: フォーマットのみの変更（動作変更なし）
* `refactor`: 機能変更を伴わないコード整理
* `test`: テストの追加・修正
* `chore`: ビルド・設定・依存関係などの変更

**例**

```text
feat: add PID slew rate control
fix: prevent memory leak in tracker buffer
docs: update MCP tool specification
```

---

## 📝 3. コミットのベストプラクティス

* **1コミット = 1つの目的** を徹底する
* 小さな単位でこまめにコミットする
* コミット前にビルド・テスト・Lint を実行する
* 不要なファイルや機密情報はコミットしない
* 意味のあるコミットメッセージを付ける

---

## 📝 4. Pull Request

* PR は小さく、レビューしやすい単位にする
* マージ前に最新の `main` を取り込む
* レビュー後は必要に応じて Squash し、履歴を整理する
