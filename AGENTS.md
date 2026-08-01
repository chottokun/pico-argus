# Project Guide for Agents

## 📋 Context & Tech Stack

- **Stack**: Python 3.12+, uv (package manager)
- **Rule**: NEVER use global Python/pip. ALL execution must use `uv`. Strict type hints required.

## 🔄 TDD, Critical Testing & Doc Cycle

1. **Red**: Write a failing test for the basic requirement first.
2. **Green**: Write the minimum code to make it pass.
3. **Critical & Real Test**: Add tests for edge cases. If real-world sample data is available, use it within a realistic scope. 
   - *Strict Rule*: Anonymize/mask all personal data and credentials before using real data. Never commit raw production secrets.
4. **Refactor & Doc**: Clean code, add Google-style docstrings, and update `README.md`.

## 🚨 Constraints

- **Secrets**: NEVER hardcode real keys/tokens. Gitleaks scanning is enforced. Use fake placeholders (e.g., `YOUR_KEY`).
- **uv audit**: Check for dependency vulnerabilities before finishing tasks.
- **Scope**: Fix only the targeted issue. Never refactor unrelated files or change working architectures.
- **Dependencies**: Prefer standard libraries. Do not add new packages unless absolutely necessary.
- **Git**: Use Conventional Commits (e.g., `feat:`, `fix:`). Commit frequently in small, logical increments (one feature/fix per commit). Push only after all local tests and security checks pass.

## 🛠️ Commands

- **Setup**: `uv sync`
- **Lint/Format**: `uv run ruff check --fix`
- **Test**: `uv run pytest`
- **Security Check**: `uv audit` && gitleaks detect --verbose --source=.

## ✅ Definition of Done

Task is complete only when all Security Checks and Tests pass, dependencies are clean, and documentation/Git history matches this guide.

## 情報源

実装時は次の順で参照する。

1. リポジトリ内のコード
2. docs/
3. README.md
4. 公式ドキュメント

docs/, README.mdは必要な場合には更新し最新の情報とすること

## Knowledge Rules: 

本プロジェクトのナレッジは Docs/ 配下で管理します。記憶のみで回答せず、必ず .agents\skills\llm-wiki-docs\SKILL.md をロードして指示に従ってください。

Immutable Raw: Docs/raw/ 内のソースファイルは編集・上書き・削除を厳禁とします。

役割の割り切り
AGENTS.md の役割（When / Where）:
「ナレッジは Docs/ にある」「手順は SKILL.md を見よ」という存在と制約の宣言のみを行う。  

SKILL.md の役割（How）:
Docs/index.md の読み方、sources フロントマターの書き方、log.md の更新方法など、具体的な処理アルゴリズムをすべて記述する。