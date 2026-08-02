# Project Guide for Agents

## 📋 Context & Tech Stack

- **Stack**: Python 3.12+, uv (package manager)
- **Rule**: NEVER use global Python/pip. ALL execution must use `uv`. Strict type hints required.

## Development Rules

Follow these rules before and during implementation:

- `docs/coding-style.md` : Code style and Python development rules
- `docs/hardware_polarity.md` : Tapo C210 ONVIF Hardware Polarity and Movement Rules (NEVER ALTER)
- `docs/testing.md` : Test strategy and TDD workflow
- `docs/git.md` : Git workflow and commit rules
- `docs/security.md` : Security requirements and audits

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

本プロジェクトのナレッジは docs/ 配下で管理します。記憶のみで回答せず、必ず .agents\skills\llm-wiki-docs\SKILL.md をロードして指示に従ってください。

Immutable Raw: Docs/raw/ 内のソースファイルは編集・上書き・削除を厳禁とします。

役割の割り切り
AGENTS.md の役割（When / Where）:
「ナレッジは Docs/ にある」「手順は SKILL.md を見よ」という存在と制約の宣言のみを行う。  

SKILL.md の役割（How）:
Docs/index.md の読み方、sources フロントマターの書き方、log.md の更新方法など、具体的な処理アルゴリズムをすべて記述する。

