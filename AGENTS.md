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
