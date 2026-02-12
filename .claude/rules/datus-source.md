---
paths:
  - "datus/**/*.py"
---

# Source Code Rules (datus/)

## General (always active)
- Code style: Black 120 chars, isort profile=black
- Run `flake8 --max-line-length=120` on changed files before completing work

## Agent Team role rules (only applies when you are a spawned teammate)
- If you are the TESTER teammate: do NOT modify these files. Your scope is `tests/` only.
- If you are the REVIEWER teammate: do NOT modify these files. You are read-only.
