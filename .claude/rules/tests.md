---
paths:
  - "tests/**/*.py"
---

# Test Code Rules (tests/)

## General (always active)
- File mapping: `datus/a/b/c.py` -> `tests/unit_tests/a/b/test_c.py`
- Follow all conventions in `.claude/skills/generate-tests/SKILL.md`
- Design principle: NO mock except LLM (see `tests/unit_tests/conftest.py`)
- Create `__init__.py` files in new test directories as needed

## Agent Team role rules (only applies when you are a spawned teammate)
- If you are the TESTER teammate: this is your scope. Use `python3 ci/run-tests-and-coverage.py` to verify results. Target: 100% pass rate AND >= 80% diff coverage.
- If you are the DEVELOPER teammate (lead): do NOT modify test files here. Spawn a tester teammate instead.
- If you are the REVIEWER teammate: do NOT modify test files. You are read-only.
