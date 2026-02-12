---
name: team-dev
description: Start 3-role agent team workflow (Developer → Tester → Reviewer)
triggers:
  - team dev
  - spawn tester
  - spawn reviewer
  - agent team
argument-hint: "[base_branch]"
---

# 3-Role Agent Team Workflow

After development is complete, sequentially spawn tester and reviewer teammates to achieve test coverage and code review.

## Input

- `$ARGUMENTS` — Optional base branch name (defaults to main)

## Step 1: Spawn Tester

spawn teammate named "tester":
"You are the TESTER role. You may ONLY modify files under the tests/ directory. You MUST NOT modify any file outside tests/.
Check current branch changes: git diff $(git merge-base HEAD origin/${ARGUMENTS:-main})..HEAD --name-only -- 'datus/**/*.py'
Follow the workflow in .claude/skills/generate-tests/SKILL.md to add test coverage.
Run python3 ci/run-tests-and-coverage.py to check results.
Run python3 ci/preview-comment.py to preview coverage reports.
You may use pytest tests/unit_tests/path/to/test_file.py -xvs to quickly validate individual files.
Iterate until all tests pass (100% pass rate) AND diff coverage >= 80%.
Report your final results when done."

Wait for the tester to complete and report results.

## Step 2: Spawn Reviewer

After the tester meets targets, spawn reviewer:

spawn teammate named "reviewer":
"You are the REVIEWER role. You are strictly READ-ONLY.
You MUST NOT create, modify, or delete any file. You MUST NOT use Write or Edit tools.
Review all changes on this branch: git diff $(git merge-base HEAD origin/${ARGUMENTS:-main})..HEAD
Check the following:
1. Code changes in datus/ are well-structured and follow project conventions
2. Tests in tests/ are real and meaningful (not just mocks), per the 9 criteria in .claude/skills/generate-tests/SKILL.md Phase 3
3. Test coverage is adequate
If you find issues, message the 'developer' or 'tester' teammate with specific fix suggestions. If everything looks good, report approval to the lead."

## Step 3: Handle Feedback

- If reviewer finds code issues → developer (you) fixes them
- If reviewer finds test issues → notify tester to fix
- All clear → workflow complete

## References
- Testing conventions: `.claude/skills/generate-tests/SKILL.md`
- File mapping: `datus/a/b/c.py` -> `tests/unit_tests/a/b/test_c.py`
- CI scripts: `ci/run-tests-and-coverage.py`, `ci/preview-comment.py`
- Code style: Black 120 chars, isort profile=black, Flake8 max 120
- Test principle: NO mock except LLM (see `tests/unit_tests/conftest.py`)
