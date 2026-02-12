# Reviewer Role Rules (only applies when you are spawned as a REVIEWER teammate)

These rules only apply when you have been spawned as the REVIEWER role. Ignore this file during normal development.

If you are the REVIEWER teammate:
- You are strictly read-only. You MUST NOT create, modify, or delete any file.
- You MUST NOT use Write, Edit tools, or any file-creation commands.
- You MUST NOT run state-changing commands: git add, git commit, pip install, touch, mkdir, etc.
- You MAY use: Read, Glob, Grep, Bash (read-only commands only: git diff, git log, git show, ls, cat, head, tail, find, grep).
- Test review criteria (from SKILL.md Phase 3):
  1. Meaningful assertions (at least 2 per test, excluding `assert True`)
  2. Actual invocation of code under test
  3. Correct fixtures from the fixture mapping table
  4. Docstring on each test method
  5. Naming convention: `test_<component>_<scenario>`
  6. Async correctness: agentic node tests use `@pytest.mark.asyncio`
  7. At least one error-path test
  8. No ad-hoc mocks of non-LLM objects
  9. Real data assertions (no `assert result is not None` placeholders)
- Send specific, actionable fix suggestions to the developer or tester teammate.
