---
name: generate-tests
description: Analyze PR changes and auto-generate unit tests to achieve diff coverage >= 80%
triggers:
  - generate tests
  - add tests
  - unit test
  - diff coverage
  - test coverage
  - write tests
argument-hint: "[base_branch]"
---

# Generate Tests Skill

Automatically analyze code changes on the current branch, generate high-quality unit tests, and iterate until diff coverage >= 80%.

## Input

- `$ARGUMENTS` — Optional base branch name (auto-detected if omitted)

## Workflow (4 phases, max 3 iterations)

Set `BASE_BRANCH` = `$ARGUMENTS` (if empty, `ci/run-tests-and-coverage.py` auto-detects).

---

### Phase 1: ANALYZER — Analyze Changes

1. Run `git diff $(git merge-base HEAD origin/${BASE_BRANCH:-main})..HEAD --name-only -- 'datus/**/*.py'` to get the list of changed files.
2. Filter out paths matching coverage omit config from `pyproject.toml`:
   - `datus/prompts/prompt_templates/*`
3. For each changed file, read the source and check whether a corresponding test file already exists (see mapping rule below).
4. On iteration 2+, focus on `violation_lines` from `ci/diff-cover.json` (uncovered lines).
5. Output: list of files needing tests + test gap analysis per file.

**File mapping rule**: `datus/a/b/c.py` -> `tests/unit_tests/a/b/test_c.py`

---

### Phase 2: TEST WRITER — Write Tests

#### Core Principle: NO MOCK EXCEPT LLM

From `tests/unit_tests/conftest.py` line 8:

> Design principle: NO mock except LLM.
> - Real AgentConfig (from config dict)
> - Real SQLite database (in tmp_path)
> - Real db_manager_instance (connecting to real SQLite)
> - Real Storage/RAG (LanceDB in tmp_path)
> - Real Tools (DBFuncTool, ContextSearchTools, etc.)
> - Real PromptManager (using built-in templates)
> - Real PathManager
>
> The ONLY allowed mock: LLMBaseModel.create_model -> returns MockLLMModel

#### Fixture Mapping Table

Select fixtures based on source path:

| Source Path Pattern | Required Fixtures | Test Characteristics |
|---|---|---|
| `datus/agent/node/*_agentic_node.py` | `real_agent_config`, `mock_llm_create` | async, MockToolCall, execute_stream |
| `datus/agent/node/*.py` (non-agentic) | `real_agent_config`, `mock_llm_create` | sync, node.run() |
| `datus/storage/**/*.py` | `tmp_path` | real LanceDB, real embeddings |
| `datus/tools/**/*.py` | `real_agent_config` | real tool execution |
| `datus/utils/**/*.py` | none / `tmp_path` | parametrize, pure functions |
| `datus/models/**/*.py` | `mock_llm_create` | model creation and invocation |
| `datus/schemas/**/*.py` | none | Pydantic validation, no fixtures needed |
| `datus/configuration/**/*.py` | `tmp_path` | config loading and validation |

#### Available Mock Utilities

- **`MockLLMModel`** — from `tests/unit_tests/mock_llm_model.py`
  - `MockToolCall(name="tool_name", arguments="{}")` — simulate LLM deciding to call a tool (tool is executed for real)
  - `MockLLMResponse(content="...", tool_calls=[...], thinking="...")` — one complete LLM response turn
  - `build_simple_response(content)` — quickly build a response with no tool calls
  - `build_sql_response(sql, tables, explanation, tool_calls)` — build a SQL result response
  - `build_tool_then_response(tool_calls, content, thinking)` — build a tool-call-then-response turn
- **`mock_llm_create` fixture** (`conftest.py:185`) — patches `LLMBaseModel.create_model` to return `MockLLMModel`; use `model.reset(responses=[...])` to configure response sequence
- **`real_agent_config` fixture** (`conftest.py:89`) — includes real SQLite database (california_schools), real namespace config

#### Writing Standards

1. Test file location: strictly follow mapping rule `datus/a/b/c.py` -> `tests/unit_tests/a/b/test_c.py`
2. Create `__init__.py` files when necessary
3. Test class organization:
   - `TestXxxInit` — initialization tests
   - `TestXxxExecution` — main flow tests
   - `TestXxxEdgeCases` — edge cases
4. Naming convention: `test_<component>_<scenario>_<expected_behavior>`
5. Each test method must have a brief docstring
6. Reference example: `tests/unit_tests/agent/node/test_gen_sql_agentic_node.py`
7. Code style: Black 120 char line width, isort profile=black

---

### Phase 3: QUALITY REVIEWER — Quality Review

Review each test file produced in Phase 2 against these 9 criteria:

1. **Meaningful assertions** — at least 2 meaningful `assert` statements per test (excluding `assert True`)
2. **Actual invocation** — test must actually call the code under test, not just test mock behavior
3. **Correct fixtures** — only use fixtures from the mapping table above; no ad-hoc mocking of non-LLM components
4. **Docstring** — each test method has a docstring describing test intent
5. **Naming convention** — `test_<component>_<scenario>` format
6. **Async correctness** — agentic node tests use `@pytest.mark.asyncio`, correctly `async for` to consume streams
7. **Error path** — at least one test covers an exception/error path
8. **No ad-hoc mocks** — no `unittest.mock.patch` on non-LLM objects (`mock_llm_create` is the only allowed mock)
9. **Real data assertions** — assert on real return values/attributes, no `assert result is not None` placeholders

**On failure**: return failing items with reasons back to Phase 2 for rewrite, max 2 retries.

---

### Phase 4: COVERAGE VERIFIER — Coverage Verification

1. Run tests with coverage:
   ```bash
   python3 ci/run-tests-and-coverage.py ${BASE_BRANCH}
   ```

2. Check test results:
   - If any tests failed, read `ci/test-report.md` for failure details, go back to Phase 2 to fix.

3. Parse coverage: read `ci/diff-cover.json`, extract `total_percent_covered` field.

4. Decision:
   - **`total_percent_covered` >= 80 AND all tests pass** -> **DONE**, output final report.
   - **`total_percent_covered` < 80** -> extract `violation_lines` per file from `ci/diff-cover.json`, go back to Phase 1 focused on those lines for the next iteration.

5. **Max 3 iterations**. If still below target after 3 rounds, output current coverage and analysis of remaining uncovered lines.

---

## Output

Final output includes:
- List of created/modified test files
- Test count per file
- Final diff coverage percentage
- If below target, remaining uncovered lines with analysis
