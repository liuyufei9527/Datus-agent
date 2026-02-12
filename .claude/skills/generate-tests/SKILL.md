---
name: generate-tests
description: Discover related tests for branch changes, run with coverage, and generate new tests only when diff coverage < 80%
triggers:
  - generate tests
  - add tests
  - unit test
  - diff coverage
  - test coverage
  - run tests
  - related tests
argument-hint: "[base_branch]"
---

# Generate Tests Skill

Analyze code changes on the current branch, discover all related existing tests (plus any newly added tests), run them with coverage measurement, and only generate new test cases when diff coverage is below 80%.

## Input

- `$ARGUMENTS` — Optional base branch name (auto-detected if omitted)

## Workflow (5 phases, max 3 iterations)

Set `BASE_BRANCH` = `$ARGUMENTS` (if empty, `ci/run-tests-and-coverage.py` auto-detects).

---

### Phase 1: DISCOVER — Analyze Changes & Find Related Tests

#### Step 1.1: Identify all changed files

```bash
MERGE_BASE=$(git merge-base HEAD origin/${BASE_BRANCH:-main})
git diff ${MERGE_BASE}..HEAD --name-only
```

Categorize into:
- **Changed source files**: paths matching `datus/**/*.py` (exclude `datus/prompts/prompt_templates/*`)
- **Changed/new test files**: paths matching `tests/**/*.py`

#### Step 1.2: Collect newly added/modified test files (ALWAYS included)

```bash
git diff ${MERGE_BASE}..HEAD --name-only -- 'tests/**/*.py'
```

These test files are **always included** in the final test set, tagged as `[NEW]`.

#### Step 1.3: Direct mapping discovery

For each changed source file, check if a corresponding test file exists on disk:

**File mapping rule**: `datus/a/b/c.py` → `tests/unit_tests/a/b/test_c.py`

If the mapped test file exists, add it to the test set, tagged as `[MAPPED]`.

#### Step 1.4: Import-based discovery

For each changed source file, derive the module path (e.g., `datus/utils/json_utils.py` → `datus.utils.json_utils`), then search for test files that import it:

```bash
grep -rl "from datus\.utils\.json_utils import\|import datus\.utils\.json_utils" tests/unit_tests/
```

Add discovered test files tagged as `[IMPORT]`.

#### Step 1.5: Conftest & fixture impact analysis

If any `conftest.py` was changed, read the diff to identify changed fixtures, then search for test files using those fixtures. Add them tagged as `[FIXTURE]`.

#### Step 1.6: Deduplicate & validate

- Merge all discovered test files into a deduplicated list.
- Verify each file exists on disk.
- Output the final list with discovery reason tags.

**If no related test files are found**, skip directly to Phase 3 (generate new tests).

---

### Phase 2: COVERAGE CHECK — Run Related Tests with Coverage

Run all discovered test files with coverage measurement:

```bash
python3 ci/run-tests-and-coverage.py ${BASE_BRANCH} --test-paths <all_discovered_test_files>
```

Only include test files that actually exist on disk.

Parse results:
1. Read `ci/test-report.md` for test pass/fail details.
2. Read `ci/diff-cover.json`, extract `total_percent_covered`.

**Decision**:
- **All tests pass AND `total_percent_covered` >= 80** → Go to Phase 5 (report success).
- **Tests fail** → Report failures. If failures are in `[NEW]` test files (from this branch), go to Phase 3 to fix them. If failures are in existing tests, report as potential regression and still proceed to check coverage.
- **`total_percent_covered` < 80** → Extract `violation_lines` per file from `ci/diff-cover.json`, proceed to Phase 3 to generate tests for uncovered lines.

---

### Phase 3: TEST WRITER — Generate Tests for Uncovered Lines

**Only reached when diff coverage < 80% or no existing tests cover the changes.**

On iteration 2+, focus specifically on `violation_lines` from `ci/diff-cover.json`.

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

### Phase 4: QUALITY REVIEWER — Quality Review (only for newly generated tests)

Review each test file produced in Phase 3 against these 9 criteria:

1. **Meaningful assertions** — at least 2 meaningful `assert` statements per test (excluding `assert True`)
2. **Actual invocation** — test must actually call the code under test, not just test mock behavior
3. **Correct fixtures** — only use fixtures from the mapping table above; no ad-hoc mocking of non-LLM components
4. **Docstring** — each test method has a docstring describing test intent
5. **Naming convention** — `test_<component>_<scenario>` format
6. **Async correctness** — agentic node tests use `@pytest.mark.asyncio`, correctly `async for` to consume streams
7. **Error path** — at least one test covers an exception/error path
8. **No ad-hoc mocks** — no `unittest.mock.patch` on non-LLM objects (`mock_llm_create` is the only allowed mock)
9. **Real data assertions** — assert on real return values/attributes, no `assert result is not None` placeholders

**On failure**: return failing items with reasons back to Phase 3 for rewrite, max 2 retries.

After review passes, go back to Phase 2 to re-run coverage with all tests (discovered + newly generated).

---

### Phase 5: REPORT — Final Output

**Max 3 iterations** of Phase 2→3→4 loop. If still below target after 3 rounds, output current state.

Final output includes:

- **Discovered tests**: list of related existing test files with discovery tags (`[NEW]`, `[MAPPED]`, `[IMPORT]`, `[FIXTURE]`)
- **Generated tests** (if any): list of newly created/modified test files with test count per file
- **Test results**: pass/fail/error/skip counts
- **Diff coverage**: final percentage
- If below target: remaining uncovered lines with analysis
