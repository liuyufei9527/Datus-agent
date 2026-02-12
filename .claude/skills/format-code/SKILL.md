---
name: format-code
description: Format Python code with Black, isort, and check with flake8
triggers:
  - format code
  - format
  - lint
  - black
  - isort
  - code style
argument-hint: "[path]"
---

# Format Code Skill

Format Python code to pass CI format checks (matching `python-format-check.yml`).

## Input

- `$ARGUMENTS` — Optional path to format (defaults to both `datus/` and `tests/`)

## Workflow

### Step 0: Check for relevant changes

If `$ARGUMENTS` is provided, skip this check and use `$ARGUMENTS` as `TARGET`.

If `$ARGUMENTS` is empty, detect whether any Python files under `datus/` or `tests/` have been modified on the current branch:

```bash
git diff --name-only HEAD $(git merge-base HEAD main) -- 'datus/**/*.py' 'tests/**/*.py'
```

- If the command returns **no files** -> print "No Python files changed under datus/ or tests/. Skipping format." and **stop here**.
- If the command returns files -> set `TARGET` to `datus/ tests/` and continue.

---

### Step 1: Format with isort

Sort imports using isort with black-compatible profile:

```bash
isort --profile=black --line-length=120 ${TARGET:-datus/ tests/}
```

Report the number of files modified.

---

### Step 2: Format with Black

Apply Black formatter with project settings:

```bash
black --line-length=120 --extend-exclude="/(mcp)/" ${TARGET:-datus/ tests/}
```

Report the number of files reformatted.

---

### Step 3: Lint with flake8

Run flake8 to check for remaining issues that auto-formatters cannot fix:

```bash
flake8 --max-line-length=120 --extend-ignore=E203,W503 ${TARGET:-datus/ tests/}
```

- If flake8 reports **no errors** -> formatting is complete.
- If flake8 reports **errors** -> display the errors and fix them manually, then re-run flake8 until clean.

---

## Output

Final output includes:
- Files modified by isort
- Files reformatted by Black
- flake8 result (pass or remaining issues)
- Summary of all changes made