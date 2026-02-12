# Language Rules

- Respond in the same language as the user's question. Any language is acceptable in conversation.
- All code (including comments, variable names, function names, etc.) MUST be written in English.

# Test Generation

- After completing code development, use `/generate-tests` to automatically analyze branch changes and generate unit tests targeting diff coverage >= 80%.
- The skill runs up to 3 iterations of: analyze changes → write tests → quality review → coverage verification.
- See `.claude/skills/generate-tests/SKILL.md` for details

# Code Formatting

- After completing any code development task (new features, bug fixes, refactoring), ALWAYS run `/format-code` to format the changed files before committing.
- This ensures all code passes CI format checks (Black, isort, flake8).
- See `.claude/skills/format-code/SKILL.md` for details
