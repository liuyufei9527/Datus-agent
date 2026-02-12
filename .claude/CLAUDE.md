# Language Rules

- Respond in the same language as the user's question. Any language is acceptable in conversation.
- All code (including comments, variable names, function names, etc.) MUST be written in English.

# Agent Team Workflow

This project supports a 3-role Agent Team workflow (Developer → Tester → Reviewer).
- Use `/team-dev` to start the workflow (spawns tester and reviewer teammates)
- See `.claude/skills/team-dev/SKILL.md` for details

# Code Formatting

- After completing any code development task (new features, bug fixes, refactoring), ALWAYS run `/format-code` to format the changed files before committing.
- This ensures all code passes CI format checks (Black, isort, flake8).
- See `.claude/skills/format-code/SKILL.md` for details
