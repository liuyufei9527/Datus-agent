# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/typing_fix.py."""

from datus.utils.typing_fix import patch_agents_typing_issue


class TestPatchAgentsTypingIssue:
    def test_returns_bool(self):
        result = patch_agents_typing_issue()
        assert isinstance(result, bool)

    def test_callable(self):
        # Just ensure it can be called without error
        patch_agents_typing_issue()
