# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/env.py."""

import os

from datus.utils.env import get_env_int


class TestGetEnvInt:
    def test_returns_default_when_not_set(self):
        key = "_TEST_ENV_INT_NOT_SET_XYZ"
        assert get_env_int(key) == 0

    def test_returns_custom_default(self):
        key = "_TEST_ENV_INT_NOT_SET_XYZ"
        assert get_env_int(key, default=42) == 42

    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("_TEST_ENV_INT_SET", "123")
        assert get_env_int("_TEST_ENV_INT_SET") == 123

    def test_returns_default_on_invalid(self, monkeypatch):
        monkeypatch.setenv("_TEST_ENV_INT_BAD", "not_a_number")
        assert get_env_int("_TEST_ENV_INT_BAD", default=99) == 99

    def test_negative_value(self, monkeypatch):
        monkeypatch.setenv("_TEST_ENV_INT_NEG", "-5")
        assert get_env_int("_TEST_ENV_INT_NEG") == -5

    def test_zero_value(self, monkeypatch):
        monkeypatch.setenv("_TEST_ENV_INT_ZERO", "0")
        assert get_env_int("_TEST_ENV_INT_ZERO", default=10) == 0

    def test_empty_string(self, monkeypatch):
        monkeypatch.setenv("_TEST_ENV_INT_EMPTY", "")
        assert get_env_int("_TEST_ENV_INT_EMPTY", default=7) == 7
