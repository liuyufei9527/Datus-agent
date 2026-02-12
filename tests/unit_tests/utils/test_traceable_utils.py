# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/traceable_utils.py."""

import datus.utils.traceable_utils as traceable_mod
from datus.utils.traceable_utils import get_trace_url, optional_traceable, setup_tracing


class TestOptionalTraceable:
    def test_decorator_returns_function(self):
        @optional_traceable(name="test_op")
        def my_func():
            return 42

        assert callable(my_func)
        assert my_func() == 42

    def test_decorator_default_args(self):
        @optional_traceable()
        def my_func(x):
            return x + 1

        assert my_func(5) == 6

    def test_decorator_with_run_type(self):
        @optional_traceable(name="retrieval", run_type="retriever")
        def retrieve(query):
            return query

        assert retrieve("test") == "test"


class TestSetupTracing:
    def test_setup_tracing_idempotent(self):
        # Reset state
        original = traceable_mod._tracing_initialized
        traceable_mod._tracing_initialized = False
        try:
            setup_tracing()
            # Second call should be no-op
            setup_tracing()
        finally:
            traceable_mod._tracing_initialized = original

    def test_setup_tracing_no_langsmith(self):
        original_flag = traceable_mod.HAS_LANGSMITH
        original_init = traceable_mod._tracing_initialized
        try:
            traceable_mod.HAS_LANGSMITH = False
            traceable_mod._tracing_initialized = False
            setup_tracing()
            # Should have initialized but done nothing
            assert traceable_mod._tracing_initialized is True
        finally:
            traceable_mod.HAS_LANGSMITH = original_flag
            traceable_mod._tracing_initialized = original_init


class TestGetTraceUrl:
    def test_returns_none_when_no_processor(self):
        original = traceable_mod._tracing_processor
        try:
            traceable_mod._tracing_processor = None
            assert get_trace_url() is None
        finally:
            traceable_mod._tracing_processor = original

    def test_returns_url_from_processor(self):
        original = traceable_mod._tracing_processor

        class FakeProcessor:
            _last_trace_url = "https://example.com/trace/123"

        try:
            traceable_mod._tracing_processor = FakeProcessor()
            assert get_trace_url() == "https://example.com/trace/123"
        finally:
            traceable_mod._tracing_processor = original
