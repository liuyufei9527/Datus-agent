# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for datus/utils/async_utils.py to improve coverage."""

import asyncio
import threading

import pytest

from datus.utils.async_utils import (
    get_or_create_event_loop,
    is_event_loop_running,
    run_async,
    setup_windows_policy,
)


class TestSetupWindowsPolicy:
    def test_non_windows_no_op(self):
        # On non-Windows platforms, this should be a no-op
        setup_windows_policy()


class TestIsEventLoopRunning:
    def test_no_loop_running(self):
        # In a sync context, no loop should be running
        assert is_event_loop_running() is False

    @pytest.mark.asyncio
    async def test_loop_is_running(self):
        assert is_event_loop_running() is True


class TestGetOrCreateEventLoop:
    def test_get_existing_loop(self):
        loop = get_or_create_event_loop()
        assert loop is not None
        assert not loop.is_closed()

    def test_creates_new_if_closed(self):
        # Create a loop and close it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.close()

        # get_or_create_event_loop should create a new one
        new_loop = get_or_create_event_loop()
        assert new_loop is not None
        assert not new_loop.is_closed()
        # Cleanup
        new_loop.close()
        asyncio.set_event_loop(None)


class TestRunAsync:
    def test_simple_coroutine(self):
        async def simple():
            return 42

        result = run_async(simple())
        assert result == 42

    def test_with_timeout(self):
        async def fast():
            await asyncio.sleep(0.01)
            return "done"

        result = run_async(fast(), timeout=5.0)
        assert result == "done"

    def test_timeout_exceeded(self):
        async def slow():
            await asyncio.sleep(10)
            return "never"

        with pytest.raises((asyncio.TimeoutError, Exception)):
            run_async(slow(), timeout=0.1)

    def test_exception_propagation(self):
        async def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_async(failing())

    def test_run_from_thread(self):
        async def threaded_coro():
            await asyncio.sleep(0.01)
            return "from_thread"

        result = [None]
        error = [None]

        def thread_target():
            try:
                result[0] = run_async(threaded_coro())
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=thread_target)
        t.start()
        t.join(timeout=5)

        assert error[0] is None
        assert result[0] == "from_thread"

    def test_coroutine_returning_none(self):
        async def returns_none():
            return None

        result = run_async(returns_none())
        assert result is None

    def test_coroutine_with_args(self):
        async def add(a, b):
            return a + b

        result = run_async(add(3, 4))
        assert result == 7

    @pytest.mark.asyncio
    async def test_run_async_from_running_loop(self):
        """Test run_async when called from inside a running event loop.
        This exercises the _run_in_thread path (lines 109-110, 216-305)."""

        async def inner_coro():
            return "inner_result"

        # run_async is called from within an async function (loop is running)
        result = run_async(inner_coro())
        assert result == "inner_result"

    @pytest.mark.asyncio
    async def test_run_async_from_running_loop_with_timeout(self):
        """Test _run_in_thread with timeout from running loop."""

        async def inner_coro():
            await asyncio.sleep(0.01)
            return "timed_result"

        result = run_async(inner_coro(), timeout=5.0)
        assert result == "timed_result"

    @pytest.mark.asyncio
    async def test_run_async_from_running_loop_exception(self):
        """Test _run_in_thread exception propagation from running loop."""

        async def inner_failing():
            raise RuntimeError("inner failure")

        with pytest.raises(RuntimeError, match="inner failure"):
            run_async(inner_failing())

    @pytest.mark.asyncio
    async def test_run_async_from_running_loop_timeout(self):
        """Test _run_in_thread timeout from running loop."""

        async def slow_inner():
            await asyncio.sleep(10)
            return "never"

        with pytest.raises((asyncio.TimeoutError, Exception)):
            run_async(slow_inner(), timeout=0.2)

    def test_nested_run_async(self):
        """Test nested run_async calls (exercises the nested detection path, line 103-104)."""
        import datus.utils.async_utils as async_mod

        async def outer():
            # Simulate nested call by setting _local.in_run_async
            async_mod._local.in_run_async = True
            try:
                return run_async(inner())
            finally:
                async_mod._local.in_run_async = False

        async def inner():
            return "nested_result"

        # First call goes through _run_in_new_loop, which sets in_run_async
        # Inner run_async should detect nesting and use _run_in_thread
        result = run_async(outer())
        assert result == "nested_result"

    def test_get_or_create_creates_new_when_no_loop(self):
        """Test creating event loop when none exists (lines 73-78)."""
        # Remove any event loop for this thread
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except RuntimeError:
            pass
        asyncio.set_event_loop(None)

        # Should create a new one
        from datus.utils.async_utils import get_or_create_event_loop

        new_loop = get_or_create_event_loop()
        assert new_loop is not None
        assert not new_loop.is_closed()
        # Cleanup
        new_loop.close()
        asyncio.set_event_loop(None)
