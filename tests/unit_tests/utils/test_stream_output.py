# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/stream_output.py."""

import pytest
from rich.console import Console

from datus.utils.stream_output import StreamOutputManager, create_stream_output_manager


@pytest.fixture
def console():
    return Console(force_terminal=True, width=120)


@pytest.fixture
def manager(console):
    return StreamOutputManager(console=console, max_message_lines=5, show_progress=True, title="Test")


class TestStreamOutputManager:
    def test_init(self, manager):
        assert manager.max_message_lines == 5
        assert manager.show_progress is True
        assert manager.title == "Test"
        assert not manager._is_running

    def test_create_progress_single_item(self, manager):
        progress = manager._create_progress(1)
        assert progress is not None

    def test_create_progress_multi_item(self, manager):
        progress = manager._create_progress(10)
        assert progress is not None

    def test_start_and_stop(self, manager):
        manager.start(total_items=5)
        assert manager._is_running
        assert manager.progress is not None
        assert manager.live is not None
        manager.stop()
        assert not manager._is_running

    def test_start_twice_no_op(self, manager):
        manager.start(total_items=3)
        manager.start(total_items=3)  # should be no-op
        assert manager._is_running
        manager.stop()

    def test_stop_when_not_running(self, manager):
        manager.stop()  # should not raise
        assert not manager._is_running

    def test_update_progress(self, manager):
        manager.start(total_items=5)
        manager.update_progress(advance=1, description="Step 1")
        manager.stop()

    def test_set_progress(self, manager):
        manager.start(total_items=10)
        manager.set_progress(completed=5, description="Halfway")
        manager.stop()

    def test_update_progress_not_started(self, manager):
        # Should not raise even if progress not started
        manager.update_progress(advance=1)

    def test_set_progress_not_started(self, manager):
        manager.set_progress(completed=3)

    def test_start_file(self, manager):
        manager.start(total_items=2)
        manager.start_file("test.sql", total_items=10)
        assert manager.current_file == "test.sql"
        assert manager.task_number == 0
        manager.stop()

    def test_complete_file(self, manager):
        manager.start(total_items=1)
        manager.start_file("test.sql")
        manager.complete_file("test.sql")
        assert manager.current_file == ""
        manager.stop()

    def test_start_task(self, manager):
        manager.start(total_items=1)
        manager.start_task("process item")
        assert manager.task_number == 1
        assert "[1]" in manager.current_task
        manager.stop()

    def test_add_message(self, manager):
        manager.start(total_items=1)
        manager.add_message("test message", style="green")
        assert len(manager.messages) == 1
        manager.stop()

    def test_add_empty_message(self, manager):
        manager.add_message("")
        assert len(manager.messages) == 0

    def test_add_multiline_message(self, manager):
        manager.add_message("line 1\nline 2\nline 3")
        assert len(manager.messages) == 3

    def test_add_llm_output(self, manager):
        manager.add_llm_output("LLM response text")
        assert len(manager.full_output) == 1
        assert len(manager.messages) == 1

    def test_complete_task_success(self, manager):
        manager.start(total_items=1)
        manager.start_task("do work")
        manager.complete_task(success=True, message="done")
        assert manager.current_task == ""
        manager.stop()

    def test_complete_task_failure(self, manager):
        manager.start(total_items=1)
        manager.start_task("do work")
        manager.complete_task(success=False, message="failed")
        assert manager.current_task == ""
        manager.stop()

    def test_complete_task_no_message(self, manager):
        manager.start(total_items=1)
        manager.start_task("do work")
        manager.complete_task()
        assert manager.current_task == ""
        manager.stop()

    def test_error(self, manager):
        manager.error("something went wrong")
        assert len(manager.messages) == 1

    def test_warning(self, manager):
        manager.warning("be careful")
        assert len(manager.messages) == 1

    def test_success(self, manager):
        manager.success("all good")
        assert len(manager.messages) == 1

    def test_render_markdown_summary_empty(self, manager):
        manager.render_markdown_summary()  # should not raise

    def test_render_markdown_summary_with_json_output(self, console):
        mgr = StreamOutputManager(console=console)
        mgr.full_output.append('{"output": "# Summary\\nThis is markdown."}')
        mgr.render_markdown_summary(title="Test Summary")
        assert len(mgr.full_output) == 0  # cleared after rendering

    def test_render_markdown_summary_plain_text(self, console):
        mgr = StreamOutputManager(console=console)
        mgr.full_output.append("just plain text without json")
        mgr.render_markdown_summary()
        assert len(mgr.full_output) == 0

    def test_render_markdown_summary_with_last_n(self, console):
        mgr = StreamOutputManager(console=console)
        mgr.full_output.append("text1")
        mgr.full_output.append("text2")
        mgr.render_markdown_summary(last_n=1)

    def test_extract_all_markdown_outputs_no_json(self, manager):
        result = manager._extract_all_markdown_outputs("plain text without json")
        assert result == ["plain text without json"]

    def test_extract_all_markdown_outputs_with_json(self, manager):
        text = '{"output": "markdown content"}'
        result = manager._extract_all_markdown_outputs(text)
        assert result == ["markdown content"]

    def test_extract_all_markdown_outputs_invalid_json(self, manager):
        text = '{"output": "valid"} then {"invalid json'
        result = manager._extract_all_markdown_outputs(text)
        assert "valid" in result

    def test_extract_all_markdown_outputs_empty_output(self, manager):
        text = '{"output": ""}'
        result = manager._extract_all_markdown_outputs(text)
        # empty output field -> falls back to original text
        assert len(result) > 0

    def test_render_produces_group(self, manager):
        result = manager._render()
        assert result is not None

    def test_render_with_components(self, manager):
        manager.start(total_items=3)
        manager.start_file("test.py")
        manager.start_task("process")
        manager.add_message("msg")
        result = manager._render()
        assert result is not None
        manager.stop()

    def test_task_context_success(self, manager):
        manager.start(total_items=1)
        with manager.task_context("do something"):
            manager.add_message("working...")
        assert manager.current_task == ""
        manager.stop()

    def test_task_context_failure(self, manager):
        manager.start(total_items=1)
        with pytest.raises(ValueError):
            with manager.task_context("do something"):
                raise ValueError("oops")
        manager.stop()

    def test_file_context_success(self, manager):
        manager.start(total_items=1)
        with manager.file_context("data.sql", total_items=5):
            assert manager.current_file == "data.sql"
        assert manager.current_file == ""
        manager.stop()

    def test_file_context_failure(self, manager):
        manager.start(total_items=1)
        with pytest.raises(RuntimeError):
            with manager.file_context("data.sql"):
                raise RuntimeError("file error")
        assert manager.current_file == ""
        manager.stop()


class TestCreateStreamOutputManager:
    def test_factory_function(self, console):
        mgr = create_stream_output_manager(
            console=console,
            max_message_lines=20,
            show_progress=False,
            title="Custom",
        )
        assert isinstance(mgr, StreamOutputManager)
        assert mgr.max_message_lines == 20
        assert mgr.show_progress is False
        assert mgr.title == "Custom"
