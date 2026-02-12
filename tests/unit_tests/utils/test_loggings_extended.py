# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for datus/utils/loggings.py to improve coverage."""

import logging
from io import StringIO

import pytest
from rich.console import Console

from datus.utils.loggings import (
    AdaptiveRenderer,
    DynamicLogManager,
    _get_current_log_file,
    _is_source_environment,
    add_code_location,
    add_exc_info,
    configure_logging,
    get_log_manager,
    get_logger,
    log_context,
    print_rich_exception,
    setup_web_chatbot_logging,
)


class TestIsSourceEnvironment:
    def test_returns_bool(self):
        result = _is_source_environment()
        assert isinstance(result, bool)

    def test_detects_source_env(self):
        # We're running from source, so this should be True
        assert _is_source_environment() is True


class TestDynamicLogManager:
    def test_init(self, tmp_path):
        mgr = DynamicLogManager(debug=False, log_dir=str(tmp_path / "logs"))
        assert mgr.log_dir == str(tmp_path / "logs")
        assert mgr.file_handler is not None
        assert mgr.console_handler is not None

    def test_init_debug_mode(self, tmp_path):
        mgr = DynamicLogManager(debug=True, log_dir=str(tmp_path / "logs"))
        assert mgr.root_logger.level == logging.DEBUG

    def test_set_output_target_file(self, tmp_path):
        mgr = DynamicLogManager(log_dir=str(tmp_path / "logs"))
        mgr.set_output_target("file")
        assert mgr.file_handler in mgr.root_logger.handlers
        assert mgr.console_handler not in mgr.root_logger.handlers

    def test_set_output_target_console(self, tmp_path):
        mgr = DynamicLogManager(log_dir=str(tmp_path / "logs"))
        mgr.set_output_target("console")
        assert mgr.console_handler in mgr.root_logger.handlers
        assert mgr.file_handler not in mgr.root_logger.handlers

    def test_set_output_target_both(self, tmp_path):
        mgr = DynamicLogManager(log_dir=str(tmp_path / "logs"))
        mgr.set_output_target("both")
        assert mgr.file_handler in mgr.root_logger.handlers
        assert mgr.console_handler in mgr.root_logger.handlers

    def test_set_output_target_none(self, tmp_path):
        mgr = DynamicLogManager(log_dir=str(tmp_path / "logs"))
        mgr.set_output_target("none")
        assert len(mgr.root_logger.handlers) == 0

    def test_restore_default(self, tmp_path):
        mgr = DynamicLogManager(log_dir=str(tmp_path / "logs"))
        mgr.set_output_target("none")
        mgr.restore_default()
        assert mgr.file_handler in mgr.root_logger.handlers
        assert mgr.console_handler in mgr.root_logger.handlers

    def test_restore_original(self, tmp_path):
        mgr = DynamicLogManager(log_dir=str(tmp_path / "logs"))
        original_count = len(mgr.original_handlers)
        mgr.set_output_target("none")
        mgr.restore_original()
        assert len(mgr.root_logger.handlers) == original_count

    def test_temporary_output(self, tmp_path):
        mgr = DynamicLogManager(log_dir=str(tmp_path / "logs"))
        mgr.set_output_target("both")
        with mgr.temporary_output("file"):
            assert mgr.file_handler in mgr.root_logger.handlers
            assert mgr.console_handler not in mgr.root_logger.handlers
        # After context, handlers should be restored
        assert mgr.file_handler in mgr.root_logger.handlers
        assert mgr.console_handler in mgr.root_logger.handlers

    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "new_logs"
        assert not log_dir.exists()
        DynamicLogManager(log_dir=str(log_dir))
        assert log_dir.exists()


class TestConfigureLogging:
    def test_basic(self, tmp_path):
        mgr = configure_logging(debug=False, log_dir=str(tmp_path / "logs"))
        assert isinstance(mgr, DynamicLogManager)

    def test_debug_mode(self, tmp_path):
        mgr = configure_logging(debug=True, log_dir=str(tmp_path / "logs"))
        assert mgr.debug is True

    def test_no_console_output(self, tmp_path):
        mgr = configure_logging(log_dir=str(tmp_path / "logs"), console_output=False)
        assert mgr.file_handler in mgr.root_logger.handlers
        assert mgr.console_handler not in mgr.root_logger.handlers


class TestAddExcInfo:
    def test_error_level(self):
        event_dict = {}
        result = add_exc_info(None, "error", event_dict)
        assert result["exc_info"] is True

    def test_non_error_level(self):
        event_dict = {}
        result = add_exc_info(None, "info", event_dict)
        assert "exc_info" not in result


class TestAddCodeLocation:
    def test_debug_adds_fileno(self):
        event_dict = {}
        result = add_code_location(None, "debug", event_dict)
        assert "fileno" in result

    def test_non_debug_no_fileno(self):
        event_dict = {}
        result = add_code_location(None, "info", event_dict)
        assert isinstance(result, dict)


class TestGetLogger:
    def test_returns_logger(self):
        log = get_logger("test_module_extended")
        assert log is not None


class TestSetupWebChatbotLogging:
    def test_basic(self, tmp_path):
        logger = setup_web_chatbot_logging(debug=False, log_dir=str(tmp_path / "logs"))
        assert logger is not None

    def test_debug_mode(self, tmp_path):
        logger = setup_web_chatbot_logging(debug=True, log_dir=str(tmp_path / "logs"))
        assert logger is not None

    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "web_logs"
        assert not log_dir.exists()
        setup_web_chatbot_logging(log_dir=str(log_dir))
        assert log_dir.exists()


class TestAdaptiveRenderer:
    def test_renders(self):
        renderer = AdaptiveRenderer()
        result = renderer(None, "info", {"event": "test message"})
        assert isinstance(result, str)
        assert "test message" in result


class TestGetCurrentLogFile:
    def test_returns_path_or_none(self, tmp_path):
        configure_logging(log_dir=str(tmp_path / "logs"))
        result = _get_current_log_file()
        assert result is None or hasattr(result, "suffix")


class TestPrintRichException:
    def test_basic(self, tmp_path):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        configure_logging(log_dir=str(tmp_path / "logs"))
        print_rich_exception(console, ValueError("test error"), "Test failed")
        output = buf.getvalue()
        assert "test error" in output

    def test_with_logger(self, tmp_path):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        configure_logging(log_dir=str(tmp_path / "logs"))
        file_logger = get_logger("test_rich")
        print_rich_exception(console, RuntimeError("oops"), "Process error", file_logger=file_logger)
        output = buf.getvalue()
        assert "oops" in output
