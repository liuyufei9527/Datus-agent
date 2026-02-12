# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/device_utils.py."""

import platform
from unittest.mock import patch

import pytest

import datus.utils.device_utils as device_mod
from datus.utils.device_utils import get_device


class TestGetDevice:
    def setup_method(self):
        """Reset the cached device before each test."""
        device_mod._DEVICE = None

    def teardown_method(self):
        """Reset the cached device after each test."""
        device_mod._DEVICE = None

    def test_returns_cached_value(self):
        device_mod._DEVICE = "test_device"
        assert get_device() == "test_device"

    def test_darwin_arm64_returns_mps(self):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            result = get_device()
            assert result == "mps"

    def test_darwin_x86_returns_cpu(self):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="x86_64"):
            result = get_device()
            assert result == "cpu"

    def test_linux_no_gpu_returns_cpu(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            result = get_device()
            assert result == "cpu"

    def test_caching_works(self):
        device_mod._DEVICE = None
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            first = get_device()
            assert first == "mps"
        # Second call should return cached value without calling platform
        second = get_device()
        assert second == "mps"

    def test_linux_cuda_found(self):
        mock_result = type("Result", (), {"returncode": 0})()
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = get_device()
            assert result == "cuda"

    def test_linux_nvidia_error(self):
        def side_effect(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                raise Exception("nvidia error")
            raise FileNotFoundError

        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", side_effect=side_effect),
        ):
            result = get_device()
            assert result == "cpu"

    def test_linux_rocm_found(self):
        call_count = [0]
        mock_result_fail = type("Result", (), {"returncode": 1})()
        mock_result_ok = type("Result", (), {"returncode": 0})()

        def side_effect(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                raise FileNotFoundError
            if cmd[0] == "rocm-smi":
                return mock_result_ok
            return mock_result_fail

        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", side_effect=side_effect),
        ):
            result = get_device()
            assert result == "rocm"

    def test_windows_cuda_found(self):
        mock_result = type("Result", (), {"returncode": 0, "stdout": "NVIDIA GPU"})()
        with (
            patch("platform.system", return_value="Windows"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = get_device()
            assert result == "cuda"

    def test_windows_amd_found(self):
        mock_nvidia_fail = type("Result", (), {"returncode": 1, "stdout": ""})()
        mock_wmic_result = type("Result", (), {"returncode": 0, "stdout": "AMD Radeon RX 6800"})()

        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            if "nvidia-smi" in cmd:
                return mock_nvidia_fail
            return mock_wmic_result

        with (
            patch("platform.system", return_value="Windows"),
            patch("subprocess.run", side_effect=side_effect),
        ):
            result = get_device()
            assert result == "rocm"

    def test_windows_no_gpu(self):
        mock_nvidia_fail = type("Result", (), {"returncode": 1, "stdout": ""})()
        mock_wmic_no_amd = type("Result", (), {"returncode": 0, "stdout": "Intel UHD Graphics"})()

        def side_effect(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                return mock_nvidia_fail
            return mock_wmic_no_amd

        with (
            patch("platform.system", return_value="Windows"),
            patch("subprocess.run", side_effect=side_effect),
        ):
            result = get_device()
            assert result == "cpu"

    def test_windows_nvidia_exception(self):
        def side_effect(cmd, **kwargs):
            raise Exception("command failed")

        with (
            patch("platform.system", return_value="Windows"),
            patch("subprocess.run", side_effect=side_effect),
        ):
            result = get_device()
            assert result == "cpu"
