# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/models/mcp_utils.py"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.models.mcp_utils import _safe_connect_server, multiple_mcp_servers


# =====================================================================
# multiple_mcp_servers
# =====================================================================


class TestMultipleMcpServers:
    @pytest.mark.asyncio
    async def test_empty_servers(self):
        async with multiple_mcp_servers({}) as servers:
            assert servers == {}

    @pytest.mark.asyncio
    async def test_server_connection_failure(self):
        mock_server = MagicMock()
        mock_server.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
        mock_server.__aexit__ = AsyncMock()

        async with multiple_mcp_servers({"failing": mock_server}) as servers:
            # Failed server should not be in connected_servers
            assert len(servers) == 0

    @pytest.mark.asyncio
    async def test_successful_connection(self):
        mock_server = MagicMock()
        mock_connected = MagicMock()

        # Create an async context manager that yields the connected server
        async def fake_aenter(*args):
            return mock_connected

        async def fake_aexit(*args):
            pass

        mock_server.__aenter__ = fake_aenter
        mock_server.__aexit__ = fake_aexit

        # We need to patch _safe_connect_server to skip the real connection
        with patch("datus.models.mcp_utils._safe_connect_server") as mock_safe:

            class FakeContextManager:
                async def __aenter__(self_inner):
                    return mock_connected

                async def __aexit__(self_inner, *args):
                    pass

            mock_safe.return_value = FakeContextManager()

            async with multiple_mcp_servers({"test": mock_server}) as servers:
                assert "test" in servers
                assert servers["test"] is mock_connected
