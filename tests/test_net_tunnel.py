"""Round-trip the SOCKS5 and HTTP-CONNECT clients through tiny in-process proxy servers."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from agents_on_modal.sandbox.net.proxy import ProxyError, UpstreamProxy, close_writer
from net_helpers import (
    port_of,
    start_echo_server,
    start_http_connect_server,
    start_socks5_server,
)


@contextlib.asynccontextmanager
async def serving(*servers):
    try:
        yield
    finally:
        for server in servers:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()


async def _round_trip(proxy: UpstreamProxy, host: str, port: int) -> bytes:
    reader, writer = await proxy.open_connection(host, port, timeout=5)
    try:
        writer.write(b"ping-123")
        await writer.drain()
        return await asyncio.wait_for(reader.readexactly(8), timeout=5)
    finally:
        await close_writer(writer)


@pytest.mark.asyncio
async def test_socks5_ip_literal_round_trip():
    echo = await start_echo_server()
    socks = await start_socks5_server()
    async with serving(echo, socks):
        proxy = UpstreamProxy("socks5", "127.0.0.1", port_of(socks))
        assert await _round_trip(proxy, "127.0.0.1", port_of(echo)) == b"ping-123"


@pytest.mark.asyncio
async def test_socks5h_domain_remote_dns_round_trip():
    echo = await start_echo_server()
    socks = await start_socks5_server()
    async with serving(echo, socks):
        # socks5h sends the name; the fake server resolves "localhost" itself (remote DNS).
        proxy = UpstreamProxy("socks5h", "127.0.0.1", port_of(socks))
        assert await _round_trip(proxy, "localhost", port_of(echo)) == b"ping-123"


@pytest.mark.asyncio
async def test_socks5_userpass_auth_round_trip():
    echo = await start_echo_server()
    socks = await start_socks5_server(username="user", password="pass")
    async with serving(echo, socks):
        proxy = UpstreamProxy("socks5", "127.0.0.1", port_of(socks), "user", "pass")
        assert await _round_trip(proxy, "127.0.0.1", port_of(echo)) == b"ping-123"


@pytest.mark.asyncio
async def test_socks5_bad_credentials_raise():
    echo = await start_echo_server()
    socks = await start_socks5_server(username="user", password="right")
    async with serving(echo, socks):
        proxy = UpstreamProxy("socks5", "127.0.0.1", port_of(socks), "user", "wrong")
        with pytest.raises(ProxyError):
            await proxy.open_connection("127.0.0.1", port_of(echo), timeout=5)


@pytest.mark.asyncio
async def test_http_connect_round_trip():
    echo = await start_echo_server()
    http = await start_http_connect_server()
    async with serving(echo, http):
        proxy = UpstreamProxy("http", "127.0.0.1", port_of(http))
        assert await _round_trip(proxy, "127.0.0.1", port_of(echo)) == b"ping-123"


@pytest.mark.asyncio
async def test_http_connect_auth_round_trip():
    echo = await start_echo_server()
    http = await start_http_connect_server(username="u", password="p")
    async with serving(echo, http):
        proxy = UpstreamProxy("http", "127.0.0.1", port_of(http), "u", "p")
        assert await _round_trip(proxy, "127.0.0.1", port_of(echo)) == b"ping-123"


@pytest.mark.asyncio
async def test_http_connect_bad_credentials_raise():
    echo = await start_echo_server()
    http = await start_http_connect_server(username="u", password="right")
    async with serving(echo, http):
        proxy = UpstreamProxy("http", "127.0.0.1", port_of(http), "u", "wrong")
        with pytest.raises(ProxyError):
            await proxy.open_connection("127.0.0.1", port_of(echo), timeout=5)
