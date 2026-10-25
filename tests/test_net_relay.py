"""Modal loopback relay: SOCKS5 and HTTP forward proxies chain through the upstream residential
proxy, the model host is refused (stays direct), and env() carries NO_PROXY. All in-process."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from agents_on_modal.sandbox.net import parse_proxy, relay as relay_module
from agents_on_modal.sandbox.net.proxy import ProxyError, UpstreamProxy, close_writer
from agents_on_modal.sandbox.net.relay import Relay
from net_helpers import port_of, start_echo_server, start_socks5_server


@contextlib.asynccontextmanager
async def serving(*servers):
    try:
        yield
    finally:
        for server in servers:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()


async def _round_trip(client: UpstreamProxy, host: str, port: int) -> bytes:
    reader, writer = await client.open_connection(host, port, timeout=5)
    try:
        writer.write(b"relayed!")
        await writer.drain()
        return await asyncio.wait_for(reader.readexactly(8), timeout=5)
    finally:
        await close_writer(writer)


def test_env_none_returns_empty():
    assert relay_module.env(None, "model.example.com") == {}


def test_env_sets_no_proxy_for_model_host():
    proxy = parse_proxy("socks5h://u:p@gw:1080", session="sess")
    env = relay_module.env(proxy, "model.example.com")
    assert "model.example.com" in env["NO_PROXY"].split(",")
    assert env["ALL_PROXY"] == "socks5://127.0.0.1:1080"


@pytest.mark.asyncio
async def test_relay_socks_chains_through_upstream():
    echo = await start_echo_server()
    upstream_socks = await start_socks5_server()
    relay = Relay(UpstreamProxy("socks5", "127.0.0.1", port_of(upstream_socks)))
    relay_server = await asyncio.start_server(relay.handle_socks, "127.0.0.1", 0)
    async with serving(echo, upstream_socks, relay_server):
        # Speak SOCKS5 to the relay; the relay chains through the fake upstream to the echo server.
        client = UpstreamProxy("socks5", "127.0.0.1", port_of(relay_server))
        assert await _round_trip(client, "127.0.0.1", port_of(echo)) == b"relayed!"


@pytest.mark.asyncio
async def test_relay_http_connect_chains_through_upstream():
    echo = await start_echo_server()
    upstream_socks = await start_socks5_server()
    relay = Relay(UpstreamProxy("socks5", "127.0.0.1", port_of(upstream_socks)))
    relay_server = await asyncio.start_server(relay.handle_http, "127.0.0.1", 0)
    async with serving(echo, upstream_socks, relay_server):
        client = UpstreamProxy("http", "127.0.0.1", port_of(relay_server))
        assert await _round_trip(client, "127.0.0.1", port_of(echo)) == b"relayed!"


@pytest.mark.asyncio
async def test_relay_socks_refuses_model_host():
    upstream_socks = await start_socks5_server()
    relay = Relay(UpstreamProxy("socks5", "127.0.0.1", port_of(upstream_socks)), model_host="model.example.com")
    relay_server = await asyncio.start_server(relay.handle_socks, "127.0.0.1", 0)
    async with serving(upstream_socks, relay_server):
        # socks5h sends the name to the relay (no local DNS), so the relay sees the model host.
        client = UpstreamProxy("socks5h", "127.0.0.1", port_of(relay_server))
        with pytest.raises(ProxyError):
            await client.open_connection("model.example.com", 443, timeout=5)


@pytest.mark.asyncio
async def test_relay_http_refuses_model_host():
    upstream_socks = await start_socks5_server()
    relay = Relay(UpstreamProxy("socks5", "127.0.0.1", port_of(upstream_socks)), model_host="model.example.com")
    relay_server = await asyncio.start_server(relay.handle_http, "127.0.0.1", 0)
    async with serving(upstream_socks, relay_server):
        reader, writer = await asyncio.open_connection("127.0.0.1", port_of(relay_server))
        writer.write(b"CONNECT model.example.com:443 HTTP/1.1\r\nHost: model.example.com:443\r\n\r\n")
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), timeout=5)
        assert b"403" in status
        await close_writer(writer)
