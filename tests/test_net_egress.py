"""Docker egress broker: the model reverse proxy injects the credential, and the forward proxy
enforces the destination policy and chains through an upstream when configured. All in-process."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from agents_on_modal.sandbox.net.egress import ForwardProxy, ModelProxy, _authorization_header
from agents_on_modal.sandbox.net.policy import DEFAULT_PORTS, Policy
from agents_on_modal.sandbox.net.proxy import UpstreamProxy, close_writer
from net_helpers import (
    port_of,
    start_echo_server,
    start_recording_http_server,
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


async def _open_listener(handler):
    return await asyncio.start_server(handler, "127.0.0.1", 0)


class _AllowAll:
    """Policy stand-in for the chaining test: returns the target unchanged (loopback allowed)."""

    async def resolve(self, host, port):
        return host, port


def test_authorization_header_helper():
    assert _authorization_header("abc.def") == "Bearer abc.def"
    assert _authorization_header("Bearer xyz") == "Bearer xyz"
    assert _authorization_header("") is None


@pytest.mark.asyncio
async def test_model_proxy_injects_auth_header():
    upstream, seen = await start_recording_http_server()
    model = ModelProxy(f"http://127.0.0.1:{port_of(upstream)}", "Bearer testtoken", "/v1")
    server = await _open_listener(model.handle)
    async with serving(upstream, server):
        reader, writer = await asyncio.open_connection("127.0.0.1", port_of(server))
        writer.write(b"GET /v1/models HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer SANDBOX_FAKE\r\n\r\n")
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), timeout=5)
        assert b"200" in status
        await close_writer(writer)
    assert len(seen) == 1
    # The sandbox's own Authorization is stripped and replaced with the injected credential.
    assert seen[0]["headers"]["authorization"] == "Bearer testtoken"
    # local_prefix "/v1" is stripped before forwarding.
    assert seen[0]["target"] == "/models"


@pytest.mark.asyncio
async def test_model_proxy_without_token_sends_no_auth():
    upstream, seen = await start_recording_http_server()
    model = ModelProxy(f"http://127.0.0.1:{port_of(upstream)}", None, "/v1")
    server = await _open_listener(model.handle)
    async with serving(upstream, server):
        reader, writer = await asyncio.open_connection("127.0.0.1", port_of(server))
        writer.write(b"GET /v1/models HTTP/1.1\r\nHost: gateway\r\n\r\n")
        await writer.drain()
        await asyncio.wait_for(reader.readline(), timeout=5)
        await close_writer(writer)
    assert "authorization" not in seen[0]["headers"]


@pytest.mark.asyncio
async def test_forward_refuses_private_destination():
    forward = ForwardProxy(Policy(DEFAULT_PORTS))
    server = await _open_listener(forward.handle)
    async with serving(server):
        reader, writer = await asyncio.open_connection("127.0.0.1", port_of(server))
        writer.write(b"CONNECT 127.0.0.1:443 HTTP/1.1\r\nHost: 127.0.0.1:443\r\n\r\n")
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), timeout=5)
        assert b"403" in status
        await close_writer(writer)


@pytest.mark.asyncio
async def test_forward_refuses_disallowed_port():
    forward = ForwardProxy(Policy(DEFAULT_PORTS))
    server = await _open_listener(forward.handle)
    async with serving(server):
        reader, writer = await asyncio.open_connection("127.0.0.1", port_of(server))
        writer.write(b"CONNECT 8.8.8.8:22 HTTP/1.1\r\nHost: 8.8.8.8:22\r\n\r\n")
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), timeout=5)
        assert b"403" in status
        await close_writer(writer)


@pytest.mark.asyncio
async def test_forward_chains_through_upstream_proxy():
    echo = await start_echo_server()
    socks = await start_socks5_server()
    upstream = UpstreamProxy("socks5", "127.0.0.1", port_of(socks))
    # _AllowAll lets the loopback echo target pass; the point is the upstream chaining.
    forward = ForwardProxy(_AllowAll(), upstream)
    server = await _open_listener(forward.handle)
    async with serving(echo, socks, server):
        reader, writer = await asyncio.open_connection("127.0.0.1", port_of(server))
        target = f"127.0.0.1:{port_of(echo)}"
        writer.write(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        assert b"200" in head
        writer.write(b"tunneled")
        await writer.drain()
        assert await asyncio.wait_for(reader.readexactly(8), timeout=5) == b"tunneled"
        await close_writer(writer)
