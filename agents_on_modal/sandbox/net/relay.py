"""Loopback proxy relay for the Modal sandbox. Run as ``python -m agents_on_modal.sandbox.net.relay``.

Modal sandboxes have full outbound internet and no second container or iptables (gVisor), so the
Docker egress model does not apply. Instead, when an upstream residential proxy is configured, this
process runs two loopback-only listeners that both chain through it:

  * ``127.0.0.1:3128``  HTTP forward proxy (CONNECT + absolute-URI ``http://``).
  * ``127.0.0.1:1080``  SOCKS5 server (no auth; it is loopback-only inside the sandbox).

Exporting ``HTTP_PROXY``/``HTTPS_PROXY``/``ALL_PROXY`` pointed at these (see ``env``) makes the
agent's code and browser egress exit via the residential IP. ``NO_PROXY`` carries the model host so
inference bypasses the relay and stays direct; as a second guard the relay itself refuses to tunnel
to the model host. If no upstream proxy is set, this service is not started.

This file ships into Linux containers; keep it LF + POSIX.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import sys
from urllib.parse import urlsplit

from . import proxy as proxy_module
from .proxy import (
    CONNECT_TIMEOUT_SECONDS,
    RELAY_HOST,
    RELAY_HTTP_PORT,
    RELAY_SOCKS_PORT,
    RequestHead,
    UpstreamProxy,
    close_writer,
    log_line,
    no_proxy_value,
    parse_proxy,
    read_request_head,
    rebuild_request,
    send_simple_response,
    split_authority,
    splice,
)


def env(
    proxy: UpstreamProxy | None, model_host: str, *, ca_cert: str | None = None
) -> dict[str, str]:
    """Env the sandbox exports. With a proxy, ``*_PROXY`` point at the relay and ``NO_PROXY`` keeps
    the model direct; without one, nothing is exported (the relay is not started)."""

    if proxy is None:
        return {}
    return proxy.env(model_host, ca_cert=ca_cert)


class Relay:
    """Chains loopback HTTP and SOCKS5 requests through the upstream residential proxy."""

    def __init__(self, upstream: UpstreamProxy, model_host: str | None = None) -> None:
        self.upstream = upstream
        self.model_host = (model_host or "").strip().strip("[]").lower() or None

    def _is_model_host(self, host: str) -> bool:
        return self.model_host is not None and host.strip("[]").lower() == self.model_host

    async def _dial(self, host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await self.upstream.open_connection(host, port, timeout=CONNECT_TIMEOUT_SECONDS)

    # ---- HTTP forward proxy -----------------------------------------------------------------
    async def handle_http(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await read_request_head(reader)
            if head is None:
                return
            if head.method.upper() == "CONNECT":
                await self._http_connect(head, reader, writer)
            else:
                await self._http_absolute(head, reader, writer)
        except ValueError as error:
            log_line(listener="relay-http", event="reject", reason=str(error))
            await send_simple_response(writer, "400 Bad Request", f"{error}\n")
        except Exception as error:  # pragma: no cover - defensive
            log_line(listener="relay-http", event="error", reason=repr(error))
        finally:
            await close_writer(writer)

    async def _http_connect(
        self, head: RequestHead, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        host, port = split_authority(head.target, 443)
        if self._is_model_host(host):
            log_line(listener="relay-http", event="deny", method="CONNECT", host=host, port=port, reason="model host must stay direct")
            await send_simple_response(writer, "403 Forbidden", "model host must stay direct\n")
            return
        try:
            upstream_reader, upstream_writer = await self._dial(host, port)
        except (OSError, asyncio.TimeoutError, proxy_module.ProxyError) as error:
            log_line(listener="relay-http", event="unreachable", method="CONNECT", host=host, port=port, reason=str(error))
            await send_simple_response(writer, "502 Bad Gateway", f"{error}\n")
            return
        log_line(listener="relay-http", event="allow", method="CONNECT", host=host, port=port, route="upstream")
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        try:
            await splice((reader, writer), (upstream_reader, upstream_writer))
        finally:
            await close_writer(upstream_writer)

    async def _http_absolute(
        self, head: RequestHead, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if not head.target.lower().startswith("http://"):
            raise ValueError(f"only CONNECT and absolute http:// targets are proxied; got {head.target!r}")
        parts = urlsplit(head.target)
        host, port = split_authority(parts.netloc, 80)
        if self._is_model_host(host):
            log_line(listener="relay-http", event="deny", method=head.method, host=host, port=port, reason="model host must stay direct")
            await send_simple_response(writer, "403 Forbidden", "model host must stay direct\n")
            return
        try:
            upstream_reader, upstream_writer = await self._dial(host, port)
        except (OSError, asyncio.TimeoutError, proxy_module.ProxyError) as error:
            log_line(listener="relay-http", event="unreachable", method=head.method, host=host, port=port, reason=str(error))
            await send_simple_response(writer, "502 Bad Gateway", f"{error}\n")
            return
        origin_target = parts.path or "/"
        if parts.query:
            origin_target += f"?{parts.query}"
        log_line(listener="relay-http", event="allow", method=head.method, host=host, port=port, route="upstream", path=parts.path)
        upstream_writer.write(rebuild_request(head, target=origin_target, host_header=parts.netloc))
        await upstream_writer.drain()
        try:
            await splice((reader, writer), (upstream_reader, upstream_writer))
        finally:
            await close_writer(upstream_writer)

    # ---- SOCKS5 server ----------------------------------------------------------------------
    async def handle_socks(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            # Greeting: accept whatever methods the client offers, choose no-auth (loopback only).
            head = await reader.readexactly(2)
            if head[0] != 0x05:
                return
            await reader.readexactly(head[1])  # method list
            writer.write(b"\x05\x00")
            await writer.drain()
            # Request: VER CMD RSV ATYP ...
            request = await reader.readexactly(4)
            if request[1] != 0x01:  # only CONNECT
                writer.write(b"\x05\x07\x00\x01\x00\x00\x00\x00\x00\x00")
                await writer.drain()
                return
            host = await self._read_socks_host(reader, request[3])
            if host is None:
                writer.write(b"\x05\x08\x00\x01\x00\x00\x00\x00\x00\x00")  # address type not supported
                await writer.drain()
                return
            port = int.from_bytes(await reader.readexactly(2), "big")
            if self._is_model_host(host):
                log_line(listener="relay-socks", event="deny", host=host, port=port, reason="model host must stay direct")
                writer.write(b"\x05\x02\x00\x01\x00\x00\x00\x00\x00\x00")  # connection not allowed
                await writer.drain()
                return
            try:
                upstream_reader, upstream_writer = await self._dial(host, port)
            except (OSError, asyncio.TimeoutError, proxy_module.ProxyError) as error:
                log_line(listener="relay-socks", event="unreachable", host=host, port=port, reason=str(error))
                writer.write(b"\x05\x01\x00\x01\x00\x00\x00\x00\x00\x00")  # general failure
                await writer.drain()
                return
            log_line(listener="relay-socks", event="allow", host=host, port=port, route="upstream")
            writer.write(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")  # success, bound 0.0.0.0:0
            await writer.drain()
            await splice((reader, writer), (upstream_reader, upstream_writer))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except Exception as error:  # pragma: no cover - defensive
            log_line(listener="relay-socks", event="error", reason=repr(error))
        finally:
            if upstream_writer is not None:
                await close_writer(upstream_writer)
            await close_writer(writer)

    @staticmethod
    async def _read_socks_host(reader: asyncio.StreamReader, atyp: int) -> str | None:
        if atyp == 0x01:  # IPv4
            return str(ipaddress.ip_address(await reader.readexactly(4)))
        if atyp == 0x04:  # IPv6
            return str(ipaddress.ip_address(await reader.readexactly(16)))
        if atyp == 0x03:  # domain name
            length = (await reader.readexactly(1))[0]
            raw = await reader.readexactly(length)
            try:
                return raw.decode("idna") or None
            except (UnicodeError, ValueError):
                return raw.decode("latin-1") or None
        return None


async def main() -> int:
    upstream = parse_proxy(
        os.environ.get("AOM_UPSTREAM_PROXY") or None,
        session=os.environ.get("AOM_PROXY_SESSION") or None,
    )
    if upstream is None:
        log_line(listener="relay", event="disabled", reason="AOM_UPSTREAM_PROXY is not set")
        return 0
    model_host = (os.environ.get("AOM_MODEL_HOST") or "").strip() or None
    relay = Relay(upstream, model_host)
    http_port = int(os.environ.get("AOM_RELAY_HTTP_PORT", str(RELAY_HTTP_PORT)))
    socks_port = int(os.environ.get("AOM_RELAY_SOCKS_PORT", str(RELAY_SOCKS_PORT)))
    http_server = await asyncio.start_server(relay.handle_http, RELAY_HOST, http_port)
    socks_server = await asyncio.start_server(relay.handle_socks, RELAY_HOST, socks_port)
    log_line(
        listener="relay",
        event="listening",
        http=f"{RELAY_HOST}:{http_port}",
        socks=f"{RELAY_HOST}:{socks_port}",
        upstream=upstream.url,
        model_host=model_host,
        no_proxy=no_proxy_value(model_host),
    )
    async with http_server, socks_server:
        await asyncio.gather(http_server.serve_forever(), socks_server.serve_forever())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
