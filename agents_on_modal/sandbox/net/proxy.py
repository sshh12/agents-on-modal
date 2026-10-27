"""Upstream residential proxy client (SOCKS5 / HTTP CONNECT) + shared stream plumbing.

``parse_proxy`` turns a proxy URL into an :class:`UpstreamProxy`.
``UpstreamProxy.open_connection`` opens a TCP tunnel to a target ``host:port`` *through* the
upstream proxy, using a minimal stdlib-asyncio SOCKS5 or HTTP-CONNECT client (no new deps). The
Docker egress broker and the Modal loopback relay both chain their validated outbound connections
through it; model inference never does.

This module also holds the small HTTP/stream helpers shared by ``egress.py`` and ``relay.py`` so
those two runnable services stay thin.

This file ships into Linux containers; keep it LF + POSIX.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import re
import socket
import sys
import time
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from ...config import AomError

# Where the Modal relay listens (loopback only). The sandbox exports these via ``*_PROXY`` env.
RELAY_HOST = "127.0.0.1"
RELAY_HTTP_PORT = 3128
RELAY_SOCKS_PORT = 1080

# Generous limits for a long-lived agent streaming model output through the proxy.
MAX_HEAD_BYTES = 64 * 1024
IDLE_TIMEOUT_SECONDS = 900.0
CONNECT_TIMEOUT_SECONDS = 30.0

_SOCKS_SCHEMES = frozenset({"socks5", "socks5h"})
_DEFAULT_PORTS = {"socks5": 1080, "socks5h": 1080, "http": 8080, "https": 8080}


class ProxyError(AomError):
    """A proxy URL could not be parsed, or an upstream tunnel could not be established."""


# ---------------------------------------------------------------------------------------------
# Upstream proxy
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class UpstreamProxy:
    """A parsed upstream residential proxy. ``username`` already has ``{session}`` substituted."""

    scheme: str  # socks5 | socks5h | http
    host: str
    port: int
    username: str | None = None
    password: str | None = None

    @property
    def url(self) -> str:
        auth = ""
        if self.username is not None:
            auth = self.username + (f":{self.password}" if self.password is not None else "") + "@"
        return f"{self.scheme}://{auth}{self.host}:{self.port}"

    # ---- env the sandbox exports so code+browser egress exits via the relay -----------------
    def env(self, model_host: str, *, ca_cert: str | None = None) -> dict[str, str]:
        """``*_PROXY`` pointing at the loopback relay, plus ``NO_PROXY`` keeping the model direct.

        Setting these in the sandbox routes code and browser traffic through the relay (and thus
        the residential IP), while requests to ``model_host`` bypass the proxy and stay direct.
        """

        http = f"http://{RELAY_HOST}:{RELAY_HTTP_PORT}"
        socks = f"socks5://{RELAY_HOST}:{RELAY_SOCKS_PORT}"
        no = no_proxy_value(model_host)
        env = {
            "HTTP_PROXY": http,
            "http_proxy": http,
            "HTTPS_PROXY": http,
            "https_proxy": http,
            "ALL_PROXY": socks,
            "all_proxy": socks,
            "NO_PROXY": no,
            "no_proxy": no,
        }
        if ca_cert:
            # Only relevant when something upstream terminates TLS (e.g. a MITM CA). The relay
            # itself does plain CONNECT tunnelling, so this is normally unset.
            for key in ("NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
                env[key] = ca_cert
        return env

    # ---- tunnelling -------------------------------------------------------------------------
    async def open_connection(
        self, host: str, port: int, *, timeout: float = CONNECT_TIMEOUT_SECONDS
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a TCP tunnel to ``host:port`` through this proxy. ``host`` may be an IP or name."""

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), timeout=timeout
        )
        try:
            if self.scheme == "http":
                await asyncio.wait_for(self._http_connect(reader, writer, host, port), timeout=timeout)
            else:
                await asyncio.wait_for(self._socks5_connect(reader, writer, host, port), timeout=timeout)
        except BaseException:
            await close_writer(writer)
            raise
        return reader, writer

    async def _http_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, host: str, port: int
    ) -> None:
        authority = f"{host}:{port}"
        lines = [f"CONNECT {authority} HTTP/1.1", f"Host: {authority}"]
        if self.username is not None:
            raw = f"{self.username}:{self.password or ''}".encode("utf-8")
            lines.append("Proxy-Authorization: Basic " + base64.b64encode(raw).decode("ascii"))
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        await writer.drain()
        blob = await reader.readuntil(b"\r\n\r\n")
        status_line = blob.split(b"\r\n", 1)[0].decode("latin-1")
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or parts[1] != "200":
            raise ProxyError(f"upstream HTTP proxy refused CONNECT {authority}: {status_line!r}")

    async def _socks5_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, host: str, port: int
    ) -> None:
        # Greeting: offer no-auth, and user/pass when we have credentials.
        methods = b"\x00\x02" if self.username is not None else b"\x00"
        writer.write(bytes([0x05, len(methods)]) + methods)
        await writer.drain()
        head = await reader.readexactly(2)
        if head[0] != 0x05:
            raise ProxyError("upstream SOCKS5 proxy spoke the wrong version")
        method = head[1]
        if method == 0xFF:
            raise ProxyError("upstream SOCKS5 proxy rejected all offered auth methods")
        if method == 0x02:
            await self._socks5_userpass(reader, writer)
        elif method != 0x00:
            raise ProxyError(f"upstream SOCKS5 proxy chose unsupported auth method {method}")
        writer.write(b"\x05\x01\x00" + await self._socks5_addr(host) + port.to_bytes(2, "big"))
        await writer.drain()
        reply = await reader.readexactly(4)
        if reply[1] != 0x00:
            raise ProxyError(f"upstream SOCKS5 CONNECT failed (reply code {reply[1]})")
        # Consume the bound address so the stream is positioned at the tunnel body.
        atyp = reply[3]
        if atyp == 0x01:
            await reader.readexactly(4)
        elif atyp == 0x04:
            await reader.readexactly(16)
        elif atyp == 0x03:
            length = (await reader.readexactly(1))[0]
            await reader.readexactly(length)
        else:
            raise ProxyError(f"upstream SOCKS5 returned unknown address type {atyp}")
        await reader.readexactly(2)

    async def _socks5_userpass(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        user = (self.username or "").encode("utf-8")
        password = (self.password or "").encode("utf-8")
        if len(user) > 255 or len(password) > 255:
            raise ProxyError("SOCKS5 username/password exceed 255 bytes")
        writer.write(bytes([0x01, len(user)]) + user + bytes([len(password)]) + password)
        await writer.drain()
        status = await reader.readexactly(2)
        if status[1] != 0x00:
            raise ProxyError("upstream SOCKS5 username/password authentication failed")

    async def _socks5_addr(self, host: str) -> bytes:
        """Encode the SOCKS5 destination. ``socks5h`` sends the name (remote DNS); ``socks5``
        resolves locally so the proxy connects to the exact address we validated."""

        literal = host.strip("[]")
        try:
            ip = ipaddress.ip_address(literal)
        except ValueError:
            ip = None
        if ip is None and self.scheme == "socks5":
            loop = asyncio.get_running_loop()
            infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            if not infos:
                raise ProxyError(f"could not resolve {host!r} for the SOCKS5 proxy")
            ip = ipaddress.ip_address(str(infos[0][4][0]))
        if ip is not None:
            return (b"\x01" if ip.version == 4 else b"\x04") + ip.packed
        try:
            name = host.encode("idna")
        except (UnicodeError, ValueError):
            name = host.encode("ascii", "ignore")
        if len(name) > 255:
            raise ProxyError(f"hostname {host!r} is too long for SOCKS5")
        return b"\x03" + bytes([len(name)]) + name


def parse_proxy(url: str | None, *, session: str | None = None) -> UpstreamProxy | None:
    """Parse a proxy URL into an :class:`UpstreamProxy`, or ``None`` when no proxy is configured.

    Accepts ``socks5://``, ``socks5h://`` and ``http://`` with an optional ``user:pass@``. A
    ``{session}`` token in the username is replaced with ``session`` for sticky residential
    sessions. Raises :class:`ProxyError` on a malformed or unsupported URL.
    """

    if url is None:
        return None
    url = url.strip()
    if not url:
        return None
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _SOCKS_SCHEMES and scheme not in {"http", "https"}:
        raise ProxyError(f"Unsupported proxy scheme {parts.scheme!r} in {url!r}.")
    if not parts.hostname:
        raise ProxyError(f"Proxy URL {url!r} is missing a host.")
    try:
        port = parts.port or _DEFAULT_PORTS[scheme]
    except ValueError as error:  # non-numeric port
        raise ProxyError(f"Proxy URL {url!r} has an invalid port.") from error
    # ``https`` and ``socks5h`` both behave as a CONNECT/remote-DNS proxy respectively; normalise
    # ``https`` to ``http`` since we tunnel plaintext CONNECT either way.
    normalised = "http" if scheme == "https" else scheme

    username = unquote(parts.username) if parts.username is not None else None
    password = unquote(parts.password) if parts.password is not None else None
    if session is not None:
        # Sticky-session ids for residential providers (e.g. decodo) must be alphanumeric; sandbox
        # ids contain dashes, so strip to [A-Za-z0-9] (and cap length) before substituting so each
        # sandbox still maps to a stable, provider-valid session (and thus a distinct sticky IP).
        safe = re.sub(r"[^A-Za-z0-9]", "", session)[:48] or "s"
        if username is not None:
            username = username.replace("{session}", safe)
        if password is not None:
            password = password.replace("{session}", safe)
    return UpstreamProxy(normalised, parts.hostname, port, username, password)


def no_proxy_value(model_host: str | None) -> str:
    """Comma-separated NO_PROXY: the model host first (inference stays direct), then loopback."""

    hosts = ["localhost", "127.0.0.1", "::1"]
    if model_host:
        host = model_host.strip().strip("[]")
        if host and host not in hosts:
            hosts.insert(0, host)
    return ",".join(hosts)


# ---------------------------------------------------------------------------------------------
# Shared HTTP / stream plumbing (used by egress.py and relay.py)
# ---------------------------------------------------------------------------------------------
def log_line(**fields: object) -> None:
    """Write one JSONL decision record to stdout."""

    fields["time"] = round(time.time(), 3)
    sys.stdout.write(json.dumps(fields, sort_keys=True) + "\n")
    sys.stdout.flush()


@dataclass
class RequestHead:
    method: str
    target: str
    version: str
    headers: dict[str, str]
    raw_headers: list[tuple[str, str]]


async def read_request_head(reader: asyncio.StreamReader) -> RequestHead | None:
    """Read and parse an HTTP request head, or ``None`` on a clean/idle disconnect."""

    try:
        blob = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=IDLE_TIMEOUT_SECONDS)
    except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
        return None
    except asyncio.LimitOverrunError as error:
        raise ValueError("request head exceeded the proxy buffer") from error
    if len(blob) > MAX_HEAD_BYTES:
        raise ValueError("request head is too large")
    lines = blob.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        raise ValueError(f"malformed request line {lines[0]!r}")
    method, target, version = parts
    headers: dict[str, str] = {}
    raw_headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"malformed header line {line!r}")
        name, value = name.strip(), value.strip()
        raw_headers.append((name, value))
        headers[name.lower()] = value
    return RequestHead(method, target, version, headers, raw_headers)


# Hop-by-hop headers, and anything that could let the sandbox influence the injected credential.
STRIPPED_REQUEST_HEADERS = frozenset(
    {
        "proxy-authorization",
        "proxy-connection",
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "authorization",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
    }
)


def split_authority(authority: str, default_port: int) -> tuple[str, int]:
    """Split ``host:port`` / ``[v6]:port`` into ``(host, port)`` with a default port."""

    authority = authority.strip()
    if authority.startswith("["):
        host, _, remainder = authority.partition("]")
        host = host[1:]
        if remainder.startswith(":") and remainder[1:].isdigit():
            return host, int(remainder[1:])
        return host, default_port
    host, separator, port_text = authority.rpartition(":")
    if separator and port_text.isdigit():
        return host, int(port_text)
    return authority, default_port


def rebuild_request(
    head: RequestHead,
    *,
    target: str,
    host_header: str,
    extra_headers: tuple[tuple[str, str], ...] = (),
    drop: frozenset[str] = STRIPPED_REQUEST_HEADERS,
) -> bytes:
    """Re-serialize a request head with a new target/Host, dropping hop-by-hop headers.

    One request per upstream connection (``Connection: close``) keeps header injection correct
    and streaming responses terminating cleanly.
    """

    lines = [f"{head.method} {target} HTTP/1.1"]
    seen_host = False
    for name, value in head.raw_headers:
        lowered = name.lower()
        if lowered in drop:
            continue
        if lowered == "host":
            if seen_host:
                continue
            seen_host = True
            value = host_header
        lines.append(f"{name}: {value}")
    if not seen_host:
        lines.insert(1, f"Host: {host_header}")
    for name, value in extra_headers:
        lines.append(f"{name}: {value}")
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


async def send_simple_response(writer: asyncio.StreamWriter, status: str, message: str) -> None:
    body = message.encode("utf-8")
    head = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin-1")
    try:
        writer.write(head + body)
        await writer.drain()
    except (OSError, ConnectionError):
        pass


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> int:
    total = 0
    try:
        while True:
            chunk = await asyncio.wait_for(reader.read(65536), timeout=IDLE_TIMEOUT_SECONDS)
            if not chunk:
                break
            total += len(chunk)
            writer.write(chunk)
            await writer.drain()
    except (asyncio.TimeoutError, ConnectionError, OSError):
        pass
    finally:
        try:
            writer.write_eof()
        except (OSError, RuntimeError):
            pass
    return total


async def splice(
    client: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    upstream: tuple[asyncio.StreamReader, asyncio.StreamWriter],
) -> None:
    await asyncio.gather(
        pipe(client[0], upstream[1]),
        pipe(upstream[0], client[1]),
        return_exceptions=True,
    )


async def close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, ConnectionError, RuntimeError):
        pass
