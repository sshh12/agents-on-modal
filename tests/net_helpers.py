"""In-process fake servers for the sandbox-net tests. No external network: everything binds
127.0.0.1 and dials only loopback targets the test sets up."""

from __future__ import annotations

import asyncio
import ipaddress

from agents_on_modal.sandbox.net.proxy import splice


def port_of(server: asyncio.AbstractServer) -> int:
    return server.sockets[0].getsockname()[1]


async def start_echo_server() -> asyncio.AbstractServer:
    """A TCP server that echoes back whatever it receives."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def start_socks5_server(
    username: str | None = None, password: str | None = None
) -> asyncio.AbstractServer:
    """A minimal SOCKS5 CONNECT proxy that dials the requested target and splices."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readexactly(2)
            methods = await reader.readexactly(head[1])
            if username is not None:
                if 0x02 not in methods:
                    writer.write(b"\x05\xff")
                    await writer.drain()
                    return
                writer.write(b"\x05\x02")
                await writer.drain()
                await reader.readexactly(1)  # auth version
                ulen = (await reader.readexactly(1))[0]
                user = await reader.readexactly(ulen)
                plen = (await reader.readexactly(1))[0]
                pw = await reader.readexactly(plen)
                if user.decode() != username or pw.decode() != (password or ""):
                    writer.write(b"\x01\x01")
                    await writer.drain()
                    return
                writer.write(b"\x01\x00")
                await writer.drain()
            else:
                writer.write(b"\x05\x00")
                await writer.drain()
            request = await reader.readexactly(4)
            atyp = request[3]
            if atyp == 0x01:
                host = str(ipaddress.ip_address(await reader.readexactly(4)))
            elif atyp == 0x04:
                host = str(ipaddress.ip_address(await reader.readexactly(16)))
            elif atyp == 0x03:
                length = (await reader.readexactly(1))[0]
                host = (await reader.readexactly(length)).decode()
            else:
                writer.write(b"\x05\x08\x00\x01\x00\x00\x00\x00\x00\x00")
                await writer.drain()
                return
            port = int.from_bytes(await reader.readexactly(2), "big")
            try:
                target_reader, target_writer = await asyncio.open_connection(host, port)
            except OSError:
                writer.write(b"\x05\x01\x00\x01\x00\x00\x00\x00\x00\x00")
                await writer.drain()
                return
            writer.write(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            await writer.drain()
            await splice((reader, writer), (target_reader, target_writer))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def start_http_connect_server(
    username: str | None = None, password: str | None = None
) -> asyncio.AbstractServer:
    """A minimal HTTP CONNECT proxy that dials the requested target and splices."""

    import base64

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            blob = await reader.readuntil(b"\r\n\r\n")
            lines = blob.decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    name, _, value = line.partition(":")
                    headers[name.strip().lower()] = value.strip()
            if method.upper() != "CONNECT":
                writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                await writer.drain()
                return
            if username is not None:
                expected = "Basic " + base64.b64encode(
                    f"{username}:{password or ''}".encode()
                ).decode()
                if headers.get("proxy-authorization") != expected:
                    writer.write(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n")
                    await writer.drain()
                    return
            host, _, port_text = target.rpartition(":")
            try:
                target_reader, target_writer = await asyncio.open_connection(host, int(port_text))
            except OSError:
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await writer.drain()
                return
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            await splice((reader, writer), (target_reader, target_writer))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def start_recording_http_server(response: bytes = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok") -> tuple[asyncio.AbstractServer, list[dict]]:
    """A plain HTTP server that records each request head and returns a fixed response."""

    seen: list[dict] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            blob = await reader.readuntil(b"\r\n\r\n")
            lines = blob.decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    name, _, value = line.partition(":")
                    headers[name.strip().lower()] = value.strip()
            seen.append({"method": method, "target": target, "headers": headers})
            writer.write(response)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, seen
