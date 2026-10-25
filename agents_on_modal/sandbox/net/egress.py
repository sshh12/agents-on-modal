"""Egress broker for the Docker sandbox. Run as ``python -m agents_on_modal.sandbox.net.egress``.

The agent container has no route to the internet and no DNS; everything it reaches goes through
this process, which runs in a separate container the agent cannot touch. Two listeners:

  * ``:3128``  HTTP forward proxy (CONNECT and absolute-URI ``http://``). Applies ``policy.py`` to
    every destination (public IPs only, allow-listed ports, no rebinding). If ``AOM_UPSTREAM_PROXY``
    is set, the validated outbound connection is chained *through* that residential proxy; otherwise
    it is made directly.

  * ``:8081``  model reverse proxy. The agent speaks plain HTTP with no credential; this process
    attaches ``Authorization: Bearer <AOM_MODEL_TOKEN>`` and forwards to ``AOM_MODEL_URL``. This path
    is **always direct** and never uses the upstream residential proxy, so the user is not billed for
    gigabytes of model traffic over a metered residential link.

Every decision is a JSONL record on stdout so the host can export the egress log after a run.

This file ships into Linux containers; keep it LF + POSIX.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import sys
from urllib.parse import urlsplit

from . import policy as policy_module
from . import proxy as proxy_module
from .policy import Policy, PolicyError
from .proxy import (
    CONNECT_TIMEOUT_SECONDS,
    RequestHead,
    UpstreamProxy,
    close_writer,
    log_line,
    parse_proxy,
    read_request_head,
    rebuild_request,
    send_simple_response,
    split_authority,
    splice,
)


class ForwardProxy:
    """HTTP forward proxy for the agent's own traffic: policy-checked, optionally proxy-chained."""

    def __init__(self, policy: Policy, upstream: UpstreamProxy | None = None) -> None:
        self.policy = policy
        self.upstream = upstream

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await read_request_head(reader)
            if head is None:
                return
            if head.method.upper() == "CONNECT":
                await self._connect(head, reader, writer)
            else:
                await self._absolute(head, reader, writer)
        except ValueError as error:
            log_line(listener="forward", event="reject", reason=str(error))
            await send_simple_response(writer, "400 Bad Request", f"{error}\n")
        except Exception as error:  # pragma: no cover - defensive
            log_line(listener="forward", event="error", reason=repr(error))
        finally:
            await close_writer(writer)

    async def _open(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, str, str]:
        """Resolve+validate, then connect directly or via the upstream proxy."""

        address, port = await self.policy.resolve(host, port)
        if self.upstream is not None:
            reader, writer = await self.upstream.open_connection(
                address, port, timeout=CONNECT_TIMEOUT_SECONDS
            )
            return reader, writer, address, "upstream"
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(address, port), timeout=CONNECT_TIMEOUT_SECONDS
        )
        return reader, writer, address, "direct"

    async def _connect(
        self, head: RequestHead, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        host, port = split_authority(head.target, 443)
        try:
            upstream_reader, upstream_writer, address, route = await self._open(host, port)
        except PolicyError as error:
            log_line(listener="forward", event="deny", method="CONNECT", host=host, port=port, reason=str(error))
            await send_simple_response(writer, "403 Forbidden", f"{error}\n")
            return
        except (OSError, asyncio.TimeoutError, proxy_module.ProxyError) as error:
            log_line(listener="forward", event="unreachable", method="CONNECT", host=host, port=port, reason=str(error))
            await send_simple_response(writer, "502 Bad Gateway", f"{error}\n")
            return
        log_line(listener="forward", event="allow", method="CONNECT", host=host, port=port, address=address, route=route)
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        try:
            await splice((reader, writer), (upstream_reader, upstream_writer))
        finally:
            await close_writer(upstream_writer)

    async def _absolute(
        self, head: RequestHead, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if not head.target.lower().startswith("http://"):
            raise ValueError(
                f"only CONNECT and absolute http:// targets are proxied; got {head.target!r}"
            )
        parts = urlsplit(head.target)
        host, port = split_authority(parts.netloc, 80)
        try:
            upstream_reader, upstream_writer, address, route = await self._open(host, port)
        except PolicyError as error:
            log_line(listener="forward", event="deny", method=head.method, host=host, port=port, reason=str(error))
            await send_simple_response(writer, "403 Forbidden", f"{error}\n")
            return
        except (OSError, asyncio.TimeoutError, proxy_module.ProxyError) as error:
            log_line(listener="forward", event="unreachable", method=head.method, host=host, port=port, reason=str(error))
            await send_simple_response(writer, "502 Bad Gateway", f"{error}\n")
            return
        origin_target = parts.path or "/"
        if parts.query:
            origin_target += f"?{parts.query}"
        log_line(listener="forward", event="allow", method=head.method, host=host, port=port, address=address, route=route, path=parts.path)
        upstream_writer.write(rebuild_request(head, target=origin_target, host_header=parts.netloc))
        await upstream_writer.drain()
        try:
            await splice((reader, writer), (upstream_reader, upstream_writer))
        finally:
            await close_writer(upstream_writer)


class ModelProxy:
    """Reverse proxy that injects the model credential the agent never holds. Always direct."""

    def __init__(self, upstream_url: str, authorization: str | None, local_prefix: str = "/v1") -> None:
        parts = urlsplit(upstream_url)
        if parts.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported model upstream scheme in {upstream_url!r}.")
        self.scheme = parts.scheme
        self.host, self.port = split_authority(parts.netloc, 443 if parts.scheme == "https" else 80)
        self.base_path = parts.path.rstrip("/")
        self.authorization = authorization
        self.local_prefix = local_prefix.rstrip("/")
        self.host_header = parts.netloc

    def upstream_target(self, target: str) -> str:
        path, separator, query = target.partition("?")
        if self.local_prefix and path.startswith(self.local_prefix):
            path = path[len(self.local_prefix) :]
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_path}{path}" + (f"{separator}{query}" if separator else "")

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            head = await read_request_head(reader)
            if head is None:
                return
            target = self.upstream_target(head.target)
            context = ssl.create_default_context() if self.scheme == "https" else None
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.host,
                    self.port,
                    ssl=context,
                    server_hostname=self.host if context else None,
                ),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
            extra: tuple[tuple[str, str], ...] = ()
            if self.authorization:
                extra = (("Authorization", self.authorization),)
            log_line(
                listener="model",
                event="forward",
                method=head.method,
                path=head.target,
                upstream=f"{self.scheme}://{self.host_header}{target}",
                credential="injected" if self.authorization else "none",
            )
            upstream_writer.write(
                rebuild_request(head, target=target, host_header=self.host_header, extra_headers=extra)
            )
            await upstream_writer.drain()
            await splice((reader, writer), (upstream_reader, upstream_writer))
        except ValueError as error:
            log_line(listener="model", event="reject", reason=str(error))
            await send_simple_response(writer, "400 Bad Request", f"{error}\n")
        except (OSError, asyncio.TimeoutError, ssl.SSLError) as error:
            log_line(listener="model", event="error", reason=str(error))
            await send_simple_response(writer, "502 Bad Gateway", f"{error}\n")
        finally:
            if upstream_writer is not None:
                await close_writer(upstream_writer)
            await close_writer(writer)


def _authorization_header(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"


async def main() -> int:
    policy = Policy(
        policy_module.parse_ports(os.environ.get("AOM_EGRESS_ALLOW_PORTS", "80,443")),
        policy_module.parse_hosts(os.environ.get("AOM_EGRESS_ALLOW_HOSTS", "")),
    )
    upstream = parse_proxy(
        os.environ.get("AOM_UPSTREAM_PROXY") or None,
        session=os.environ.get("AOM_PROXY_SESSION") or None,
    )
    forward_port = int(os.environ.get("AOM_EGRESS_PORT", "3128"))
    model_port = int(os.environ.get("AOM_MODEL_PORT", "8081"))
    model_url = (os.environ.get("AOM_MODEL_URL") or "").strip()
    authorization = _authorization_header(os.environ.get("AOM_MODEL_TOKEN", ""))

    servers = []
    forward = ForwardProxy(policy, upstream)
    servers.append(await asyncio.start_server(forward.handle, "0.0.0.0", forward_port))
    log_line(
        listener="forward",
        event="listening",
        port=forward_port,
        allowed_ports=sorted(policy.allowed_ports),
        allowed_hosts=list(policy.allowed_hosts) or ["<any public address>"],
        upstream=upstream.url if upstream else None,
    )
    if model_url:
        model = ModelProxy(model_url, authorization, os.environ.get("AOM_MODEL_PREFIX", "/v1"))
        servers.append(await asyncio.start_server(model.handle, "0.0.0.0", model_port))
        log_line(
            listener="model",
            event="listening",
            port=model_port,
            upstream=f"{model.scheme}://{model.host_header}{model.base_path}",
            credential="injected" if model.authorization else "none",
        )
    async with servers[0]:
        await asyncio.gather(*(server.serve_forever() for server in servers))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
