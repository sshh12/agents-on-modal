"""A loopback auth-injecting reverse proxy for the dsh harness.

The ``dsh`` DeepSeek provider authenticates its upstream request with a header the Modal endpoint's
edge proxy-auth rejects with 401 (the edge accepts only ``Authorization: Bearer <token>`` or
``Modal-Key``/``Modal-Secret``; dsh sends neither). Rather than reverse-engineer the bundled dsh
runtime, this tiny loopback proxy bridges the gap: dsh is pointed at ``http://127.0.0.1:<port>/v1``
and every request is forwarded verbatim to the real gateway with ``Authorization: Bearer <token>``
injected (the client's own auth stripped). Responses stream back unbuffered so SSE works.

It binds only to loopback inside the sandbox, forwards one upstream host, and holds the token in
memory for the process lifetime -- the token never reaches ``/shared`` or disk. Pure stdlib, so the
harness stays dependency-free and this is unit-testable offline against a local echo server.
"""

from __future__ import annotations

import http.client
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

#: Request headers never forwarded upstream (we set Host/auth ourselves; drop the client's creds).
_DROP_REQUEST = {
    "host", "authorization", "x-api-key", "api-key", "connection", "proxy-authorization",
    "content-length", "accept-encoding", "transfer-encoding", "keep-alive", "te", "upgrade",
}
#: Response headers never forwarded downstream (we re-frame the body and close the connection).
_DROP_RESPONSE = {
    "connection", "transfer-encoding", "content-length", "keep-alive", "content-encoding",
    "proxy-authenticate",
}


def _log(msg: str) -> None:
    # stderr -> the runner tees it into the trace as a harness line (a ``log`` event).
    sys.stderr.write(f"[dsh-auth-proxy] {msg}\n")
    sys.stderr.flush()


def make_handler(upstream: str, token: str, *, verbose: bool) -> type[BaseHTTPRequestHandler]:
    parts = urlsplit(upstream)
    up_scheme = parts.scheme or "https"
    up_host = parts.hostname or ""
    up_port = parts.port or (443 if up_scheme == "https" else 80)
    up_prefix = parts.path.rstrip("/")  # e.g. "" for a bare root

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "aom-dsh-auth-proxy/1"

        def log_message(self, *_a: Any) -> None:  # silence the default stderr spam
            return

        def _proxy(self) -> None:
            self.close_connection = True  # read-until-close framing (below)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""

            headers = {k: v for k, v in self.headers.items() if k.lower() not in _DROP_REQUEST}
            headers["Authorization"] = f"Bearer {token}"
            headers["Accept-Encoding"] = "identity"
            headers["Host"] = up_host
            path = up_prefix + self.path

            conn_cls = http.client.HTTPSConnection if up_scheme == "https" else http.client.HTTPConnection
            conn = conn_cls(up_host, up_port, timeout=7200)
            try:
                conn.request(self.command, path, body=body or None, headers=headers)
                resp = conn.getresponse()
                if verbose:
                    _log(f"{self.command} {self.path} -> {resp.status} ({up_host}{path})")
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in _DROP_RESPONSE:
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except Exception as exc:  # noqa: BLE001 - surface upstream failures as a 502
                if verbose:
                    _log(f"{self.command} {self.path} -> upstream error {exc!r}")
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(f"auth-proxy upstream error: {exc}".encode())
                except Exception:  # noqa: BLE001 - client already gone
                    pass
            finally:
                conn.close()

        # every method funnels through _proxy
        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _proxy

    return Handler


def start(upstream: str, token: str, *, host: str = "127.0.0.1", verbose: bool = False) -> tuple[str, ThreadingHTTPServer]:
    """Start the proxy on an ephemeral loopback port. Returns ``(base_url, server)``.

    ``base_url`` is the loopback root to hand dsh (callers typically append ``/v1``). ``upstream`` is
    the gateway root (no ``/v1``); the client's path is forwarded onto it unchanged. Stop with
    ``server.shutdown()``.
    """

    handler = make_handler(upstream.rstrip("/"), token, verbose=verbose)
    httpd = ThreadingHTTPServer((host, 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, name="dsh-auth-proxy", daemon=True)
    thread.start()
    if verbose:
        _log(f"listening on {host}:{port} -> {upstream}")
    return f"http://{host}:{port}", httpd


__all__ = ["start", "make_handler"]
