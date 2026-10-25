#!/usr/bin/env python3
"""Dashboard backend for aom swarms -- stdlib ``http.server`` only, no third-party deps.

It is **stateless**: every request rediscovers the live swarm with
``agents_on_modal.swarm.Swarm.reconnect(swarm_id)`` (which lists the tagged Modal sandboxes and
reattaches to their runs), so a browser refresh or a server restart resumes a mid-flight swarm with
no local state. Trace tailing is offset-based: the browser keeps the byte offset per agent and the
server returns only the new events, so polling is cheap and idempotent.

    AOM_SWARM=swarm-20261025-abc123 python examples/swarm/server.py
    # then open http://127.0.0.1:8787

Endpoints
---------
    GET  /                                      -> index.html
    GET  /api/agents?swarm=<id>                 -> {swarm, model, agents:[{id,run_id,state,idx}]}
    GET  /api/events?swarm=&agent=&run=&offset= -> {events:[...], offset, state, done}
    POST /api/spawn  {harness,model,count,...}  -> {swarm} (launches a new swarm; hits Modal)
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents_on_modal import sandbox  # noqa: E402
from agents_on_modal.config import AomError  # noqa: E402
from agents_on_modal.swarm import Swarm  # noqa: E402

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"
HOST = os.environ.get("AOM_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("AOM_DASHBOARD_PORT", "8787"))
TERMINAL = ("done", "timeout", "failed", "cancelled")


def _agents_payload(swarm_id: str) -> dict:
    swarm = Swarm.reconnect(swarm_id)
    agents = []
    for agent in swarm.agents():
        agents.append({
            "id": agent.id,
            "run_id": agent.run_id,
            "idx": agent.idx,
            "state": agent.state(),
        })
    return {"swarm": swarm.id, "model": swarm.model, "agents": agents}


def _events_payload(agent_id: str, run_id: str, offset: int) -> dict:
    box = sandbox.get("modal", agent_id)
    run = box.run(run_id)
    envelopes, new_offset = run.read_trace(offset)
    events = [event.to_json() for envelope in envelopes for event in run.parse(envelope)]
    state = str(run.status().get("state", "starting"))
    return {"events": events, "offset": new_offset, "state": state, "done": state in TERMINAL}


def _spawn(body: dict) -> dict:
    harness = body.get("harness") or "claude"
    model = body.get("model")
    count = int(body.get("count") or 1)
    if not model:
        raise AomError("spawn requires a model name.")
    swarm = Swarm.launch(
        harness, model, body.get("prompt") or "",
        count=count,
        goal=bool(body.get("goal")),
        timeout=float(body["timeout"]) if body.get("timeout") else None,
        proxy=body.get("proxy") or None,
        effort=body.get("effort") or None,
    )
    return {"swarm": swarm.id, "model": swarm.model, "count": len(swarm.agents())}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # quieter console
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._write_response(status, "application/json", body)

    def _send_index(self) -> None:
        try:
            body = INDEX.read_bytes()
        except OSError:
            self._send_json({"error": "index.html missing"}, 500)
            return
        self._write_response(200, "text/html; charset=utf-8", body)

    def _write_response(self, status: int, content_type: str, body: bytes) -> None:
        # The dashboard is polled continuously; a browser that navigates away or refreshes
        # drops the socket mid-write, which surfaces as ConnectionError here. That is normal
        # client behavior, not a server fault, so swallow it instead of letting it bubble up
        # as a noisy traceback (see handle_error on the server below).
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            self.close_connection = True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        one = lambda key, default="": (query.get(key) or [default])[0]  # noqa: E731
        try:
            if parsed.path in ("/", "/index.html"):
                self._send_index()
            elif parsed.path == "/api/agents":
                self._send_json(_agents_payload(one("swarm", os.environ.get("AOM_SWARM", ""))))
            elif parsed.path == "/api/events":
                self._send_json(_events_payload(one("agent"), one("run"), int(one("offset", "0") or 0)))
            elif parsed.path == "/api/config":
                self._send_json({"swarm": os.environ.get("AOM_SWARM", "")})
            else:
                self._send_json({"error": "not found"}, 404)
        except AomError as error:
            self._send_json({"error": str(error)}, 400)
        except Exception as error:  # noqa: BLE001 - keep the dev server alive
            self._send_json({"error": f"{type(error).__name__}: {error}"}, 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
            if parsed.path == "/api/spawn":
                self._send_json(_spawn(body))
            else:
                self._send_json({"error": "not found"}, 404)
        except AomError as error:
            self._send_json({"error": str(error)}, 400)
        except Exception as error:  # noqa: BLE001
            self._send_json({"error": f"{type(error).__name__}: {error}"}, 500)


class _QuietServer(ThreadingHTTPServer):
    """A browser polling the dashboard and then navigating away aborts in-flight requests;
    on Windows that is ConnectionAbortedError. These are client-side and benign, so we drop
    them silently instead of printing a traceback per aborted poll (which drowned the log)."""

    def handle_error(self, request: object, client_address: object) -> None:  # type: ignore[override]
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def main() -> int:
    server = _QuietServer((HOST, PORT), Handler)
    swarm_id = os.environ.get("AOM_SWARM", "")
    print(f"aom swarm dashboard on http://{HOST}:{PORT}"
          + (f"  (swarm {swarm_id})" if swarm_id else "  (set AOM_SWARM or spawn one in the UI)"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
