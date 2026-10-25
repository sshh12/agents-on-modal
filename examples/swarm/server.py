#!/usr/bin/env python3
"""Dashboard backend for aom swarms -- stdlib ``http.server`` only, no third-party deps.

It is **stateless**: every request rediscovers the live swarm from Modal (naive swarms via
``Swarm.reconnect``) or from the shared control volume (dynamic swarms), so a browser refresh or a
server restart resumes a mid-flight swarm with no local state. Trace tailing is offset-based.

Two modes:
  * **naive**   -- N identical agents (``Swarm``).
  * **dynamic** -- one orchestrator agent that spawns subagents in waves/pools via the
    ``aom-swarm`` CLI; a local :class:`~agents_on_modal.dynamic.Controller` (started here in a
    thread) executes its plans against Modal and publishes state to ``/shared/_swarm``.

    AOM_SWARM=swarm-20261025-abc123 python examples/swarm/server.py   # then open :8787

Endpoints
---------
    GET  /                                      -> index.html
    GET  /api/config                            -> {swarm}
    GET  /api/agents?swarm=<id>                 -> {swarm, model, mode, agents:[...], workflows:[...]}
    GET  /api/events?agent=&run=&offset=        -> {events, offset, state, done}
    GET  /api/files?swarm=&path=/               -> {paths:[...]}            (live /shared listing)
    GET  /api/file?swarm=&path=...              -> {path, text, truncated}  (one shared file)
    GET  /api/injected?harness=&model=          -> {text}                   (orchestrator context)
    POST /api/spawn  {mode,harness,model,...}   -> {swarm, mode}            (launches; hits Modal)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents_on_modal import models, sandbox  # noqa: E402
from agents_on_modal.config import AomError  # noqa: E402
from agents_on_modal.dynamic import (  # noqa: E402
    DEFAULT_TIMEOUT,
    RESERVED,
    DynamicSwarm,
    VolumeControlPlane,
    build_orchestrator_instructions,
    _model_efforts,
)
from agents_on_modal.swarm import Swarm  # noqa: E402

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"
HOST = os.environ.get("AOM_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("AOM_DASHBOARD_PORT", "8787"))
TERMINAL = ("done", "timeout", "failed", "cancelled")
#: The orchestrator must outlive the whole run; default its cap generously if the form omits one.
DYNAMIC_ORCHESTRATOR_TIMEOUT = 6 * 3600
MAX_FILE_BYTES = 200_000

# swarm_id -> {"thread": Thread, "stop": Event}. Controllers run here, in the dashboard process.
_controllers: dict[str, dict] = {}
_controllers_lock = threading.Lock()


# --------------------------------------------------------------------------------------------- #
# controller lifecycle (dynamic swarms)                                                         #
# --------------------------------------------------------------------------------------------- #
def _ensure_controller(ds: DynamicSwarm, *, proxy: str | None = None) -> None:
    """Start a controller thread for ``ds`` if one is not already running in this process."""

    with _controllers_lock:
        entry = _controllers.get(ds.id)
        if entry and entry["thread"].is_alive():
            return
        endpoint = models.endpoint(models.get(ds.model))
        controller = ds.controller(endpoint, proxy=proxy)
        stop = threading.Event()
        thread = threading.Thread(
            target=controller.run_forever, kwargs={"stop": stop}, daemon=True,
            name=f"aom-controller-{ds.id}",
        )
        _controllers[ds.id] = {"thread": thread, "stop": stop}
        thread.start()


def _read_meta(swarm_id: str) -> dict | None:
    """The dynamic-swarm meta (``None`` for a naive swarm or one not yet started)."""

    try:
        return VolumeControlPlane(f"aom-shared-{swarm_id}").get_json(f"{RESERVED}/meta.json")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------------- #
# payloads                                                                                       #
# --------------------------------------------------------------------------------------------- #
def _naive_agents_payload(swarm_id: str) -> dict:
    swarm = Swarm.reconnect(swarm_id)
    agents = [
        {"id": a.id, "run_id": a.run_id, "idx": a.idx, "state": a.state(), "role": "agent"}
        for a in swarm.agents()
    ]
    return {"swarm": swarm.id, "model": swarm.model, "mode": "naive", "agents": agents,
            "workflows": []}


def _dynamic_agents_payload(swarm_id: str, meta: dict) -> dict:
    control = VolumeControlPlane(f"aom-shared-{swarm_id}")
    # resume: make sure a controller is driving this swarm (e.g. after a server restart)
    ds = DynamicSwarm.reconnect(swarm_id)
    if ds.model:
        try:
            _ensure_controller(ds)  # proxy is not recoverable after a restart (never on the volume)
        except Exception:  # noqa: BLE001 - surfacing agents still works even if resume fails
            pass

    agents: list[dict] = []
    if meta.get("orchestrator_id"):
        agents.append({"id": meta["orchestrator_id"], "run_id": meta.get("orchestrator_run_id"),
                       "idx": None, "role": "orchestrator", "workflow": None, "stage": None})
    for name in control.listdir(f"{RESERVED}/agents"):
        if not name.endswith(".json"):
            continue
        a = control.get_json(f"{RESERVED}/agents/{name}")
        if a and a.get("id"):
            agents.append({"id": a["id"], "run_id": a.get("run_id"), "idx": a.get("idx"),
                           "role": "subagent", "workflow": a.get("workflow"),
                           "stage": a.get("stage")})
    workflows = []
    for name in control.listdir(f"{RESERVED}/workflows"):
        if name.endswith(".json"):
            wf = control.get_json(f"{RESERVED}/workflows/{name}")
            if wf:
                workflows.append(wf)
    return {"swarm": swarm_id, "model": meta.get("model", ""), "mode": "dynamic",
            "agents": agents, "workflows": workflows}


def _agents_payload(swarm_id: str) -> dict:
    if not swarm_id:
        return {"swarm": "", "model": "", "mode": "naive", "agents": [], "workflows": []}
    meta = _read_meta(swarm_id)
    if meta:
        return _dynamic_agents_payload(swarm_id, meta)
    return _naive_agents_payload(swarm_id)


def _events_payload(agent_id: str, run_id: str, offset: int) -> dict:
    box = sandbox.get("modal", agent_id)
    run = box.run(run_id)
    envelopes, new_offset = run.read_trace(offset)
    events = [event.to_json() for envelope in envelopes for event in run.parse(envelope)]
    state = str(run.status().get("state", "starting"))
    return {"events": events, "offset": new_offset, "state": state, "done": state in TERMINAL}


def _files_payload(swarm_id: str, path: str) -> dict:
    swarm = Swarm(id=swarm_id, model="", shared_volume=f"aom-shared-{swarm_id}")
    try:
        paths = swarm.list_shared(path or "/", recursive=True)
    except AomError:
        paths = []
    return {"paths": sorted(paths)}


def _file_payload(swarm_id: str, path: str) -> dict:
    swarm = Swarm(id=swarm_id, model="", shared_volume=f"aom-shared-{swarm_id}")
    data = swarm.read_shared(path)
    truncated = len(data) > MAX_FILE_BYTES
    text = data[:MAX_FILE_BYTES].decode("utf-8", "replace")
    return {"path": path, "text": text, "truncated": truncated}


def _injected_payload(harness: str, model: str) -> dict:
    m = models.get(model)
    text = build_orchestrator_instructions(
        default_timeout=DEFAULT_TIMEOUT, model_name=m.name, allowed_efforts=_model_efforts(m)
    )
    return {"text": text, "model": m.name, "harness": harness or "claude"}


def _spawn(body: dict) -> dict:
    mode = (body.get("mode") or "naive").lower()
    harness = body.get("harness") or "claude"
    model = body.get("model")
    if not model:
        raise AomError("spawn requires a model name.")
    proxy = body.get("proxy") or None
    effort = body.get("effort") or None

    if mode == "dynamic":
        timeout = float(body["timeout"]) if body.get("timeout") else DYNAMIC_ORCHESTRATOR_TIMEOUT
        ds = DynamicSwarm.launch(
            harness, model, body.get("prompt") or "",
            timeout=timeout, effort=effort, proxy=proxy,
        )
        _ensure_controller(ds, proxy=proxy)
        return {"swarm": ds.id, "model": ds.model, "mode": "dynamic"}

    swarm = Swarm.launch(
        harness, model, body.get("prompt") or "",
        count=int(body.get("count") or 1),
        goal=bool(body.get("goal")),
        timeout=float(body["timeout"]) if body.get("timeout") else None,
        proxy=proxy, effort=effort,
    )
    return {"swarm": swarm.id, "model": swarm.model, "mode": "naive",
            "count": len(swarm.agents())}


# --------------------------------------------------------------------------------------------- #
# HTTP                                                                                          #
# --------------------------------------------------------------------------------------------- #
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
            elif parsed.path == "/api/files":
                self._send_json(_files_payload(one("swarm", os.environ.get("AOM_SWARM", "")), one("path", "/")))
            elif parsed.path == "/api/file":
                self._send_json(_file_payload(one("swarm", os.environ.get("AOM_SWARM", "")), one("path")))
            elif parsed.path == "/api/injected":
                self._send_json(_injected_payload(one("harness"), one("model", "qwen3.8-27b")))
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
