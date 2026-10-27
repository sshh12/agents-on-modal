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
    GET  /api/models                            -> {models:[{name,description,gpu,context_window}]}
    GET  /api/presets                           -> {presets:[{name,label,mode,prompt?,locked}]}
    POST /api/spawn  {mode,harness,model,...}   -> {swarm, mode}            (launches; hits Modal)
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
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

import presets as presets_mod  # noqa: E402 - local module beside this file

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"
HOST = os.environ.get("AOM_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("AOM_DASHBOARD_PORT", "8787"))
TERMINAL = ("done", "timeout", "failed", "cancelled", "stopped")
#: The orchestrator must outlive the whole run; default its cap generously if the form omits one.
DYNAMIC_ORCHESTRATOR_TIMEOUT = 6 * 3600
MAX_FILE_BYTES = 200_000
#: Residential proxy pool: a gitignored ``proxies.csv`` beside this file, one proxy URL per line.
#: When a spawn sets ``use_proxies``, swarm agents egress browser/web traffic through one of these
#: (the sandbox relay); model inference ALWAYS stays direct (NO_PROXY), so no model bytes are metered
#: over the residential link.
PROXIES_FILE = HERE.parents[1] / "proxies.csv"  # repo root (gitignored, holds residential proxy URLs)


def _load_proxies() -> list[str]:
    """Proxy URLs from proxies.csv (blank lines and ``#`` comments skipped); ``[]`` if none."""

    try:
        lines = PROXIES_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _proxy_template() -> str | None:
    """A residential proxy URL whose sticky-session token is templated to ``{session}`` so each
    agent sandbox egresses from its own sticky residential IP (``parse_proxy`` substitutes
    ``{session}`` with the sandbox id per sandbox). ``None`` when no proxies are configured.
    """

    proxies = _load_proxies()
    if not proxies:
        return None
    # decodo-style sticky session lives in the username (``...-session-<token>:``); template it so
    # every sandbox gets a distinct sticky session (and thus a distinct residential IP).
    return re.sub(r"(-session-)[^-:@/\s]+", r"\g<1>{session}", proxies[0], count=1)


def _proxy_optin_note() -> str:
    """A succinct, token-efficient block injected into the task for 'opt-in' mode: the raw SOCKS
    endpoint(s) + a one-line WHEN + a one-line HOW, so the agent routes through a residential proxy
    only when it actually needs to (proxy traffic costs money). Browser/web ONLY -- never the model.
    """

    eps = " , ".join(_load_proxies()[:2])  # a couple of distinct sticky endpoints to choose from
    return (
        "## Residential proxies (optional — use for browser/web requests ONLY, never for anything else)\n"
        f"SOCKS5 endpoints: {eps}\n"
        "WHEN: use ONLY if a site blocks or rate-limits the sandbox's direct IP — they cost money, so "
        "the direct IP is the default; do not route traffic through them otherwise.\n"
        "HOW: `curl --socks5-hostname '<endpoint>' <url>`, or `export ALL_PROXY='<endpoint>'` for a "
        "shell/tool, or set that ALL_PROXY before launching `agent-browser`."
    )

# swarm_id -> {"thread": Thread, "stop": Event}. Controllers run here, in the dashboard process.
_controllers: dict[str, dict] = {}
_controllers_lock = threading.Lock()


# --------------------------------------------------------------------------------------------- #
# controller lifecycle (dynamic swarms)                                                         #
# --------------------------------------------------------------------------------------------- #
def _ensure_controller(ds: DynamicSwarm, *, proxy: str | None = None,
                       proxy_note: str | None = None) -> None:
    """Start a controller thread for ``ds`` if one is not already running in this process."""

    with _controllers_lock:
        entry = _controllers.get(ds.id)
        if entry and entry["thread"].is_alive():
            return
        endpoint = models.endpoint(models.get(ds.model))
        controller = ds.controller(endpoint, proxy=proxy, proxy_note=proxy_note)
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


def _orchestrator_state(meta: dict) -> str:
    """Live state of the orchestrator sandbox (its state is not published to the volume)."""

    oid, rid = meta.get("orchestrator_id"), meta.get("orchestrator_run_id")
    if not oid or not rid:
        return "running"
    try:
        return str(sandbox.get("modal", oid).run(rid).status().get("state", "running"))
    except Exception as error:  # noqa: BLE001
        # A gone/terminated orchestrator sandbox cannot be read; report it as stopped rather than
        # forever "running". Only a genuine not-found counts as gone -- a transient read blip during
        # a live run still assumes running so the UI keeps polling.
        msg = str(error).lower()
        if "no modal sandbox" in msg or "not found" in msg or "does not exist" in msg:
            return "stopped"
        return "running"


def _models_payload() -> dict:
    """The model catalog, for the dashboard's model dropdown."""

    out = [
        {"name": m.name, "repo": m.repo, "description": m.description, "gpu": m.gpu,
         "context_window": m.context_window}
        for m in models.list_models()
    ]
    return {"models": out}


def _endpoint_status(model) -> dict:
    """Live status of one model's shared Modal endpoint (deployed? current autoscaler cap?).

    A model endpoint is an account-level shared resource: every swarm (naive or dynamic) on the
    same model id hits the same Modal app under one autoscaler. This reports whether that app is
    deployed and its current ``max_containers`` -- the global cap all those swarms share.
    """

    info = {"name": model.name, "repo": model.repo, "gpu": model.gpu,
            "context_window": model.context_window, "deployed": False,
            "max_containers": None, "min_containers": None}
    try:
        st = models.status(model)
        info["url"] = st.get("url")
        if st.get("server_error"):
            info["deployed"] = False
        else:
            asg = st.get("autoscaler") or {}
            info["deployed"] = True
            info["max_containers"] = asg.get("max_containers")
            info["min_containers"] = asg.get("min_containers")
    except Exception as error:  # noqa: BLE001 - not deployed / unreachable
        info["error"] = f"{type(error).__name__}: {error}"
    return info


def _endpoints_payload() -> dict:
    """Status of every catalog model's shared endpoint, queried concurrently (each hits Modal)."""

    cat = models.list_models()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(_endpoint_status, cat))
    return {"endpoints": out}


def _scale(body: dict) -> dict:
    """Set a model endpoint's global ``max_containers`` live (shared by all swarms on that model)."""

    model = body.get("model")
    if not model:
        raise AomError("scale requires a model name.")
    raw = body.get("max_instances")
    if raw in (None, "", "unbounded"):
        raise AomError("scale requires a positive integer max_instances.")
    try:
        n = max(1, int(raw))
    except (TypeError, ValueError):
        raise AomError(f"invalid max_instances: {raw!r}") from None
    models.scale(models.get(model), max_containers=n)
    return {"model": model, "max_containers": n}


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
                       "idx": None, "role": "orchestrator", "workflow": None, "stage": None,
                       "state": _orchestrator_state(meta)})
    for name in control.listdir(f"{RESERVED}/agents"):
        if not name.endswith(".json"):
            continue
        a = control.get_json(f"{RESERVED}/agents/{name}")
        if a and a.get("id"):
            agents.append({"id": a["id"], "run_id": a.get("run_id"), "idx": a.get("idx"),
                           "role": "subagent", "workflow": a.get("workflow"),
                           "stage": a.get("stage"), "state": a.get("state", "starting")})
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


def _events_payload(agent_id: str, run_id: str, offset: int, swarm_id: str = "") -> dict:
    # Live-first: read the running sandbox's trace (byte-offset, incremental).
    try:
        box = sandbox.get("modal", agent_id)
        run = box.run(run_id)
        envelopes, new_offset = run.read_trace(offset)
        events = [event.to_json() for envelope in envelopes for event in run.parse(envelope)]
        state = str(run.status().get("state", "starting"))
        return {"events": events, "offset": new_offset, "state": state,
                "done": state in TERMINAL, "source": "live"}
    except Exception:  # noqa: BLE001 - sandbox gone/unreachable -> serve the durable copy
        if not swarm_id:
            raise
        # The controller persists every agent's full trace (parsed events) to the volume, so a
        # completed/torn-down agent still renders. Returns the whole trace each call (persisted).
        control = VolumeControlPlane(f"aom-shared-{swarm_id}")
        text = control.read_text(f"{RESERVED}/traces/{agent_id}.jsonl")
        if text is None:
            raise  # nothing live and nothing persisted -> surface the live error; the UI retries
        events = []
        for line in text.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        record = control.get_json(f"{RESERVED}/agents/{agent_id}.json")
        state = (record or {}).get("state", "done")
        # A running subagent whose live read merely blipped keeps done=False so the UI retries live;
        # a terminal subagent or the orchestrator (no agent record once gone) is done.
        done = state in TERMINAL or record is None
        return {"events": events, "offset": len(events), "state": state, "done": done,
                "source": "persisted"}


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


def _presets_payload() -> dict:
    """The swarm presets from presets.json, with encrypted entries decrypted where possible.

    Encrypted entries we can decrypt (the key is in the environment) come back with their
    cleartext ``prompt``; those we cannot come back ``locked: true`` with no prompt, so a
    missing key never fails the whole endpoint.
    """

    try:
        return {"presets": presets_mod.resolve_presets()}
    except Exception as error:  # noqa: BLE001 - a bad presets.json should not 500 the UI
        return {"presets": [], "error": f"{type(error).__name__}: {error}"}


def _injected_payload(harness: str, model: str) -> dict:
    catalog = models.list_models()
    m = models.get(model) if model else (catalog[0] if catalog else None)
    if m is None:
        return {"text": "", "model": "", "harness": harness or "claude"}
    text = build_orchestrator_instructions(
        default_timeout=DEFAULT_TIMEOUT, model_name=m.name, allowed_efforts=_model_efforts(m)
    )
    return {"text": text, "model": m.name, "harness": harness or "claude"}


def _apply_max_instances(model_name: str, max_instances) -> None:
    """Apply the dashboard's max-instances cap to the (shared) endpoint's live autoscaler.

    ``1`` (the default) caps the endpoint at one GPU container so it throttles gracefully instead of
    fanning out GPUs; ``0`` / ``None`` / blank means "unbounded" -- leave the endpoint at its catalog
    autoscaler policy. Best-effort: the model is a shared account-level resource and a spawn must not
    fail just because the live autoscaler update hiccuped.
    """

    if max_instances in (None, "", 0, "0", "unbounded"):
        return  # unbounded -> leave the catalog autoscaler policy untouched
    try:
        n = max(1, int(max_instances))
    except (TypeError, ValueError):
        return
    try:
        models.scale(models.get(model_name), max_containers=n)
    except Exception:  # noqa: BLE001 - shared resource; never fail a spawn on a scaling hiccup
        pass


def _spawn(body: dict) -> dict:
    mode = (body.get("mode") or "naive").lower()
    harness = body.get("harness") or "claude"
    model = body.get("model")
    if not model:
        raise AomError("spawn requires a model name.")
    prompt = body.get("prompt") or ""
    proxy = body.get("proxy") or None
    proxy_note = None
    # proxy_mode: off | always | optin (DEFAULT).
    #   optin  -> (default) egress stays direct, but the raw SOCKS endpoints + a one-line WHEN/HOW are
    #             injected into the task so the agent routes manually only when a site blocks the
    #             direct IP -- minimizes proxy cost (proxy traffic is metered).
    #   always -> relay ALL agent browser/web egress through a residential proxy ({session} gives each
    #             agent sandbox a distinct sticky IP); the agent never sees the proxy credentials.
    #   off    -> no proxy at all.
    # Model inference ALWAYS stays direct in every mode (the relay/NO_PROXY excludes the model host).
    proxy_mode = (body.get("proxy_mode") or "optin").lower()
    if not proxy and _load_proxies():
        if proxy_mode == "always":
            proxy = _proxy_template()
        elif proxy_mode == "optin":
            proxy_note = _proxy_optin_note()
            prompt = proxy_note + "\n\n---\n\n" + prompt  # orchestrator / naive agents
    effort = body.get("effort") or None
    _apply_max_instances(model, body.get("max_instances"))

    if mode == "dynamic":
        timeout = float(body["timeout"]) if body.get("timeout") else DYNAMIC_ORCHESTRATOR_TIMEOUT
        ds = DynamicSwarm.launch(
            harness, model, prompt,
            timeout=timeout, effort=effort, proxy=proxy,
        )
        _ensure_controller(ds, proxy=proxy, proxy_note=proxy_note)  # subagents get the note too
        return {"swarm": ds.id, "model": ds.model, "mode": "dynamic"}

    swarm = Swarm.launch(
        harness, model, prompt,
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
                self._send_json(_events_payload(one("agent"), one("run"), int(one("offset", "0") or 0),
                                                one("swarm", os.environ.get("AOM_SWARM", ""))))
            elif parsed.path == "/api/files":
                self._send_json(_files_payload(one("swarm", os.environ.get("AOM_SWARM", "")), one("path", "/")))
            elif parsed.path == "/api/file":
                self._send_json(_file_payload(one("swarm", os.environ.get("AOM_SWARM", "")), one("path")))
            elif parsed.path == "/api/injected":
                self._send_json(_injected_payload(one("harness"), one("model")))
            elif parsed.path == "/api/models":
                self._send_json(_models_payload())
            elif parsed.path == "/api/endpoints":
                self._send_json(_endpoints_payload())
            elif parsed.path == "/api/presets":
                self._send_json(_presets_payload())
            elif parsed.path == "/api/config":
                self._send_json({"swarm": os.environ.get("AOM_SWARM", ""),
                                 "proxies_available": len(_load_proxies())})
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
            elif parsed.path == "/api/scale":
                self._send_json(_scale(body))
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
