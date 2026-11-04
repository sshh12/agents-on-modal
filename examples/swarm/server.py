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

    python examples/swarm/server.py                       # any python; opens the dashboard
    python examples/swarm/server.py --swarm swarm-...     # reopen a specific swarm

Setup is automatic (see ``examples/bootstrap.py``): it loads the repo ``.env``, re-launches on the repo
``.venv`` (creating it on first run), and runs the Modal login / proxy-token mint if missing.

Endpoints
---------
    GET  /                                      -> index.html
    GET  /api/config                            -> {swarm}
    GET  /api/agents?swarm=<id>                 -> {swarm, model, mode, agents:[...], workflows:[...]}
    GET  /api/events?agent=&run=&offset=[&poffset=] -> {events, offset, state, done, source}
                                                   (persisted: also poffset, reset)
    GET  /api/files?swarm=&path=/               -> {paths:[...]}            (live /shared listing)
    GET  /api/file?swarm=&path=...              -> {path, text, truncated}  (one shared file)
    GET  /api/zip?swarm=[&all=1]                -> application/zip  (/shared, zipped on Modal's side)
    GET  /api/injected?harness=&model=          -> {text}                   (orchestrator context)
    GET  /api/models                            -> {models:[{name,description,gpu,context_window}]}
    GET  /api/presets                           -> {presets:[{name,label,mode,prompt?,locked}]}
    POST /api/spawn  {mode,harness,model,max_instances?,...} -> {swarm, mode}  (launches; hits Modal)
    POST /api/scale  {model,max_instances}      -> {model, max_containers}  (live endpoint cap)
    POST /api/stop   {swarm}                    -> {swarm, mode, found, stopped, errors}
                                                   (hard-kills every live sandbox of the run)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for the shared bootstrap

import bootstrap  # noqa: E402 - stdlib-only module shared by the examples (examples/bootstrap.py)

if __name__ == "__main__":
    bootstrap.prepare()  # .env, interpreter/.venv, Modal login -- must precede the aom imports below

from agents_on_modal import models, sandbox  # noqa: E402
from agents_on_modal.config import AomError  # noqa: E402
from agents_on_modal.dynamic import (  # noqa: E402
    DEFAULT_TIMEOUT,
    RESERVED,
    STOPPED_FILE,
    DynamicSwarm,
    VolumeControlPlane,
    build_orchestrator_instructions,
    _model_efforts,
)
from agents_on_modal.swarm import Swarm, stop_swarm  # noqa: E402

import export  # noqa: E402 - local module beside this file
import presets as presets_mod  # noqa: E402 - local module beside this file

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"
HOST = os.environ.get("AOM_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("AOM_DASHBOARD_PORT", "8787"))
TERMINAL = ("done", "timeout", "failed", "cancelled", "stopped")
#: The orchestrator must outlive the whole run; default its cap generously if the form omits one.
DYNAMIC_ORCHESTRATOR_TIMEOUT = 6 * 3600
#: The model-endpoint GPU container cap a spawn asks for (``max_instances`` in the spawn body
#: overrides it). The first spawn deploys the endpoint at it; on a live endpoint it only ever
#: raises the cap (models.ensure_deployed), so bumping a busy endpoint to 2 for a second, parallel
#: swarm is the endpoints panel's "set" (POST /api/scale), which can also lower it.
SPAWN_MAX_INSTANCES = 1
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

# swarm_id -> {"thread": Thread, "stop": Event, "controller": Controller}. Controllers run here, in
# the dashboard process.
_controllers: dict[str, dict] = {}
_controllers_lock = threading.Lock()
#: How long a hard stop waits for this process's controller to finish its tick and halt, before its
#: second sweep picks up any agent the controller was creating mid-wave.
STOP_CONTROLLER_WAIT = 30.0


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
        _controllers[ds.id] = {"thread": thread, "stop": stop, "controller": controller}
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
            "active": False, "active_containers": None,
            "max_containers": None, "min_containers": None,
            "spawn_max_instances": SPAWN_MAX_INSTANCES}  # the cap a spawn asks for
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
            # Deployed != running. Query the live container count so the UI can tell a cold,
            # scaled-to-zero app (no GPU cost) apart from one that's actually serving (warm, billing).
            n = models.active_containers(model)
            info["active_containers"] = n
            info["active"] = bool(n)  # n >= 1 -> containers are up right now
    except Exception as error:  # noqa: BLE001 - not deployed / unreachable
        info["error"] = f"{type(error).__name__}: {error}"
    return info


def _endpoints_payload() -> dict:
    """Status of every catalog model's shared endpoint, queried concurrently (each hits Modal)."""

    cat = models.list_models()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(_endpoint_status, cat))
    return {"endpoints": out}


def _max_instances(raw) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise AomError(f"invalid max_instances: {raw!r}") from None
    if n < 1:
        raise AomError(f"max_instances must be >= 1, got {n}.")
    return n


def _scale(body: dict) -> dict:
    """Set a model endpoint's global ``max_containers`` live (shared by all swarms on that model)."""

    model = body.get("model")
    if not model:
        raise AomError("scale requires a model name.")
    raw = body.get("max_instances")
    if raw in (None, "", "unbounded"):
        raise AomError("scale requires a positive integer max_instances.")
    n = _max_instances(raw)
    models.scale(models.get(model), max_containers=n)
    return {"model": model, "max_containers": n}


def _dynamic_agents_payload(swarm_id: str, meta: dict) -> dict:
    control = VolumeControlPlane(f"aom-shared-{swarm_id}")
    stopped = control.exists(f"{RESERVED}/{STOPPED_FILE}")
    # resume: make sure a controller is driving this swarm (e.g. after a server restart) -- unless
    # it was hard-stopped, which must stay stopped
    ds = DynamicSwarm.reconnect(swarm_id)
    if ds.model and not stopped:
        try:
            _ensure_controller(ds)  # proxy is not recoverable after a restart (never on the volume)
        except Exception:  # noqa: BLE001 - surfacing agents still works even if resume fails
            pass

    agents: list[dict] = []
    if meta.get("orchestrator_id"):
        agents.append({"id": meta["orchestrator_id"], "run_id": meta.get("orchestrator_run_id"),
                       "idx": None, "role": "orchestrator", "workflow": None, "stage": None,
                       "state": _orchestrator_state(meta)})
    # One Modal read per file: in parallel, or ~100 agent records alone took 30-90 s.
    agent_files = [n for n in control.listdir(f"{RESERVED}/agents") if n.endswith(".json")]
    workflow_files = [n for n in control.listdir(f"{RESERVED}/workflows") if n.endswith(".json")]
    paths = ([f"{RESERVED}/agents/{n}" for n in agent_files]
             + [f"{RESERVED}/workflows/{n}" for n in workflow_files])
    with concurrent.futures.ThreadPoolExecutor(max_workers=AGENTS_READ_CONCURRENCY) as pool:
        docs = list(pool.map(control.get_json, paths)) if paths else []
    for a in docs[:len(agent_files)]:
        if a and a.get("id"):
            agents.append({"id": a["id"], "run_id": a.get("run_id"), "idx": a.get("idx"),
                           "role": "subagent", "workflow": a.get("workflow"),
                           "stage": a.get("stage"), "state": a.get("state", "starting")})
    workflows = [wf for wf in docs[len(agent_files):] if wf]
    return {"swarm": swarm_id, "model": meta.get("model", ""), "mode": "dynamic",
            "agents": agents, "workflows": workflows, "stopped": stopped}


#: Concurrent volume reads when building the agents view.
AGENTS_READ_CONCURRENCY = 16
#: The agents view is memoized for a few seconds and built at most once at a time per swarm: while
#: one request rebuilds it, others get the previous payload (or wait for the first build) instead
#: of each starting its own -- the UI polls every 2 s, and overlapping 30-90 s builds of a
#: ~100-agent swarm piled up until nothing answered.
_AGENTS_CACHE: dict[str, tuple[float, dict]] = {}
_AGENTS_BUILDING: dict[str, threading.Event] = {}
_AGENTS_CACHE_LOCK = threading.Lock()
AGENTS_CACHE_TTL = 3.0
#: How long a request with nothing cached waits for another request's build before building itself.
AGENTS_BUILD_WAIT = 120.0


def _build_agents_payload(swarm_id: str) -> dict:
    meta = _read_meta(swarm_id)
    return _dynamic_agents_payload(swarm_id, meta) if meta else _naive_agents_payload(swarm_id)


def _agents_payload(swarm_id: str) -> dict:
    if not swarm_id:
        return {"swarm": "", "model": "", "mode": "naive", "agents": [], "workflows": []}
    with _AGENTS_CACHE_LOCK:
        hit = _AGENTS_CACHE.get(swarm_id)
        if hit and time.time() - hit[0] < AGENTS_CACHE_TTL:
            return hit[1]
        building = _AGENTS_BUILDING.get(swarm_id)
        owner = building is None
        if owner:
            building = _AGENTS_BUILDING[swarm_id] = threading.Event()
    if not owner:
        if hit:
            return hit[1]  # stale while the in-flight build refreshes it
        building.wait(AGENTS_BUILD_WAIT)
        with _AGENTS_CACHE_LOCK:
            hit = _AGENTS_CACHE.get(swarm_id)
        if hit:
            return hit[1]
        return _build_agents_payload(swarm_id)  # the other build failed: try ourselves
    try:
        payload = _build_agents_payload(swarm_id)
        with _AGENTS_CACHE_LOCK:
            _AGENTS_CACHE[swarm_id] = (time.time(), payload)
        return payload
    finally:
        with _AGENTS_CACHE_LOCK:
            _AGENTS_BUILDING.pop(swarm_id, None)
        building.set()


def _invalidate_agents(swarm_id: str) -> None:
    with _AGENTS_CACHE_LOCK:
        _AGENTS_CACHE.pop(swarm_id, None)


def _events_payload(agent_id: str, run_id: str, offset: int, swarm_id: str = "",
                    poffset: int | None = None) -> dict:
    """``offset`` is the live cursor (bytes of the sandbox's raw trace) and ``poffset`` the persisted
    one (bytes of the controller's parsed JSONL) -- two different files, so never one cursor."""

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
        # completed/torn-down agent still renders.
        control = VolumeControlPlane(f"aom-shared-{swarm_id}")
        text = control.read_text(f"{RESERVED}/traces/{agent_id}.jsonl")
        if text is None:
            raise  # nothing live and nothing persisted -> surface the live error; the UI retries
        # Incremental on its own cursor, so a 5,000-event trace of a running agent whose live read
        # keeps blipping is not re-shipped every poll. ``reset``: the client must REPLACE what it
        # shows -- a first persisted read, or the file shrank (the controller caps it from the
        # front), so the old cursor no longer points at a line boundary.
        raw = text.encode("utf-8")
        reset = poffset is None or poffset > len(raw)
        start = 0 if reset else poffset
        events = []
        for line in raw[start:].splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        record = control.get_json(f"{RESERVED}/agents/{agent_id}.json")
        state = (record or {}).get("state", "done")
        # A running subagent whose live read merely blipped keeps done=False so the UI retries live;
        # a terminal subagent or the orchestrator (no agent record once gone) is done.
        done = state in TERMINAL or record is None
        return {"events": events, "offset": offset, "poffset": len(raw), "reset": reset,
                "state": state, "done": done, "source": "persisted"}


def _files_payload(swarm_id: str, path: str) -> dict:
    swarm = Swarm(id=swarm_id, model="", shared_volume=f"aom-shared-{swarm_id}")
    try:
        paths = swarm.list_shared(path or "/", recursive=True)
    except AomError:
        paths = []
    return {"paths": sorted(paths)}


def _file_payload(swarm_id: str, path: str, offset: int = 0) -> dict:
    """One shared file, as a byte-offset window so a large file can be paged instead of silently cut
    mid-content (issue S7). ``offset`` is a byte offset; the reply carries ``size``/``next_offset``/
    ``eof`` so a client can fetch the rest and reassemble. ``truncated`` is kept for back-compat
    (true when the whole file exceeds one window)."""

    swarm = Swarm(id=swarm_id, model="", shared_volume=f"aom-shared-{swarm_id}")
    data = swarm.read_shared(path)
    total = len(data)
    offset = max(0, min(offset, total))
    chunk = data[offset:offset + MAX_FILE_BYTES]
    next_offset = offset + len(chunk)
    text = chunk.decode("utf-8", "replace")
    return {"path": path, "text": text, "size": total, "offset": offset,
            "next_offset": next_offset, "eof": next_offset >= total,
            "truncated": total > MAX_FILE_BYTES}


#: Swarm ids are minted by ``sandbox.new_id``; anything else is not a shared volume we made.
_SWARM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _zip_export(swarm_id: str, all_files: bool) -> tuple:
    """The swarm's shared volume as a zip: ``(file, size, filename, stats)``, zipped on Modal's side.

    Built into a temp file before any response byte is sent, so a failure is a JSON error, never a
    truncated download. Needs only the volume, so it works after the swarm's sandboxes are gone.
    """

    if not _SWARM_ID.fullmatch(swarm_id or ""):
        raise AomError(f"invalid swarm id: {swarm_id!r}")
    volume = f"aom-shared-{swarm_id}"
    handle = tempfile.TemporaryFile()
    try:
        stats = export.zip_shared_volume(volume, handle, all_files=all_files)
        size = handle.tell()
        handle.seek(0)
    except BaseException:
        handle.close()
        raise
    return handle, size, f"{volume}{'-all' if all_files else ''}.zip", stats


def _stop(body: dict) -> dict:
    """Hard-stop one swarm run: every live sandbox tagged with it is terminated (``stop_swarm``).

    A dynamic swarm's controller in this process is told to halt first, so it creates nothing more,
    even mid-wave. Once it has finished its tick, a second sweep terminates anything it created
    meanwhile; the stop marker keeps it, and any later controller, from scheduling again.
    """

    swarm_id = str(body.get("swarm") or "")
    if not _SWARM_ID.fullmatch(swarm_id):
        raise AomError(f"invalid swarm id: {swarm_id!r}")
    with _controllers_lock:
        entry = _controllers.get(swarm_id)
    if entry:
        entry["controller"].request_halt()
    result = stop_swarm(swarm_id)
    _invalidate_agents(swarm_id)  # the next poll shows the stop
    if entry:
        entry["thread"].join(timeout=STOP_CONTROLLER_WAIT)
        entry["stop"].set()  # in case it is still mid-tick: it must not tick again
        late = stop_swarm(swarm_id)
        result["found"] += late["found"]
        result["stopped"] += late["stopped"]
        result["errors"] += late["errors"]
    return result


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
        default_timeout=DEFAULT_TIMEOUT, allowed_efforts=_model_efforts(m)
    )
    return {"text": text, "model": m.name, "harness": harness or "claude"}


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
    # The endpoint cap this spawn asks for: it deploys an undeployed model at it, or raises a live
    # one to it -- never lowers it or restarts the endpoint under anyone else using it.
    max_instances = _max_instances(body.get("max_instances") or SPAWN_MAX_INSTANCES)

    if mode == "dynamic":
        timeout = float(body["timeout"]) if body.get("timeout") else DYNAMIC_ORCHESTRATOR_TIMEOUT
        ds = DynamicSwarm.launch(
            harness, model, prompt,
            timeout=timeout, max_containers=max_instances, effort=effort, proxy=proxy,
        )
        _ensure_controller(ds, proxy=proxy, proxy_note=proxy_note)  # subagents get the note too
        return {"swarm": ds.id, "model": ds.model, "mode": "dynamic"}

    swarm = Swarm.launch(
        harness, model, prompt,
        count=int(body.get("count") or 1),
        max_containers=max_instances,
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

    def _send_zip(self, swarm_id: str, all_files: bool) -> None:
        handle, size, filename, stats = _zip_export(swarm_id, all_files)
        with handle:
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("X-Aom-Files", str(stats.get("files", "")))
                self.end_headers()
                shutil.copyfileobj(handle, self.wfile, 1 << 20)
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
                poffset = one("poffset", "")
                self._send_json(_events_payload(one("agent"), one("run"), int(one("offset", "0") or 0),
                                                one("swarm", os.environ.get("AOM_SWARM", "")),
                                                int(poffset) if poffset.isdigit() else None))
            elif parsed.path == "/api/files":
                self._send_json(_files_payload(one("swarm", os.environ.get("AOM_SWARM", "")), one("path", "/")))
            elif parsed.path == "/api/file":
                self._send_json(_file_payload(one("swarm", os.environ.get("AOM_SWARM", "")),
                                              one("path"), int(one("offset", "0") or 0)))
            elif parsed.path == "/api/zip":
                self._send_zip(one("swarm", os.environ.get("AOM_SWARM", "")),
                               one("all").lower() in ("1", "true", "yes"))
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
            elif parsed.path == "/api/stop":
                self._send_json(_stop(body))
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

    # On Windows SO_REUSEADDR lets a second server bind a port that is still in use, so a stale
    # dashboard and a new one would silently share :8787. Without it, a busy port fails to bind and
    # main() moves on to the next one.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request: object, client_address: object) -> None:  # type: ignore[override]
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def _bind(host: str, port: int, tries: int = 20) -> _QuietServer:
    """A server on ``port``, or on the next free port after it if that one is taken."""

    for candidate in range(port, port + tries):
        try:
            return _QuietServer((host, candidate), Handler)
        except OSError as error:
            last = error
    raise last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="aom swarm dashboard")
    parser.add_argument("--swarm", default=os.environ.get("AOM_SWARM", ""),
                        help="swarm id to open (default: $AOM_SWARM, else spawn one in the UI)")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT,
                        help="port to serve on; the next free one is used if it is taken")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args(argv)
    if args.swarm:
        os.environ["AOM_SWARM"] = args.swarm  # the handlers' default swarm (/api/config etc.)

    server = _bind(args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    if args.swarm:
        url += f"?swarm={quote(args.swarm)}"
    print(f"aom swarm dashboard on {url}"
          + ("" if args.swarm else "  (spawn a swarm in the UI, or pass --swarm <id>)"), flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
