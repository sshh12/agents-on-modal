# agents-on-modal: design

`agents-on-modal` (Python package `agents_on_modal`, CLI `aom`) runs agent CLIs ("harnesses")
inside sandboxes against open-weight models served on Modal GPUs. Three independent axes:

| axis | implementations | module |
|---|---|---|
| **model** | any entry of `agents_on_modal/catalog.json`, served by SGLang or llama.cpp on Modal | `agents_on_modal/models/` |
| **harness** | `claude` (Claude Code), `codex` (Codex CLI); more later | `agents_on_modal/harness/<name>.py` |
| **sandbox** | `local` (host, no container), `docker` (local Docker, internet but no LAN), `modal` (Modal Sandboxes) | `agents_on_modal/sandbox/<name>.py` |

All inference goes to Modal. Every harness talks to the model through the **gateway**, which runs
inside the model's GPU container next to the engine and speaks every harness protocol.

This is a clean rewrite; there is no backward compatibility with `codex_modal`. Old code is in git
history (`git show c4872ed:codex_modal/<file>`). Research notes behind every decision live in the
session scratchpad `research/*.md` (paths are given to implementers in their briefs).

---

## 1. Request path

```
harness CLI ──(Anthropic Messages | OpenAI Responses | chat)──▶ [sandbox model access]
   ──HTTPS + Modal proxy token──▶ Modal edge (*.modal.direct, proxy auth)
   ──▶ GPU container: gateway :8000 ──chat completions (stream)──▶ engine :8001 (SGLang | llama-server)
```

* **Gateway** (`agents_on_modal/gateway/`): aiohttp app. Exposes `POST /v1/messages`,
  `POST /v1/messages/count_tokens`, `POST /v1/responses`, `POST /v1/chat/completions` (pass-through
  with reasoning mapping), `GET /v1/models`, `GET /health`, `HEAD /`. Upstream is always the engine's
  streaming `/v1/chat/completions`. It owns: protocol translation (via one internal representation),
  reasoning-effort mapping per model, `max_tokens` clamping to the context window, keepalives during
  prefill/cold start (Anthropic `event: ping`; Responses `response.in_progress` events), error mapping
  (context overflow -> Claude `prompt is too long: N tokens > M maximum`, Codex
  `context_length_exceeded`), structured output (`output_config.format` / `text.format` ->
  `response_format` json_schema), Codex `custom` tools (single string arg) and `namespace` tools
  (flattened `ns__name`, restored on output). The request's `model` field is ignored; the served model
  is used. The gateway starts listening immediately at container start and holds requests (with
  keepalives) until the engine is healthy, so cold starts look like slow first tokens, not 503s.
  If the engine process dies the container exits so Modal replaces it.
* **Engines** (`agents_on_modal/models/engines/`): `sglang` (registry image, `python -m
  sglang.launch_server`) and `llamacpp` (CUDA devel image, llama-server built from a pinned commit).
  The engine binds `127.0.0.1:8001`; the wrapper owns host/port/model path/served name/context length/TP.

## 2. Models

`catalog.json` → `Model` dataclasses (`agents_on_modal/models/__init__.py`, already written). One
catalog entry fully describes a deployment; adding a model is a catalog edit.

* Each model is a **persistent Modal app** `aom-<name>` with a persistent weights Volume
  `aom-<name>-weights`. Deploy is idempotent: deploy the app, run the CPU `prepare` function (download
  missing/changed files, verify sizes, write manifest) before any GPU is requested, then return.
  Scale-to-zero (`min_containers=0`) is the cost control; there are no per-run ephemeral deployments,
  no watchdogs, no ownership records.
* **Scaling** (`Scaling`): `min_containers`, `max_containers`, `buffer_containers`,
  `scaledown_window`, `target_concurrency`. Defaults: one container max, scale to zero after 15 min.
  Overridable at deploy, and live via `Server.autoscaler.update(...)` (no redeploy) for fan-outs:
  `models.scale(model, min_containers=4, max_containers=32)`.
* **Endpoint**: `Endpoint(model, url, token)`; `url` = `https://<ws>--aom-<name>-server.<region>.modal.direct`.
  `models.endpoint(model)` attaches, `models.deploy(model)` deploys, `models.wait_ready(endpoint)` polls
  the gateway's `/health` (which is 200 only when the engine answers a real generation).
* **Reasoning** (`Reasoning`): harness efforts are normalized to `EFFORTS = (none, minimal, low,
  medium, high, xhigh, max)` (`ultra` -> `max`). Each model maps every normalized effort to an engine
  value: a string -> send `reasoning_effort=<value>` and `chat_template_kwargs[<kwarg>]=true`; `true` ->
  thinking on without an effort; `false` -> thinking off. `default` is the harness default effort.
* **Context**: `context_window` is passed to the engine (`--context-length`), the gateway, and the
  harness config. `max_output_tokens` bounds generation.
* **Sampling**: optional per-model defaults the gateway applies when the client sends none.

## 3. Harnesses

`Harness` ABC (`agents_on_modal/harness/base.py`, already written). A harness turns a `ModelAccess`
(what the model looks like from inside a sandbox) plus a home directory into files + env + argv,
and parses its own machine-readable output into normalized `Event`s.

Common policy for every harness:
* **Maximally permissive**: bypass all approvals, no harness-managed sandboxing; the sandbox is the
  boundary. Pre-seed every first-run dialog (onboarding, trust, bypass warning) so TUIs start clean.
* **One model everywhere**: main loop, subagents, background/aux tasks, compaction, goal evaluators all
  use the served model. Subagents enabled.
* **Context** set from the catalog so auto-compaction triggers correctly; **effort** default from the
  catalog, all normalized levels advertised (the gateway maps them).
* **Isolated home**: never touch the user's own `~/.claude` / `~/.codex`.
* **No nonessential traffic**: telemetry, update checks, marketplace, web-search server tools off.
* **Instructions**: the sandbox supplies one root instructions text; the harness writes it where that
  harness reads user-level instructions (Claude: `$CLAUDE_CONFIG_DIR/CLAUDE.md`; Codex:
  `$CODEX_HOME/AGENTS.md`). Never write into the user's workspace.
* **Skills**: sandbox-provided skills (`SKILL.md` dirs) are installed into the harness's user skill dir.
* **MCP**: sandbox-provided MCP servers (e.g. Playwright for the browser) are registered in the
  harness's config.
* **Long-running backends**: retry/timeout knobs sized for a cold-starting, autoscaling backend.
* **Goal mode**: `goal=True` runs the prompt as a long-horizon goal using the harness's native goal
  machinery (Claude `/goal <condition>`; Codex `create_goal`/`update_goal` tools + `exec resume`
  continuation). The in-sandbox runner keeps resuming until the harness reports the goal complete
  or the deadline passes, so an impossible goal runs (auto-compacting) until the timeout.

## 4. Sandboxes

`Sandbox` ABC (`agents_on_modal/sandbox/base.py`, already written). A sandbox = a place + its setup
(image, instructions, skills, MCP servers, network policy, optional upstream proxy).

| | local | docker | modal |
|---|---|---|---|
| where | host, `workdir` = given dir (default cwd) | container from the aom image | Modal Sandbox from the aom image |
| image | none | `agents_on_modal/sandbox/image/Dockerfile` | same Dockerfile via `Image.from_dockerfile`, published as a named image |
| user | current user | `agent` (uid 10001, passwordless sudo) | `agent` via `gosu` (Modal runs as root), `IS_SANDBOX=1` |
| network | host | internet only: agent on a Docker `--internal` network whose default route is an egress container that transparently proxies all TCP, serves DNS, refuses private/LAN/host destinations | Modal default (full internet) |
| model access | direct URL + user's proxy token | `http://<egress>:8081`; egress injects the token (agent never sees it) | direct URL + a run-scoped proxy token (created per swarm/sandbox, deleted at teardown) |
| upstream proxy | n/a | egress chains all agent TCP through it (transparent) | loopback relay `127.0.0.1:3128` (HTTP) / `:1080` (SOCKS5) + `*_PROXY` env + Chromium flags |
| batteries | host tools | dev/cyber toolchain, headless Chromium, Playwright MCP, agent-browser, gh, ... | same |

**Upstream proxy** (`proxy=` / `AOM_PROXY`): `socks5://`, `socks5h://`, `http://` with optional
`user:pass@`. `{session}` in the URL is replaced with a per-sandbox id (sticky residential sessions).
Model inference never goes through it. The Modal relay and the Docker egress share
`agents_on_modal/sandbox/net/` (upstream connector, SOCKS5/HTTP servers, destination policy).

**Runs**: `sandbox.start(harness, model_access?, prompt, goal=False, timeout=None) -> Run` starts the
in-sandbox runner (`python -m agents_on_modal.runner <run_dir>`) detached. The runner launches the
harness headless, tees its JSON stream into `<run_dir>/trace.jsonl` (envelope per line: `{"ts",
"src": "harness"|"runner", "data"}`), drives goal continuation, enforces the deadline, and writes
`<run_dir>/status.json`. Everything a reader needs is on disk inside the sandbox, so any process can
rediscover sandboxes (`Sandbox.list()`), reattach to runs (`sandbox.runs()`), and tail traces
(`run.events(follow=True)`) after a restart. Interactive sessions (`sandbox.attach(harness)`) exec the
harness TUI on the caller's terminal.

**Swarms** (`agents_on_modal/swarm.py`): N Modal sandboxes running the same harness/model/prompt,
tagged `aom-swarm=<id>`, sharing `/shared`. Shared dir = a per-swarm **hub** sandbox (small HTTP file
service, persisted to a Volume v2) plus a sync daemon in every agent sandbox that mirrors `/shared`
both ways (last-writer-wins), because Modal Volumes do not show other containers' writes without
reloads. Traces are mirrored to the swarm Volume so they survive sandbox exit. Launch is bounded
(semaphore + retries + idempotent sandbox names) and pre-warms the model (`models.scale`).

## 5. Layout

```
agents_on_modal/
  __init__.py  __main__.py  cli.py  config.py  traces.py  runner.py  swarm.py  catalog.json
  models/      __init__.py (catalog)  deploy.py  serve.py (Modal app module)  engines/{__init__,base,sglang,llamacpp}.py
  gateway/     __init__.py  server.py  ir.py  anthropic.py  responses.py  chat.py  reasoning.py
  harness/     __init__.py (registry)  base.py  claude.py  codex.py
  sandbox/     __init__.py (registry)  base.py  local.py  docker.py  modal.py  instructions.py
               net/{__init__,proxy,egress,relay,dns}.py   share/{__init__,hub,sync}.py
               image/{Dockerfile, init.sh, skills/...}
examples/      local_session.py  docker_session.py  swarm/{server.py,index.html}
tests/         unit tests (no network), tests/fakes/ (fake engine), tests/e2e/ (Modal credits; opt-in)
```

State lives in `~/.agents-on-modal/` (override `AOM_HOME`): `token.json` (Modal proxy token, 0600),
`local/<sandbox-id>/` (local sandbox homes + runs), `docker-runs/<id>/` (exported logs).

## 6. Conventions

* Python >= 3.10, type hints, dataclasses, no new runtime deps beyond `modal` and `aiohttp`.
* Library code raises exceptions (`AomError` subclasses); only `cli.py` prints/exits.
* Windows hosts are first-class for local/docker (paths, `subprocess` flags, no `fork`).
* Files shipped into Linux containers use LF line endings (`.gitattributes`).
* Unit tests: `python -m pytest tests -q` must pass offline. E2E tests: `tests/e2e/`, opt-in via
  `AOM_E2E=1`, spend Modal credits.
