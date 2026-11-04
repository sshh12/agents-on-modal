"""Dynamic swarms: an orchestrator agent that spawns *other* agents through a shared-volume CLI.

A naive :class:`~agents_on_modal.swarm.Swarm` fans out N identical agents. A *dynamic* swarm runs
one **orchestrator** agent (claude/codex in a Modal sandbox) that, given a natural-language prompt,
researches as needed and then drives spawning of **subagents** -- in ordered *waves* (stages) and/or
bounded-concurrency *pools* -- by talking to the ``aom-swarm`` CLI (:mod:`agents_on_modal.swarmctl`).

Everything flows over the shared Modal Volume mounted at ``/shared`` in every agent. A single
reserved namespace, ``/shared/_swarm/``, is the control plane:

    _swarm/meta.json              {swarm_id, model, harness, orchestrator_id, default_timeout}
    _swarm/launch.json            launch settings + the raw task, for tracking and reproduction
    _swarm/orchestrator_prompt.md the exact prompt the orchestrator got (instructions + task)
    _swarm/requests/<reqid>.json  CLI -> controller commands   {id, ts, verb, payload}
    _swarm/ack/<reqid>.json       controller -> CLI replies    {id, ok, errors, warnings, result}
    _swarm/workflows/<name>.json  controller-published plan + live stage/agent state
    _swarm/agents/<agent_id>.json controller-published per-agent status
    _swarm/traces/<agent_id>.jsonl durable per-agent trace (parsed events; survives teardown)
    _swarm/outputs/<agent_id>.json structured outputs for agents given a `schema` (soft contract)
    _swarm/now.json               the controller's clock {ts, utc} (sandbox clocks can drift)

**Nothing else under ``/shared`` is a lib primitive.** How subagents coordinate -- a message board,
per-task report files, directory layout -- is entirely up to what the orchestrator writes into the
subagent prompts. The injected instructions only *suggest* patterns.

The CLI (in a sandbox) and the controller (local, the only actor with Modal credentials) never touch
each other directly; they exchange files through the volume. Both sides go through a small
:class:`ControlPlane` so the whole protocol is testable offline against a plain directory.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, field
from typing import Any

from .config import AomError
from .harness import HARNESSES
from .models import EFFORTS, normalize_effort

#: Every subagent needs an explicit deadline (Modal sandboxes always take a timeout); this is the
#: fallback when a plan omits one. One hour is enough for a single /goal pass without stranding GPUs.
DEFAULT_TIMEOUT = 3600.0

#: The one reserved directory under ``/shared``. Everything else there is free-form.
RESERVED = "_swarm"
#: Written once at launch under RESERVED: the orchestrator's exact prompt, and the launch settings.
ORCHESTRATOR_PROMPT_FILE = "orchestrator_prompt.md"
LAUNCH_FILE = "launch.json"
#: Written under RESERVED by a hard stop (``swarm.stop_swarm``). Every controller of the swarm -- in
#: any process, now or after a restart -- halts on seeing it and never schedules another agent.
STOPPED_FILE = "stopped.json"
#: Written under RESERVED by the controller: its own clock (``{ts, utc}``), the trusted time source.
NOW_FILE = "now.json"

TERMINAL_STATES = ("done", "timeout", "failed", "cancelled")

#: Subagent sandboxes the controller creates / polls at once. A create+start is ~10 s of Modal round
#: trips and a refresh a few more, so one at a time made a 100-agent wave take ~20 min to fill (each
#: agent's deadline ticking from its own late start) and stalled the rest of the tick behind it --
#: acks, workflow state, the orchestrator's trace tail.
CONTROLLER_CONCURRENCY = 16

#: How often (seconds) the controller reloads a sandbox's ``/shared`` mount so it sees what others
#: wrote (a Modal sandbox's volume mount never shows outside writes until reloaded). The
#: orchestrator needs it most -- its acks and workflow state live there -- so it is reloaded far
#: more often. A reload briefly disturbs file access in the sandbox, so subagents, who mostly write
#: their own files, get it rarely.
ORCHESTRATOR_RELOAD_INTERVAL = 10.0
SUBAGENT_RELOAD_INTERVAL = 120.0


class PlanError(AomError):
    """A dynamic-swarm plan failed validation."""


# --------------------------------------------------------------------------------------------- #
# Plan model + validation (pure; shared by the CLI's client-side lint and the controller)        #
# --------------------------------------------------------------------------------------------- #
@dataclass
class AgentSpec:
    """One subagent to spawn."""

    prompt: str
    goal: bool = False
    timeout: float = DEFAULT_TIMEOUT
    effort: str | None = None
    schema: dict[str, Any] | None = None
    #: The agent CLI to run it under; ``None`` -> the swarm's (the orchestrator's) harness.
    harness: str | None = None
    #: One canonical lane token and the absolute report directory built from it. Carried as fields
    #: (not only inside the prompt) so `status` and a later merge never have to guess the tree.
    lane: str | None = None
    report_dir: str | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"prompt": self.prompt, "goal": self.goal, "timeout": self.timeout}
        if self.effort:
            data["effort"] = self.effort
        if self.schema is not None:
            data["schema"] = self.schema
        if self.harness:
            data["harness"] = self.harness
        if self.lane:
            data["lane"] = self.lane
        if self.report_dir:
            data["report_dir"] = self.report_dir
        return data


@dataclass
class Stage:
    """A group of agents run together: up to ``max_concurrent`` at once (a pool); the whole stage
    must finish before the next stage starts (so ordered stages are waves)."""

    name: str
    agents: list[AgentSpec]
    max_concurrent: int = 0  # 0 -> all at once (set in __post_init__)

    def __post_init__(self) -> None:
        if not self.max_concurrent or self.max_concurrent < 1:
            self.max_concurrent = max(1, len(self.agents))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "agents": [a.to_json() for a in self.agents],
        }


@dataclass
class Plan:
    """An ordered list of stages. ``default_timeout`` fills any agent that omits one."""

    stages: list[Stage]

    def to_json(self) -> dict[str, Any]:
        return {"stages": [s.to_json() for s in self.stages]}

    @classmethod
    def from_dict(cls, data: Any, *, default_timeout: float = DEFAULT_TIMEOUT) -> "Plan":
        """Parse + validate, raising :class:`PlanError` (joined messages) on any error."""

        plan, errors, _ = parse_plan(data, default_timeout=default_timeout)
        if errors:
            raise PlanError("invalid plan:\n  - " + "\n  - ".join(errors))
        assert plan is not None
        return plan


#: "append ... to /shared/board.md": in a 100-agent run the task text won over the instructions, and
#: 62 agents' lines were lost. Flagged at submit time (a warning; the plan still runs).
_SHARED_APPEND = re.compile(r"\bappend\b[^\n]{0,80}?/shared/(?!_swarm/)[\w./-]+\.\w+", re.I)


def _parse_agent(
    data: Any, where: str, errors: list[str], warnings: list[str], default_timeout: float
) -> AgentSpec | None:
    if not isinstance(data, dict):
        errors.append(f"{where}: must be an object")
        return None
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append(f"{where}: 'prompt' is required and must be a non-empty string")
        prompt = prompt if isinstance(prompt, str) else ""

    goal = bool(data.get("goal", False))

    raw_timeout = data.get("timeout", default_timeout)
    timeout = default_timeout
    if raw_timeout is None:
        timeout = default_timeout
    else:
        try:
            timeout = float(raw_timeout)
            if timeout <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{where}: 'timeout' must be a positive number of seconds")

    effort = data.get("effort")
    if effort is not None:
        norm = normalize_effort(effort)
        if norm is None:
            warnings.append(
                f"{where}: unknown effort {effort!r}; ignoring (known: {', '.join(EFFORTS)})"
            )
            effort = None
        else:
            effort = norm  # the model's catalog maps it to the nearest supported level downstream

    schema = data.get("schema")
    if schema is not None and not isinstance(schema, dict):
        errors.append(f"{where}: 'schema' must be a JSON object (inline, or a path to a JSON file "
                      "that `aom-swarm submit` can read)")
        schema = None

    if _SHARED_APPEND.search(prompt):
        warnings.append(
            f"{where}: the prompt has agents append to a shared /shared file -- concurrent writers "
            "overwrite each other (last writer wins); give each agent its own file and merge them"
        )

    harness = data.get("harness")
    if harness is not None and harness not in HARNESSES:
        errors.append(f"{where}: unknown harness {harness!r} (known: {', '.join(HARNESSES)})")
        harness = None

    lane = data.get("lane")
    lane = str(lane).strip() if lane else None
    if lane and ("/" in lane or any(c.isspace() for c in lane)):
        errors.append(f"{where}: 'lane' must be one token (no '/' or whitespace), got {lane!r}")
    report_dir = data.get("report_dir")
    if report_dir is not None:
        report_dir = str(report_dir).rstrip("/")
        parts = report_dir.split("/")
        if not report_dir.startswith("/shared/") or ".." in parts:
            errors.append(f"{where}: 'report_dir' must be an absolute path under /shared/, "
                          f"got {report_dir!r}")
        elif len(parts) > 2 and parts[2] == RESERVED:
            errors.append(f"{where}: 'report_dir' must not be under /shared/{RESERVED}/ "
                          "(the control plane)")
        elif lane and lane not in report_dir:
            warnings.append(
                f"{where}: lane {lane!r} does not appear in report_dir {report_dir!r}; a merge "
                "that globs lane names will miss it"
            )

    for key in data:
        if key not in ("prompt", "goal", "timeout", "effort", "schema", "harness", "lane",
                       "report_dir"):
            warnings.append(f"{where}: unknown field {key!r} ignored")

    return AgentSpec(prompt=prompt, goal=goal, timeout=timeout, effort=effort, schema=schema,
                     harness=harness, lane=lane, report_dir=report_dir)


def parse_plan(
    data: Any, *, default_timeout: float = DEFAULT_TIMEOUT
) -> tuple[Plan | None, list[str], list[str]]:
    """Return ``(plan_or_None, errors, warnings)``. ``plan`` is ``None`` iff there are errors.

    This is the single source of truth for both the CLI's instant client-side lint and the
    controller's server-side check, so the orchestrator gets identical feedback either way.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return None, ["plan must be a JSON object with a 'stages' array"], warnings
    raw_stages = data.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        return None, ["plan needs a non-empty 'stages' array"], warnings

    seen_names: set[str] = set()
    stages: list[Stage] = []
    for si, raw in enumerate(raw_stages):
        where = f"stages[{si}]"
        if not isinstance(raw, dict):
            errors.append(f"{where}: must be an object")
            continue
        name = raw.get("name") or f"stage{si + 1}"
        name = str(name)
        if name in seen_names:
            errors.append(f"{where}: duplicate stage name {name!r}")
        seen_names.add(name)

        raw_agents = raw.get("agents")
        if not isinstance(raw_agents, list) or not raw_agents:
            errors.append(f"{where} ({name}): needs a non-empty 'agents' array")
            raw_agents = []
        agents: list[AgentSpec] = []
        for ai, raw_agent in enumerate(raw_agents):
            spec = _parse_agent(
                raw_agent, f"{where}.agents[{ai}]", errors, warnings, default_timeout
            )
            if spec is not None:
                agents.append(spec)

        max_concurrent = raw.get("max_concurrent", 0)
        try:
            max_concurrent = int(max_concurrent or 0)
            if max_concurrent < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{where} ({name}): 'max_concurrent' must be a non-negative integer")
            max_concurrent = 0

        if agents:
            stages.append(Stage(name=name, agents=agents, max_concurrent=max_concurrent))

    if errors:
        return None, _collapse_repeats(errors), _collapse_repeats(warnings)
    return Plan(stages=stages), errors, _collapse_repeats(warnings)


_PER_AGENT = re.compile(r"(?P<stage>stages\[\d+\]\.agents)\[(?P<idx>\d+)\]: (?P<msg>.*)", re.S)


def _collapse_repeats(messages: list[str]) -> list[str]:
    """Fold a per-agent message repeated across a stage into one line, in first-seen order:
    ``stages[0].agents[0-111] (112 agents): ...``. One shared mistake in a 112-agent wave
    otherwise prints 112 identical lines, and a caller's ``tail`` hides where they start."""

    groups: dict[tuple[str, str], list[int]] = {}
    order: list[Any] = []
    for message in messages:
        hit = _PER_AGENT.fullmatch(message)
        if not hit:
            order.append(message)
            continue
        key = (hit["stage"], hit["msg"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(hit["idx"]))
    out = []
    for item in order:
        if isinstance(item, str):
            out.append(item)
            continue
        (stage, msg), idxs = item, groups[item]
        count = f" ({len(idxs)} agents)" if len(idxs) > 1 else ""
        out.append(f"{stage}[{_index_ranges(idxs)}]{count}: {msg}")
    return out


def _index_ranges(idxs: list[int]) -> str:
    """``[0, 1, 2, 5, 7, 8]`` -> ``"0-2,5,7-8"``."""

    spans: list[list[int]] = []
    for i in sorted(set(idxs)):
        if spans and i == spans[-1][1] + 1:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


# --------------------------------------------------------------------------------------------- #
# Pure scheduler (no Modal): pool + wave semantics, fully unit-testable                          #
# --------------------------------------------------------------------------------------------- #
def stage_next_indices(stage: Stage, started: set[int], finished: set[int]) -> list[int]:
    """Agent indices to start now in ``stage``: fill up to ``max_concurrent`` running slots, in
    order, among those not yet started."""

    running = len(started - finished)
    free = max(0, stage.max_concurrent - running)
    if free <= 0:
        return []
    pending = [i for i in range(len(stage.agents)) if i not in started]
    return pending[:free]


def stage_complete(stage: Stage, finished: set[int]) -> bool:
    return len(finished & set(range(len(stage.agents)))) >= len(stage.agents)


def active_stage_index(plan: Plan, finished_by_stage: list[set[int]]) -> int | None:
    """Index of the first not-yet-complete stage (stages run strictly in order), or ``None`` when
    every stage is complete."""

    for si, stage in enumerate(plan.stages):
        if not stage_complete(stage, finished_by_stage[si]):
            return si
    return None


# --------------------------------------------------------------------------------------------- #
# Control plane: a tiny file store shared by the CLI (sandbox fs) and controller (Modal Volume)  #
# --------------------------------------------------------------------------------------------- #
class ControlPlane(ABC):
    """JSON/text files under a base dir. The CLI uses a plain-filesystem impl pointed at
    ``/shared/_swarm`` (inside the sandbox); the controller uses the Modal Volume API from
    outside. Tests point both at one temp directory."""

    @abstractmethod
    def put_json(self, relpath: str, obj: Any) -> None: ...

    @abstractmethod
    def get_json(self, relpath: str) -> Any | None: ...

    @abstractmethod
    def put_text(self, relpath: str, text: str) -> None: ...

    @abstractmethod
    def read_text(self, relpath: str) -> str | None: ...

    @abstractmethod
    def listdir(self, reldir: str) -> list[str]: ...

    def exists(self, relpath: str) -> bool:
        return self.read_text(relpath) is not None

    def read_shared_json(self, abspath: str) -> Any | None:
        """JSON at an absolute in-sandbox path under ``/shared`` (e.g. a lane's ``result.json``),
        for a control plane rooted at the shared volume -- the controller's, whose paths all start
        with ``_swarm/``. ``None`` for anything outside ``/shared`` or not parseable."""

        if not abspath.startswith("/shared/"):
            return None
        rel = abspath[len("/shared/"):].strip("/")
        if not rel or ".." in rel.split("/"):
            return None
        return self.get_json(rel)


class LocalDirControlPlane(ControlPlane):
    """Plain filesystem. Used in-sandbox (base ``/shared/_swarm``) and in tests. Stdlib only."""

    def __init__(self, base_dir: str) -> None:
        from pathlib import Path

        self.base = Path(base_dir)

    def _path(self, relpath: str) -> Any:
        return self.base.joinpath(*relpath.split("/"))

    def put_json(self, relpath: str, obj: Any) -> None:
        self.put_text(relpath, json.dumps(obj, indent=2, default=str))

    def get_json(self, relpath: str) -> Any | None:
        text = self.read_text(relpath)
        if text is None:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None

    def put_text(self, relpath: str, text: str) -> None:
        path = self._path(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)  # atomic on same fs; readers never see a half-written file

    def read_text(self, relpath: str) -> str | None:
        try:
            return self._path(relpath).read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            return None

    def listdir(self, reldir: str) -> list[str]:
        path = self._path(reldir)
        try:
            return sorted(p.name for p in path.iterdir() if p.is_file())
        except (FileNotFoundError, NotADirectoryError):
            return []


class VolumeControlPlane(ControlPlane):
    """Reads/writes the control plane on a Modal Volume v2 from *outside* Modal (the controller).
    Runtime only: imports ``modal`` lazily. Never exercised by the offline tests."""

    def __init__(self, volume_name: str, *, create: bool = True) -> None:
        self.volume_name = volume_name
        #: False: never create the volume (reads of a missing one come back empty, writes raise).
        self.create = create
        self._vol: Any = None

    def _volume(self) -> Any:
        if self._vol is None:
            import modal

            self._vol = modal.Volume.from_name(self.volume_name, version=2,
                                               create_if_missing=self.create)
        return self._vol

    def put_text(self, relpath: str, text: str) -> None:
        self._put_bytes(relpath, text.encode("utf-8"))

    def put_json(self, relpath: str, obj: Any) -> None:
        self._put_bytes(relpath, json.dumps(obj, indent=2, default=str).encode("utf-8"))

    def _put_bytes(self, relpath: str, data: bytes) -> None:
        vol = self._volume()
        with vol.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(data), "/" + relpath.lstrip("/"))

    def read_text(self, relpath: str) -> str | None:
        try:
            return b"".join(self._volume().read_file("/" + relpath.lstrip("/"))).decode(
                "utf-8", "replace"
            )
        except FileNotFoundError:
            return None
        except Exception:  # noqa: BLE001 - treat any read failure as "not there yet"
            return None

    def get_json(self, relpath: str) -> Any | None:
        text = self.read_text(relpath)
        if text is None:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None

    def listdir(self, reldir: str) -> list[str]:
        try:
            entries = self._volume().listdir("/" + reldir.lstrip("/"))
        except Exception:  # noqa: BLE001 - missing dir / not committed yet
            return []
        names = []
        for entry in entries:
            path = getattr(entry, "path", str(entry))
            names.append(path.rsplit("/", 1)[-1])
        return sorted(names)


# --------------------------------------------------------------------------------------------- #
# Schema soft-contract                                                                           #
# --------------------------------------------------------------------------------------------- #
_SCHEMA_INSTRUCTION = (
    "\n\n---\nREQUIRED OUTPUT: as soon as you have the result -- before any final review, summary "
    "or status bookkeeping -- write it as a single JSON document that conforms to this JSON schema "
    "to the file `{path}` (create parent directories if needed).{copy} It is part of the task, not "
    "an afterthought: an agent that runs out of time before writing it has not delivered.\n"
    "```json\n{schema}\n```\n"
)
#: Appended to the schema instruction when the agent has a ``report_dir``: the copy keeps the result
#: beside its evidence, and the controller falls back to it when the primary path is missing.
_SCHEMA_COPY = " Then write the identical JSON to `{copy_path}` as well."
#: The lane-tree copy's file name under ``report_dir``.
RESULT_FILE = "result.json"


def validate_against_schema(value: Any, schema: dict[str, Any]) -> list[str]:
    """A deliberately small JSON-Schema check (type / required / properties / items). Enough to
    tell the orchestrator whether a subagent produced the shape it asked for, not a full validator."""

    def check(val: Any, sch: Any, where: str) -> list[str]:
        if not isinstance(sch, dict):
            return []
        problems: list[str] = []
        typ = sch.get("type")
        type_ok = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }
        if isinstance(typ, str) and typ in type_ok:
            if typ == "number" and isinstance(val, bool):
                problems.append(f"{where}: expected number, got boolean")
            elif not isinstance(val, type_ok[typ]):
                problems.append(f"{where}: expected {typ}, got {type(val).__name__}")
        if typ == "object" and isinstance(val, dict):
            for req in sch.get("required", []) or []:
                if req not in val:
                    problems.append(f"{where}: missing required property {req!r}")
            for key, subschema in (sch.get("properties") or {}).items():
                if key in val:
                    problems += check(val[key], subschema, f"{where}.{key}")
        if typ == "array" and isinstance(val, list) and isinstance(sch.get("items"), dict):
            for i, item in enumerate(val):
                problems += check(item, sch["items"], f"{where}[{i}]")
        return problems

    return check(value, schema, "$")


# --------------------------------------------------------------------------------------------- #
# Controller: the only actor that touches Modal. Ingests requests, runs stages, publishes state. #
# --------------------------------------------------------------------------------------------- #
@dataclass
class _AgentRun:
    idx: int
    stage: str
    workflow: str
    spec: AgentSpec
    box: Any
    run: Any
    state: str = "starting"
    schema_ok: bool | None = None
    offset: int = 0
    error: str | None = None
    last_reload: float = 0.0
    #: When a status read last succeeded, and the consecutive failures since. A terminal status.json
    #: is the LAST thing an exiting run can be read at; after that every read fails, and leaving the
    #: record "running" stranded whole stages behind `wait --all`.
    last_status_ok: float = field(default_factory=time.time)
    refresh_errors: int = 0
    #: Terminated by a `stop`: stays "cancelled" even if an in-flight refresh (requests are now
    #: ingested while refreshes run) reads the pre-stop "running" afterwards.
    stopped: bool = False
    #: Terminal and fully published, so later ticks skip it. Set on the refresh AFTER the first one
    #: that read a terminal state, so trace lines the runner writes after flipping status still land.
    settled: bool = False
    _seen_terminal: bool = False

    @property
    def id(self) -> str:
        return self.box.id

    @property
    def run_id(self) -> str | None:
        return getattr(self.run, "run_id", None)


@dataclass
class _WorkflowRuntime:
    name: str
    plan: Plan
    paused: bool = False
    cancelled: bool = False
    created_at: float = field(default_factory=time.time)
    # stage index -> {agent idx -> _AgentRun}
    runs: dict[int, dict[int, _AgentRun]] = field(default_factory=dict)


class Controller:
    """Drives a dynamic swarm. Call :meth:`tick` in a loop (or :meth:`run_forever`).

    ``sandboxes`` defaults to :mod:`agents_on_modal.sandbox`; tests pass a fake exposing
    ``create(name, **opts)``, ``get(name, id)`` and ``list_sandboxes(name, **filters)``.
    """

    def __init__(
        self,
        control: ControlPlane,
        swarm_id: str,
        endpoint: Any,
        shared_volume: str,
        harness: str,
        *,
        proxy: str | None = None,
        proxy_note: str | None = None,
        orchestrator_id: str | None = None,
        orchestrator_run_id: str | None = None,
        default_timeout: float = DEFAULT_TIMEOUT,
        sandboxes: Any = None,
    ) -> None:
        self.control = control
        self.swarm_id = swarm_id
        self.endpoint = endpoint
        self.shared_volume = shared_volume
        self.harness = harness
        self.proxy = proxy
        #: 'agent opt-in' proxy note prepended to EVERY subagent's prompt (so fan-out agents can
        #: route through the residential proxy only when needed). None unless opt-in mode is set.
        self.proxy_note = proxy_note
        self.orchestrator_id = orchestrator_id
        self.orchestrator_run_id = orchestrator_run_id
        self.default_timeout = default_timeout
        if sandboxes is None:
            from . import sandbox as sandboxes  # noqa: PLC0415 - lazy so swarmctl stays import-light
        self._sb = sandboxes
        self._workflows: dict[str, _WorkflowRuntime] = {}
        self._wrote_meta = False
        #: Request ids already dispatched -- in memory plus a durable marker -- so a request is
        #: executed at most once even across a controller restart (issue S5).
        self._handled: set[str] = set()
        # Orchestrator trace tailing (its sandbox is not a subagent; tail it so its trace persists
        # to the volume too and survives teardown).
        self._orch_run: Any = None
        self._orch_box: Any = None
        self._orch_reloaded = 0.0
        self._orch_offset = 0
        #: Orchestrator harness session id, discovered from its trace and published to meta.json so a
        #: resume does not have to scrape the trace (issue S9) -- also what resume_orchestrator reads.
        self._orch_session: str | None = None
        #: Opt-in orchestrator supervision bookkeeping (issue S3); see _supervise_orchestrator.
        self._orch_restarts = 0
        self._last_restart_at = 0.0
        #: Hard stop: requested from another thread (``request_halt``) or seen as STOPPED_FILE on
        #: the control plane; ``halted`` once this controller's agents are terminated (``_halt``).
        self._halt_requested = False
        self.halted = False
        self._now_published = 0.0

    # -- lifecycle ---------------------------------------------------------------------------- #
    def run_forever(self, interval: float = 3.0, stop: Any = None) -> None:
        while True:
            if stop is not None and getattr(stop, "is_set", lambda: False)():
                return
            try:
                self.tick()
            except Exception as error:  # noqa: BLE001 - a controller must not die on one bad tick
                self.control.put_json(
                    f"{RESERVED}/controller_error.json", {"ts": time.time(), "error": str(error)}
                )
            if self.halted:
                return
            time.sleep(interval)

    def request_halt(self) -> None:
        """Hard-stop this controller from another thread: no agent is created from now on (even mid
        wave), and its next tick terminates every agent it started, then ``run_forever`` returns."""

        self._halt_requested = True

    def tick(self) -> None:
        """One pass: write meta, ingest new requests, advance workflows, publish state."""

        if self._halt_requested or self.control.exists(f"{RESERVED}/{STOPPED_FILE}"):
            self._halt()
            return

        if not self._wrote_meta:
            self.control.put_json(
                f"{RESERVED}/meta.json",
                {
                    "swarm_id": self.swarm_id,
                    "model": getattr(self.endpoint.model, "name", ""),
                    "harness": self.harness,
                    "orchestrator_id": self.orchestrator_id,
                    "orchestrator_run_id": self.orchestrator_run_id,
                    "default_timeout": self.default_timeout,
                    "created_at": time.time(),
                },
            )
            self._wrote_meta = True

        # Sandbox VM clocks can be hours off from the host. Publish the controller's clock so agents
        # stamp campaign files (and `aom-swarm status` ages records) from one trusted source. Only as
        # often as the orchestrator's mount reloads -- nobody can see a fresher one.
        if time.time() - self._now_published >= ORCHESTRATOR_RELOAD_INTERVAL:
            self._now_published = time.time()
            self.control.put_json(f"{RESERVED}/{NOW_FILE}", {
                "ts": self._now_published,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._now_published)),
            })

        self._ingest_requests()

        for wf in list(self._workflows.values()):
            self._advance(wf)
            self._publish_workflow(wf)

        self._reload_orchestrator()  # so its `aom-swarm` sees the acks + state written (throttled)
        self._tail_orchestrator()
        self._supervise_orchestrator()

    # -- request handling --------------------------------------------------------------------- #
    def _ingest_requests(self) -> bool:
        """Dispatch every new request and deliver the acks. Runs at the top of each tick AND while a
        wave's creates/refreshes are in flight (see :meth:`_map`): a request that landed mid-tick
        used to wait out the whole 100-agent tick, so acks took 90-105 s and every retry of a
        "timed out" submit double-ran the wave."""

        acked = False
        for name in self.control.listdir(f"{RESERVED}/requests"):
            if not name.endswith(".json"):
                continue
            reqid = name[: -len(".json")]
            # Dedupe: run each request id at most once, even across a controller restart (issue S5).
            # A durable 'handled' marker is written at DISPATCH (before any work), so a crash mid-
            # handling never resurfaces the request as a duplicate workflow ('spawn-2').
            if (reqid in self._handled
                    or self.control.exists(f"{RESERVED}/ack/{reqid}.json")
                    or self.control.exists(f"{RESERVED}/handled/{reqid}.json")):
                self._handled.add(reqid)
                continue
            request = self.control.get_json(f"{RESERVED}/requests/{name}")
            if isinstance(request, dict):
                self._claim(reqid, request.get("verb"))
                self._handle_request(reqid, request)
                acked = True

        if acked:
            # Deliver acks NOW: the orchestrator's mount only shows them after a reload.
            self._reload_orchestrator(force=True)
        return acked

    def _ingest_between(self) -> None:
        """:meth:`_ingest_requests` for :meth:`_map`'s in-flight callback: never raises."""

        try:
            self._ingest_requests()
        except Exception as error:  # noqa: BLE001 - the tick's own ingest records real failures
            try:
                self.control.put_json(f"{RESERVED}/controller_error.json",
                                      {"ts": time.time(), "error": f"mid-tick ingest: {error}"})
            except Exception:  # noqa: BLE001
                pass

    def _ack(self, reqid: str, *, ok: bool, errors=None, warnings=None, result=None) -> None:
        self.control.put_json(
            f"{RESERVED}/ack/{reqid}.json",
            {
                "id": reqid,
                "ts": time.time(),
                "ok": ok,
                "errors": errors or [],
                "warnings": warnings or [],
                "result": result or {},
            },
        )

    def _claim(self, reqid: str, verb: Any) -> None:
        """Mark a request handled at dispatch (before any work) so it is never executed twice (S5)."""

        self._handled.add(reqid)
        self.control.put_json(
            f"{RESERVED}/handled/{reqid}.json",
            {"id": reqid, "verb": verb, "ts": time.time(), "pid": os.getpid()},
        )

    def _handle_request(self, reqid: str, request: dict[str, Any]) -> None:
        verb = request.get("verb")
        payload = request.get("payload") or {}
        try:
            if verb == "submit":
                self._do_submit(reqid, payload)
            elif verb == "spawn":
                self._do_spawn(reqid, payload)
            elif verb == "stop":
                self._do_stop(reqid, payload)
            elif verb == "resume":
                self._do_resume(reqid, payload)
            else:
                self._ack(reqid, ok=False, errors=[f"unknown verb {verb!r}"])
        except Exception as error:  # noqa: BLE001 - never let one request kill the controller
            self._ack(reqid, ok=False, errors=[f"{type(error).__name__}: {error}"])

    def _unique_name(self, base: str) -> str:
        name = base or "workflow"
        if name not in self._workflows:
            return name
        i = 2
        while f"{name}-{i}" in self._workflows:
            i += 1
        return f"{name}-{i}"

    def _do_submit(self, reqid: str, payload: dict[str, Any]) -> None:
        plan, errors, warnings = parse_plan(payload, default_timeout=self.default_timeout)
        if errors:
            self._ack(reqid, ok=False, errors=errors, warnings=warnings)
            return
        assert plan is not None
        requested = str(payload.get("name") or reqid)
        name = self._unique_name(requested)
        if name != requested:
            warnings.append(f"workflow name {requested!r} was already used; this one is {name!r}")
        self._workflows[name] = _WorkflowRuntime(name=name, plan=plan)
        self._ack(
            reqid,
            ok=True,
            warnings=warnings,
            result={"workflow": name, "stages": len(plan.stages),
                    "agents": sum(len(s.agents) for s in plan.stages)},
        )

    def _do_spawn(self, reqid: str, payload: dict[str, Any]) -> None:
        # A one-off spawn is just a one-stage, one-agent workflow.
        agent = payload.get("agent", payload)
        plan, errors, warnings = parse_plan(
            {"stages": [{"name": "spawn", "agents": [agent]}]}, default_timeout=self.default_timeout
        )
        if errors:
            self._ack(reqid, ok=False, errors=errors, warnings=warnings)
            return
        assert plan is not None
        explicit = payload.get("workflow")
        requested = str(explicit or "spawn")
        name = self._unique_name(requested)
        if explicit and name != requested:
            warnings.append(f"workflow name {requested!r} was already used; this one is {name!r}")
        self._workflows[name] = _WorkflowRuntime(name=name, plan=plan)
        self._ack(reqid, ok=True, warnings=warnings, result={"workflow": name})

    def _do_stop(self, reqid: str, payload: dict[str, Any]) -> None:
        target = str(payload.get("target") or "")
        if target in self._workflows:
            wf = self._workflows[target]
            wf.paused = True  # stop scheduling new agents; running ones keep going
            self._ack(reqid, ok=True, result={"workflow": target, "status": "paused"})
            return
        # otherwise treat it as an agent id to terminate
        for wf in self._workflows.values():
            for stage_runs in wf.runs.values():
                for ar in stage_runs.values():
                    if ar.id == target:
                        if ar.state in TERMINAL_STATES:
                            # Keep its real outcome: a stop must not rewrite "done" as "cancelled".
                            self._ack(reqid, ok=True, result={"agent": target, "status": ar.state},
                                      warnings=[f"agent {target} had already finished ({ar.state})"])
                            return
                        self._terminate(ar)
                        self._ack(reqid, ok=True, result={"agent": target, "status": "stopped"})
                        return
        self._ack(reqid, ok=False, errors=[f"no workflow or agent named {target!r}"])

    def _do_resume(self, reqid: str, payload: dict[str, Any]) -> None:
        name = str(payload.get("workflow") or "")
        wf = self._workflows.get(name)
        if wf is None:
            self._ack(reqid, ok=False, errors=[f"no workflow named {name!r}"])
            return
        wf.paused = False
        self._ack(reqid, ok=True, result={"workflow": name, "status": "running"})

    # -- scheduling --------------------------------------------------------------------------- #
    def _advance(self, wf: _WorkflowRuntime) -> None:
        if wf.cancelled:
            return
        # Refresh every running agent's state first (and validate schema on completion), THEN decide
        # what is finished -- otherwise completion would always lag a tick behind reality.
        live = [ar for stage_runs in wf.runs.values() for ar in stage_runs.values() if not ar.settled]
        errors = [error for _, _, error in self._map(self._refresh, live,
                                                     between=self._ingest_between) if error]
        finished_by_stage = [self._finished_indices(wf, si) for si in range(len(wf.plan.stages))]
        if not wf.paused:  # re-read: a `stop` may have been ingested during the refreshes
            active = active_stage_index(wf.plan, finished_by_stage)
            if active is not None:  # None: workflow complete
                stage = wf.plan.stages[active]
                stage_runs = wf.runs.setdefault(active, {})
                todo = stage_next_indices(stage, set(stage_runs), finished_by_stage[active])
                for idx, ar, error in self._map(lambda i: self._spawn_agent(wf, active, i), todo,
                                                between=self._ingest_between):
                    if error:  # left unstarted: the next tick retries it
                        errors.append(error)
                    elif ar is not None:  # None: paused mid-wave; the resume starts it
                        stage_runs[idx] = ar
        if errors:
            raise errors[0]  # run_forever records it; everything that succeeded is already kept

    #: While a :meth:`_map` batch is in flight, run its ``between`` callback this often (seconds).
    INFLIGHT_POLL = 5.0

    def _map(self, fn: Any, items: list, *,
             between: Any = None) -> list[tuple[Any, Any, BaseException | None]]:
        """``fn`` over ``items`` on up to :data:`CONTROLLER_CONCURRENCY` threads, in order:
        ``[(item, result, error)]`` -- one failure never discards the others' results.

        ``between`` (no args) runs on the calling thread every :attr:`INFLIGHT_POLL` seconds until
        the batch finishes, so a long batch (a 100-agent create) does not hold up the requests and
        acks behind it."""

        if not items:
            return []
        with ThreadPoolExecutor(max_workers=min(CONTROLLER_CONCURRENCY, len(items))) as pool:
            futures = [(item, pool.submit(fn, item)) for item in items]
            pending = {future for _, future in futures}
            try:
                while between is not None and pending:
                    _, pending = futures_wait(pending, timeout=self.INFLIGHT_POLL)
                    if pending:
                        between()
            except Exception:  # noqa: BLE001 - never lose the batch: a create that succeeded but
                pass  # whose result is dropped is a sandbox nobody tracks (or stops)
        out = []
        for item, future in futures:
            error = future.exception()
            out.append((item, None if error else future.result(), error))
        return out

    def _finished_indices(self, wf: _WorkflowRuntime, stage_index: int) -> set[int]:
        done = set()
        for idx, ar in wf.runs.get(stage_index, {}).items():
            if ar.state in TERMINAL_STATES:
                done.add(idx)
        return done

    def _spawn_agent(self, wf: _WorkflowRuntime, stage_index: int, idx: int) -> _AgentRun | None:
        if self._halt_requested:  # a hard stop landed mid-wave: create nothing more
            raise AomError(f"swarm {self.swarm_id} was stopped")
        if wf.paused or wf.cancelled:  # `stop WORKFLOW` acked mid-wave: start no more of it
            return None
        stage = wf.plan.stages[stage_index]
        spec = stage.agents[idx]
        box = self._sb.create(
            "modal",
            endpoint=self.endpoint,
            proxy=self.proxy,
            swarm=self.swarm_id,
            shared_volume=self.shared_volume,
        )
        prompt = spec.prompt
        if self.proxy_note:  # 'agent opt-in' mode: give this fan-out agent the proxy endpoints too
            prompt = self.proxy_note + "\n\n---\n\n" + prompt
        if spec.schema is not None:
            out = f"/shared/{RESERVED}/outputs/{box.id}.json"
            copy = (_SCHEMA_COPY.format(copy_path=f"{spec.report_dir}/{RESULT_FILE}")
                    if spec.report_dir else "")
            prompt += _SCHEMA_INSTRUCTION.format(path=out, copy=copy,
                                                 schema=json.dumps(spec.schema, indent=2))
        run = box.start(
            spec.harness or self.harness,
            self.endpoint,
            prompt,
            goal=spec.goal,
            timeout=spec.timeout,
            effort=spec.effort,
            meta={
                "swarm_id": self.swarm_id,
                "workflow": wf.name,
                "stage": stage.name,
                "idx": idx,
                "role": "subagent",
            },
        )
        ar = _AgentRun(idx=idx, stage=stage.name, workflow=wf.name, spec=spec, box=box, run=run)
        self._refresh(ar)
        return ar

    #: A status read may blip; once reads have failed this long AND this many times in a row while
    #: the box still polls alive, the run cannot report again (its process exited and the read-back
    #: is gone), so settle it -- a finished lane left "running" hangs `aom-swarm wait` and the stage
    #: behind it. Both bounds: one failed read after a slow tick (a 100-agent spawn tick can run past
    #: a minute) is not "unreadable for a minute", and settling a live lane would start the next
    #: stage under it.
    STATUS_UNREADABLE_SETTLE = 120.0
    STATUS_UNREADABLE_READS = 3

    def _refresh(self, ar: _AgentRun) -> None:
        try:
            state = str(ar.run.status().get("state", ar.state))
            ar.state = "cancelled" if ar.stopped and state not in TERMINAL_STATES else state
            ar.last_status_ok = time.time()
            ar.refresh_errors = 0
        except Exception:  # noqa: BLE001 - mid-flight read; keep the last known state...
            ar.refresh_errors += 1
            # ...unless the sandbox itself is gone, or the status has been unreadable long enough
            # that it cannot report again: then the run is over, whatever the last read said.
            if ar.stopped:
                ar.state = "cancelled"
            elif ar.state not in TERMINAL_STATES:
                unreadable_for = time.time() - ar.last_status_ok
                if not _box_alive(ar.box):
                    ar.state = "failed"
                    ar.error = ("sandbox exited before the run finished "
                                "(timeout, idle timeout, or OOM)")
                elif (unreadable_for >= self.STATUS_UNREADABLE_SETTLE
                      and ar.refresh_errors >= self.STATUS_UNREADABLE_READS):
                    ar.state = "failed"
                    ar.error = (f"status unreadable for {unreadable_for:.0f}s "
                                f"({ar.refresh_errors} reads in a row); settled as failed")
        self._publish_trace(ar)
        # Checked on the first terminal refresh and, if still missing, once more on the settling
        # one: the agent's last write can trail its exit on the volume by a few seconds.
        if (ar.state in TERMINAL_STATES and ar.spec.schema is not None
                and ar.schema_ok is not True and not ar.settled):
            self._check_schema(ar)
        self._publish_agent(ar)
        if ar.state in TERMINAL_STATES:
            ar.settled, ar._seen_terminal = ar._seen_terminal, True
        elif time.time() - ar.last_reload >= SUBAGENT_RELOAD_INTERVAL:
            ar.last_reload = time.time()
            _reload_volumes(ar.box)  # let it see what its siblings wrote to /shared

    def _check_schema(self, ar: _AgentRun) -> None:
        primary = f"{RESERVED}/outputs/{ar.id}.json"
        produced = self.control.get_json(primary)
        if produced is None and ar.spec.report_dir:
            # Agents that miss the injected path often still wrote the copy in their lane tree
            # (the path the prompt names, or one level down under their own id). Promote it to the
            # canonical path so the orchestrator reads every result from one place.
            for candidate in (f"{ar.spec.report_dir}/{RESULT_FILE}",
                              f"{ar.spec.report_dir}/{ar.id}/{RESULT_FILE}"):
                produced = self.control.read_shared_json(candidate)
                if produced is not None:
                    try:
                        self.control.put_json(primary, produced)
                    except Exception:  # noqa: BLE001 - validation below still uses it
                        pass
                    break
        if produced is None:
            ar.schema_ok = False
            return
        ar.schema_ok = not validate_against_schema(produced, ar.spec.schema or {})

    def _terminate(self, ar: _AgentRun) -> None:
        ar.stopped = True
        try:
            ar.box.stop()
        except Exception:  # noqa: BLE001 - best effort
            pass
        ar.state = "cancelled"
        self._publish_agent(ar)

    def _halt(self) -> None:
        """Hard stop (idempotent): cancel every workflow, terminate every agent still running
        (including any created mid-wave after the stop was requested), publish, then go quiet."""

        if self.halted:
            return
        self._halt_requested = True

        def stop(ar: _AgentRun) -> None:
            try:  # one that finished since the last tick keeps its real state, not "cancelled"
                state = str(ar.run.status().get("state", ar.state))
            except Exception:  # noqa: BLE001 - unreadable: terminate it regardless
                state = ar.state
            if state in TERMINAL_STATES:
                ar.state = state
                try:
                    ar.box.stop()
                except Exception:  # noqa: BLE001 - best effort
                    pass
                self._publish_agent(ar)
            else:
                self._terminate(ar)

        live = [ar for wf in self._workflows.values() for runs in wf.runs.values()
                for ar in runs.values() if ar.state not in TERMINAL_STATES]
        self._map(stop, live)  # in parallel; one failure never stops the rest
        for wf in self._workflows.values():
            wf.cancelled = True
            self._publish_workflow(wf)
        self.halted = True

    # -- publishing ---------------------------------------------------------------------------- #
    #: Keep this many of the most recent events per agent on the volume. Generous: enough for the
    #: UI to render a completed agent's whole trace after teardown, bounded so a very long /goal run
    #: can't grow a trace file without limit.
    TRACE_EVENT_CAP = 5000

    def _publish_trace(self, ar: _AgentRun) -> None:
        try:
            envelopes, ar.offset = ar.run.read_trace(ar.offset)
        except Exception:  # noqa: BLE001
            return
        self._append_events(ar.id, ar.run, envelopes)

    def _reload_orchestrator(self, *, force: bool = False) -> None:
        if not self.orchestrator_id:
            return
        if not force and time.time() - self._orch_reloaded < ORCHESTRATOR_RELOAD_INTERVAL:
            return
        self._orch_reloaded = time.time()
        try:
            if self._orch_box is None:
                self._orch_box = self._sb.get("modal", self.orchestrator_id)
        except Exception:  # noqa: BLE001 - orchestrator gone; nothing to reload
            return
        _reload_volumes(self._orch_box)

    def _tail_orchestrator(self) -> None:
        """Persist the orchestrator's trace too (it is a sandbox, not a subagent), so it survives
        teardown and the dashboard can show it after the run ends."""

        if not self.orchestrator_id or not self.orchestrator_run_id:
            return
        try:
            if self._orch_run is None:
                self._orch_run = self._sb.get("modal", self.orchestrator_id).run(
                    self.orchestrator_run_id
                )
            envelopes, self._orch_offset = self._orch_run.read_trace(self._orch_offset)
        except Exception:  # noqa: BLE001 - orchestrator gone / mid-flight read
            return
        if envelopes and not self._orch_session:
            self._discover_orchestrator_session(envelopes)
        self._append_events(self.orchestrator_id, self._orch_run, envelopes)

    def _discover_orchestrator_session(self, envelopes: list) -> None:
        """Find the orchestrator's session id in its trace and publish it to meta.json once (S9)."""

        for env in envelopes:
            try:
                for ev in self._orch_run.parse(env):
                    sid = (ev.data or {}).get("session_id") if ev.kind == "session" else None
                    if sid:
                        self._orch_session = sid
                        meta = self.control.get_json(f"{RESERVED}/meta.json") or {}
                        if meta.get("orchestrator_session") != sid:
                            meta["orchestrator_session"] = sid
                            self.control.put_json(f"{RESERVED}/meta.json", meta)
                        return
            except Exception:  # noqa: BLE001 - never let trace parsing break the tail
                continue

    # -- orchestrator supervision (issue S3) -------------------------------------------------- #
    def _has_unfinished_workflows(self) -> bool:
        try:
            for name in self.control.listdir(f"{RESERVED}/workflows"):
                if name.endswith(".json"):
                    wf = self.control.get_json(f"{RESERVED}/workflows/{name}") or {}
                    if wf.get("status") in ("running", "paused"):
                        return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _supervise_orchestrator(self) -> None:
        """Relaunch a dead orchestrator (issue S3). OPT-IN via ``AOM_SWARM_SUPERVISE`` (default OFF so
        it can never destabilize an existing run). Only fires when the orchestrator has actually
        exited AND work remains; bounded by ``AOM_SWARM_MAX_RESTARTS`` with exponential backoff; each
        attempt is recorded to ``_swarm/supervisor.json``. The relaunch is a FRESH orchestrator
        session carrying a state snapshot (not a true --resume -- see resume_orchestrator / S9)."""

        if not os.environ.get("AOM_SWARM_SUPERVISE"):
            return
        if not self.orchestrator_id or not self.orchestrator_run_id or self._orch_run is None:
            return
        try:
            state = str(self._orch_run.status().get("state"))
        except Exception:  # noqa: BLE001 - mid-flight read; try again next tick
            return
        if state not in TERMINAL_STATES or not self._has_unfinished_workflows():
            return
        if self._orch_restarts >= int(os.environ.get("AOM_SWARM_MAX_RESTARTS", "3")):
            return
        now = time.time()
        backoff = min(300.0, 10.0 * (2 ** self._orch_restarts))
        if now - self._last_restart_at < backoff:
            return
        self._last_restart_at = now
        try:
            result = resume_orchestrator(
                self.swarm_id, endpoint=self.endpoint, sandboxes=self._sb, proxy=self.proxy
            )
        except Exception as error:  # noqa: BLE001 - record and retry next eligible tick
            self.control.put_json(f"{RESERVED}/supervisor.json",
                                   {"ts": now, "error": str(error), "restarts": self._orch_restarts})
            return
        self._orch_restarts += 1
        self.orchestrator_id = result["orchestrator_id"]
        self.orchestrator_run_id = result["orchestrator_run_id"]
        self._orch_run = None
        self._orch_offset = 0
        self._orch_session = None
        self.control.put_json(
            f"{RESERVED}/supervisor.json",
            {"ts": now, "restarts": self._orch_restarts, "orchestrator_id": self.orchestrator_id,
             "resumed_from": result.get("resumed_from")},
        )

    def _append_events(self, agent_id: str, run: Any, envelopes: list) -> None:
        """Parse new trace envelopes into events and append them to the agent's durable trace file
        on the volume (``_swarm/traces/<id>.jsonl``, one event JSON per line). Parsed, not raw, so
        the dashboard can render a completed agent straight from the volume once its sandbox is
        gone -- the live ``/api/events`` path reads the sandbox; this is the fallback."""

        if not envelopes:
            return
        events: list[dict] = []
        for env in envelopes:
            try:
                events.extend(ev.to_json() for ev in run.parse(env))
            except Exception:  # noqa: BLE001 - never let one bad envelope drop the batch
                continue
        if not events:
            return
        existing = self.control.read_text(f"{RESERVED}/traces/{agent_id}.jsonl") or ""
        lines = existing.splitlines()
        lines.extend(json.dumps(ev, default=str) for ev in events)
        self.control.put_text(
            f"{RESERVED}/traces/{agent_id}.jsonl", "\n".join(lines[-self.TRACE_EVENT_CAP:]) + "\n"
        )

    def _publish_agent(self, ar: _AgentRun) -> None:
        self.control.put_json(
            f"{RESERVED}/agents/{ar.id}.json",
            {
                "id": ar.id,
                "run_id": ar.run_id,
                "workflow": ar.workflow,
                "stage": ar.stage,
                "idx": ar.idx,
                "role": "subagent",
                "harness": ar.spec.harness or self.harness,
                "state": ar.state,
                "goal": ar.spec.goal,
                "timeout": ar.spec.timeout,
                "effort": ar.spec.effort,
                "lane": ar.spec.lane,
                "report_dir": ar.spec.report_dir,
                "updated_at": ar.last_status_ok,
                "schema": ar.spec.schema is not None,
                "schema_ok": ar.schema_ok,
                "error": ar.error,
            },
        )

    def _publish_workflow(self, wf: _WorkflowRuntime) -> None:
        finished_by_stage = [self._finished_indices(wf, si) for si in range(len(wf.plan.stages))]
        active = active_stage_index(wf.plan, finished_by_stage)
        if wf.cancelled:
            status = "cancelled"
        elif active is None:
            status = "done"
        elif wf.paused:
            status = "paused"
        else:
            status = "running"
        stages = []
        for si, stage in enumerate(wf.plan.stages):
            runs = wf.runs.get(si, {})
            if stage_complete(stage, finished_by_stage[si]):
                sstatus = "done"
            elif runs:
                sstatus = "running"
            elif active is not None and si < active:
                sstatus = "done"
            elif active == si:
                sstatus = "running"
            else:
                sstatus = "pending"
            stages.append(
                {
                    "name": stage.name,
                    "max_concurrent": stage.max_concurrent,
                    "status": sstatus,
                    "planned": len(stage.agents),
                    "agents": [
                        {
                            "idx": idx,
                            "id": ar.id,
                            "run_id": ar.run_id,
                            "state": ar.state,
                            "goal": ar.spec.goal,
                            "timeout": ar.spec.timeout,
                            "effort": ar.spec.effort,
                            "lane": ar.spec.lane,
                            "report_dir": ar.spec.report_dir,
                            "updated_at": ar.last_status_ok,
                            "schema_ok": ar.schema_ok,
                        }
                        for idx, ar in sorted(runs.items())
                    ],
                }
            )
        self.control.put_json(
            f"{RESERVED}/workflows/{wf.name}.json",
            {"name": wf.name, "status": status, "created_at": wf.created_at, "stages": stages},
        )


# --------------------------------------------------------------------------------------------- #
# DynamicSwarm handle                                                                            #
# --------------------------------------------------------------------------------------------- #
@dataclass
class DynamicSwarm:
    """A handle on a dynamic swarm: the orchestrator sandbox + its shared control volume.

    :meth:`launch` deploys the model, spawns the single orchestrator agent (with the injected
    instructions), and returns. The *controller* that executes the orchestrator's plans runs
    separately (the dashboard starts one in a thread); it is intentionally not part of the handle
    so the handle stays a cheap, reconnectable value like :class:`~agents_on_modal.swarm.Swarm`.
    """

    id: str
    model: str
    shared_volume: str
    harness: str
    orchestrator_id: str | None = None
    orchestrator_run_id: str | None = None
    default_timeout: float = DEFAULT_TIMEOUT
    _orchestrator_box: Any = None
    _orchestrator_run: Any = None

    @classmethod
    def launch(
        cls,
        harness: str,
        model: str | Any,
        prompt: str,
        *,
        timeout: float,
        max_containers: int,
        effort: str | None = None,
        proxy: str | None = None,
        shared_volume: str | None = None,
        default_subagent_timeout: float = DEFAULT_TIMEOUT,
        control: "ControlPlane | None" = None,
    ) -> "DynamicSwarm":
        from . import models, sandbox
        from .models import lifecycle as model_ops

        resolved = models.get(model)
        # Attach to the shared model (deploying it only if it isn't yet); never redeploy it or size
        # it by agent count. The model is an account-level shared resource -- Modal's concurrency
        # autoscaler sizes it from aggregate load across the orchestrator, this swarm's subagents,
        # and any other consumer, under the catalog policy (``Scaling``) and the container cap the
        # apps on it asked for (``max_containers`` only ever raises it) -- and a redeploy would
        # replace its containers under all of them.
        endpoint = model_ops.ensure_deployed(resolved, max_containers=max_containers)

        swarm_id = sandbox.new_id("swarm-")
        shared = shared_volume or f"aom-shared-{swarm_id}"
        allowed = _model_efforts(resolved)
        instructions = build_orchestrator_instructions(
            default_timeout=default_subagent_timeout,
            allowed_efforts=allowed,
        )
        full_prompt = f"{instructions}\n\n---\n\n# Your task\n\n{prompt}\n"

        box = sandbox.create(
            "modal", endpoint=endpoint, proxy=proxy, swarm=swarm_id, shared_volume=shared
        )
        _install_cli_shim(box)
        run = box.start(
            harness,
            endpoint,
            full_prompt,
            goal=True,  # the orchestrator must persist until the whole workflow is done
            timeout=timeout,
            effort=effort,
            meta={"swarm_id": swarm_id, "role": "orchestrator"},
        )
        # Write meta up-front so the dashboard recognizes the swarm as dynamic immediately (the
        # controller rewrites it too). Best-effort: a volume hiccup must not fail the launch.
        cp = control if control is not None else VolumeControlPlane(shared)
        try:
            cp.put_json(
                f"{RESERVED}/meta.json",
                {
                    "swarm_id": swarm_id,
                    "model": resolved.name,
                    "harness": harness,
                    "orchestrator_id": box.id,
                    "orchestrator_run_id": getattr(run, "run_id", None),
                    "default_timeout": default_subagent_timeout,
                    "created_at": time.time(),
                },
            )
        except Exception:  # noqa: BLE001
            pass
        # The launch record, for tracking and reproduction: the exact prompt the orchestrator got
        # (injected instructions + task) and the settings to relaunch it. Records only WHETHER a
        # proxy was used -- its URL can carry credentials.
        try:
            cp.put_text(f"{RESERVED}/{ORCHESTRATOR_PROMPT_FILE}", full_prompt)
            cp.put_json(
                f"{RESERVED}/{LAUNCH_FILE}",
                {
                    "swarm_id": swarm_id,
                    "created_at": time.time(),
                    "harness": harness,
                    "model": resolved.name,
                    "effort": effort,
                    "timeout": timeout,
                    "max_containers": max_containers,
                    "default_subagent_timeout": default_subagent_timeout,
                    "proxied": bool(proxy),
                    "orchestrator_id": box.id,
                    "orchestrator_run_id": getattr(run, "run_id", None),
                    "task": prompt,
                    "orchestrator_prompt_file": f"{RESERVED}/{ORCHESTRATOR_PROMPT_FILE}",
                },
            )
        except Exception:  # noqa: BLE001 - best-effort, like meta.json
            pass
        return cls(
            id=swarm_id,
            model=resolved.name,
            shared_volume=shared,
            harness=harness,
            orchestrator_id=box.id,
            orchestrator_run_id=getattr(run, "run_id", None),
            default_timeout=default_subagent_timeout,
            _orchestrator_box=box,
            _orchestrator_run=run,
        )

    def controller(self, endpoint: Any, *, proxy: str | None = None,
                   proxy_note: str | None = None, sandboxes: Any = None) -> Controller:
        """Build a :class:`Controller` for this swarm (the dashboard runs ``.run_forever()``)."""

        return Controller(
            VolumeControlPlane(self.shared_volume),
            self.id,
            endpoint,
            self.shared_volume,
            self.harness,
            proxy=proxy,
            proxy_note=proxy_note,
            orchestrator_id=self.orchestrator_id,
            orchestrator_run_id=self.orchestrator_run_id,
            default_timeout=self.default_timeout,
            sandboxes=sandboxes,
        )

    @classmethod
    def reconnect(cls, swarm_id: str, *, sandboxes: Any = None) -> "DynamicSwarm":
        """Rebuild a handle from the swarm id alone (control plane + live sandboxes)."""

        if sandboxes is None:
            from . import sandbox as sandboxes
        shared = f"aom-shared-{swarm_id}"
        control = VolumeControlPlane(shared)
        meta = control.get_json(f"{RESERVED}/meta.json") or {}
        return cls(
            id=swarm_id,
            model=str(meta.get("model", "")),
            shared_volume=shared,
            harness=str(meta.get("harness", "")),
            orchestrator_id=meta.get("orchestrator_id"),
            orchestrator_run_id=meta.get("orchestrator_run_id"),
            default_timeout=float(meta.get("default_timeout", DEFAULT_TIMEOUT)),
        )

    def control(self) -> ControlPlane:
        return VolumeControlPlane(self.shared_volume)

    def workflows(self) -> list[dict[str, Any]]:
        control = self.control()
        out = []
        for name in control.listdir(f"{RESERVED}/workflows"):
            if name.endswith(".json"):
                wf = control.get_json(f"{RESERVED}/workflows/{name}")
                if wf:
                    out.append(wf)
        return out

    def agents(self) -> list[dict[str, Any]]:
        control = self.control()
        out = []
        if self.orchestrator_id:
            out.append({"id": self.orchestrator_id, "role": "orchestrator",
                        "run_id": None, "state": "running"})
        for name in control.listdir(f"{RESERVED}/agents"):
            if name.endswith(".json"):
                agent = control.get_json(f"{RESERVED}/agents/{name}")
                if agent:
                    out.append(agent)
        return out


def _state_snapshot(control: ControlPlane) -> str:
    """A compact, human-readable snapshot of a swarm's control-plane state for a resuming
    orchestrator: each workflow/stage with its status and done/total agent counts, plus the
    top-level shared files it should read to recover intent."""

    lines: list[str] = []
    try:
        for name in sorted(control.listdir(f"{RESERVED}/workflows")):
            if not name.endswith(".json"):
                continue
            wf = control.get_json(f"{RESERVED}/workflows/{name}") or {}
            stages = wf.get("stages") or []
            lines.append(f"- workflow `{wf.get('name', name)}`: {wf.get('status', '?')} "
                         f"({len(stages)} stage(s))")
            for st in stages:
                agents = st.get("agents") or []
                done = sum(1 for a in agents if a.get("state") in TERMINAL_STATES)
                lines.append(f"    - stage `{st.get('name', '?')}`: {st.get('status', '?')} "
                             f"[{done}/{len(agents)} agents done]")
    except Exception:  # noqa: BLE001 - a snapshot is best-effort
        pass
    try:
        files = [n for n in control.listdir("") if n != RESERVED]
        if files:
            lines.append("- shared files to read first: " + ", ".join(sorted(files)[:40]))
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines) or "(no workflows recorded yet)"


def resume_orchestrator(
    swarm_id: str,
    *,
    endpoint: Any = None,
    sandboxes: Any = None,
    proxy: str | None = None,
    timeout: float | None = None,
    effort: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Relaunch a dynamic swarm's orchestrator after it has exited (issue S3), from the HOST or from
    the controller's opt-in supervisor.

    The orchestrator is a single point of failure: when it dies nothing restarts it, and there is no
    control-plane verb that brings it back. This launches a NEW orchestrator sandbox bound to the
    same swarm id and shared workspace, primed with the standard instructions plus a snapshot of the
    current control-plane state so it can continue the mission and not restart finished work.

    HONEST LIMITATION: this is a *fresh* orchestrator session, not a true harness ``--resume`` of the
    old one -- a new sandbox gets an empty harness home, so the old session's files are gone (see
    Sandbox.start and issue S9). Continuity comes from the shared workspace (PLAN.md, notes, outputs)
    and the injected snapshot -- exactly the manual recovery the issue describes, now automated.
    Updates ``meta.json`` and returns ``{swarm, orchestrator_id, orchestrator_run_id, resumed_from}``.
    """

    from . import models  # noqa: PLC0415 - lazy so the module stays modal-free to import
    if sandboxes is None:
        from . import sandbox as sandboxes  # noqa: PLC0415

    shared = f"aom-shared-{swarm_id}"
    control = VolumeControlPlane(shared)
    meta = control.get_json(f"{RESERVED}/meta.json") or {}
    model_name = str(meta.get("model") or "")
    if not model_name:
        raise AomError(f"swarm {swarm_id!r} has no meta/model recorded; cannot resume its orchestrator.")
    harness = str(meta.get("harness") or "claude")
    model = models.get(model_name)
    if endpoint is None:
        endpoint = models.endpoint(model)
    default_timeout = float(meta.get("default_timeout", DEFAULT_TIMEOUT))

    instructions = build_orchestrator_instructions(
        default_timeout=default_timeout, allowed_efforts=_model_efforts(model),
    )
    prior = meta.get("orchestrator_id")
    # The supervisor's corrections are the newest intent; a replacement must start from them.
    supervisor = (control.read_text("NOTE-supervisor.md") or "").strip()
    if len(supervisor) > 6000:
        supervisor = "...(older notes cut; read /shared/NOTE-supervisor.md)\n" + supervisor[-6000:]
    task = (
        f"{instructions}\n\n---\n\n# Resuming an in-flight mission\n\n"
        + (f"{note}\n\n" if note else "")
        + (f"Supervisor notes so far (they outrank the plan):\n\n{supervisor}\n\n"
           if supervisor else "")
        + "You are a REPLACEMENT orchestrator: the previous orchestrator for this swarm exited. The "
        "shared workspace under `/shared` is intact — read `PLAN.md` and any notes there FIRST to "
        "recover the mission's intent and conventions. Do NOT restart work that is already finished. "
        "Current control-plane state:\n\n"
        f"{_state_snapshot(control)}\n\n"
        "Continue from here: resume unfinished workflows/stages, spawn only what is still missing, "
        "and drive the mission to completion.\n"
    )

    box = sandboxes.create("modal", endpoint=endpoint, proxy=proxy, swarm=swarm_id, shared_volume=shared)
    _install_cli_shim(box)
    run = box.start(
        harness, endpoint, task, goal=True, timeout=(timeout or default_timeout), effort=effort,
        meta={"swarm_id": swarm_id, "role": "orchestrator", "resumed_from": prior},
    )
    run_id = getattr(run, "run_id", None)
    meta.update({
        "orchestrator_id": box.id,
        "orchestrator_run_id": run_id,
        "orchestrator_session": None,  # fresh session; the controller's tail rediscovers it
        "resumed_from": prior,
        "resumed_at": time.time(),
    })
    control.put_json(f"{RESERVED}/meta.json", meta)
    return {"swarm": swarm_id, "orchestrator_id": box.id,
            "orchestrator_run_id": run_id, "resumed_from": prior}


def _model_efforts(model: Any) -> list[str]:
    """The distinct engine-side effort levels this model actually exercises, cheapest first."""

    efforts = getattr(getattr(model, "reasoning", None), "efforts", {}) or {}
    distinct = {v for v in efforts.values() if isinstance(v, str)}
    ordered = [e for e in EFFORTS if e in distinct]
    return ordered or ["low", "medium", "high"]


def _reload_volumes(box: Any) -> None:
    """Best-effort ``box.reload_volumes()``: a failed reload only delays visibility to the next one."""

    try:
        box.reload_volumes()
    except Exception:  # noqa: BLE001 - no reload_volumes() (fakes), sandbox gone, or a flaky call
        pass


def _box_alive(box: Any) -> bool:
    """Whether ``box`` still exists; ``True`` when that can't be told (never fail a run on doubt)."""

    try:
        return bool(box.alive())
    except Exception:  # noqa: BLE001 - no alive() (fakes) or a flaky poll
        return True


#: ``aom`` in a sandbox: the host CLI needs Modal credentials a sandbox never has, but agents reach
#: for it by analogy with ``aom-swarm``, so answer with a pointer instead of "command not found".
_AOM_HINT_SHIM = (
    "#!/bin/sh\n"
    "echo 'aom: the host-side aom CLI (models, deploys) is not available inside a sandbox.' >&2\n"
    "echo 'Use aom-swarm to drive subagents: aom-swarm --help' >&2\n"
    "exit 127\n"
)


def _install_cli_shim(box: Any) -> None:
    """Best-effort: put an ``aom-swarm`` wrapper (and an ``aom`` pointer to it) on PATH in the
    orchestrator sandbox. If it fails, the injected instructions also give the
    ``python3 -m agents_on_modal.swarmctl`` form."""

    shims = {
        "aom-swarm": '#!/bin/sh\nexec python3 -m agents_on_modal.swarmctl "$@"\n',
        "aom": _AOM_HINT_SHIM,
    }
    for name, text in shims.items():
        try:
            box.write_file(f"/tmp/{name}", text)
            box.exec(["install", "-m", "0755", f"/tmp/{name}", f"/usr/local/bin/{name}"], root=True)
        except Exception:  # noqa: BLE001 - sugar only
            pass


# --------------------------------------------------------------------------------------------- #
# Injected orchestrator instructions (shown verbatim in the dashboard)                           #
# --------------------------------------------------------------------------------------------- #
def build_orchestrator_instructions(
    *, default_timeout: float, allowed_efforts: list[str]
) -> str:
    """Markdown prepended to the orchestrator's prompt (and surfaced by the dashboard so the user
    sees exactly what was injected)."""

    efforts = ", ".join(allowed_efforts)
    timeout = int(default_timeout)
    return f"""# You are a dynamic-swarm orchestrator

You can spawn and manage a fleet of **subagents** (each its own Modal sandbox running the same
harness and model as you) to accomplish the task below. You do this with the `aom-swarm` command.

The endpoint is shared and batch-served: it runs many requests per GPU at once, so a fleet is cheaper
per unit of work than a trickle and an agent that sits on a tool call costs the GPU nothing. Agents
are only expensive when the box is already full or idle-waiting on each other — so prefer a wide
concurrent pool over a narrow one, unless the task itself says otherwise.

## Shared workspace
`/work` and `$HOME` are local to each agent's own sandbox and start empty; `/shared` (a network
volume) is the only directory every agent sees. Anything another agent must use — a venv, a built
binary, a dataset — goes under `/shared` together with a script that sets it up. `/shared/_swarm/`
is the `aom-swarm` control plane: you may read it, never write there. Everything else is yours to
organize — e.g. one `/shared/reports/<lane>/` directory per subagent — and you must spell the
layout out in each subagent's prompt (they only know what you tell them).

**One writer per file.** Agents writing *different* files under `/shared` all land fine, but two
agents appending to or editing the *same* file overwrite each other (last writer wins — 100 agents
appending to one board file kept 1 line). Give each its own path and combine them yourself after
`aom-swarm wait`. The mount is eventually consistent: a file another agent just wrote can take a
minute or two to show up (a missing report is lag, not a missing agent), and after rewriting a whole
file, re-read it and check its size/mtime before relying on it.

Subagents are separate sandboxes: your harness's own sub-agent and messaging tools do not reach
them. To steer running subagents, write `/shared/NOTE-<lane>.md` and tell each one in its prompt to
read its note at start and at every checkpoint; to end one, `aom-swarm stop` it.

## Timing and clocks
- **Latency is not failure.** `submit`/`spawn` wait up to ~100 s for the controller's ack, and under
  load an ack can arrive later than that. A request that timed out is very likely accepted: it is
  listed under `PENDING` in `aom-swarm status` until its ack lands. Never resubmit it — run
  `aom-swarm wait`. (The CLI refuses a second `submit` under a name that is pending or running.)
- A subagent that just finished can show `running` for a few more seconds; its report on disk is
  what counts.
- Sandbox clocks can be far off real time. For timestamps, read `utc` (or `ts`) from
  `/shared/_swarm/now.json` — the controller's clock — instead of running `date`.

## The `aom-swarm` CLI
Run it in your shell. (If `aom-swarm` is not found, use `python3 -m agents_on_modal.swarmctl`.)
It is the only control command here: the host-side `aom` CLI (models, deploys) is not available in
a sandbox.

- `aom-swarm submit --file plan.json [--name NAME]` — run a whole plan (waves + pools). Validates
  the plan and prints errors; fix them and resubmit. You can submit several plans, each under its
  own name; to run a plan again on purpose, give it a new name (e.g. `wave1-retry`).
- `aom-swarm spawn --prompt "..." [--goal] --timeout N [--effort E] [--lane L] [--report-dir D]
  [--schema s.json | --no-schema] [--workflow W]` — spawn one subagent ad hoc. When
  `/shared/schema.json` exists it is attached by default; pass `--no-schema` for a setup or helper
  agent. Name the workflow after the lane (e.g. `--workflow w1-topic-a-retry`) so it is easy to find.
- `aom-swarm status [--workflow W] [--json]` — workflows, stages and subagents (with their lane, and
  `STUCK` when the controller has had no status from one for 5 min), plus requests still pending an
  ack. Use `--workflow W` to look at one workflow.
- `aom-swarm trace AGENT_ID [--tail N]` — read a subagent's recent activity.
- `aom-swarm wait [--all|--any] [--workflow W]` — block until subagents finish. Use this to
  sequence work: submit a wave, `wait --all`, read results from `/shared`, then spawn the next wave
  or a final synthesis agent. One call blocks up to ~100 s and then says how many agents are done;
  just run it again until it prints `ok`. Every return also prints whatever is new in
  `/shared/NOTE*` files; a *supervisor* note there (from the human running the swarm) outranks your
  plan. Do NOT poll `status` in a loop or use `sleep` — `aom-swarm wait` is the ONLY correct way to
  wait.
- `aom-swarm stop TARGET` — by workflow name: pause it (no new agents start; running ones continue);
  by agent id: terminate that subagent. It prints what the target covers before acting.
- `aom-swarm resume --workflow W` — un-pause a workflow.

## Plan format (`submit`)
```json
{{
  "name": "research",
  "stages": [
    {{"name": "wave1", "max_concurrent": 6, "agents": [
        {{"lane": "w1-topic-a", "report_dir": "/shared/reports/w1-topic-a",
          "prompt": "…full instructions, incl. writing to /shared/reports/w1-topic-a…",
          "goal": true, "timeout": 1800, "effort": "medium"}}
    ]}},
    {{"name": "summary", "agents": [
        {{"prompt": "read the reports the others wrote and produce a summary at …",
          "timeout": 1800}}
    ]}}
  ]
}}
```
- **Stages run in order** (a stage must fully finish before the next starts) — this is how you make
  waves. **Within a stage**, up to `max_concurrent` agents run at once and the next starts as one
  finishes — this is how you make a bounded pool (set `max_concurrent` to e.g. 2).
- Per agent: `prompt` (required), `goal` (true = run under /goal until it declares the goal met),
  `timeout` (seconds — **always set one**; default {timeout}), `effort` (one of: {efforts} — the
  model maps any level to its nearest supported); optional `lane` (a short name for the subagent —
  no spaces or `/` — shown in `status`), `report_dir` (the absolute `/shared/...` directory it
  writes to; put the same path in its prompt), `schema` (a JSON Schema, inline or as a path to a
  JSON file — one `/shared/schema.json` can serve a whole wave; the subagent is asked to also emit
  matching JSON, which is validated — a soft contract, not a hard guarantee; with a `report_dir` it
  also writes a copy to `<report_dir>/result.json`), and `harness` (`claude` or `codex`; default:
  the one you run on).
- Use `goal: true` with a long timeout for open-ended work that must keep going until it is done;
  plain runs with short timeouts for bounded lookups, checks and merges. A number you give a
  subagent ("find 20 sources") is a minimum it may exceed — say so in its prompt.

## How to work
1. Do any research you need yourself (browser, web search) to design the workflow.
2. Write `plan.json` and `aom-swarm submit` it — a whole wave goes in one submitted stage; `spawn`
   is for a single reactive subagent (a retry, a setup step), never for a whole wave.
3. `aom-swarm wait` for stages, inspect `/shared`, and continue — add stages, spawn a final
   synthesis agent, etc. Each `wait` return is a checkpoint: act on any new note it printed and
   merge the reports that have landed. The job is not done until the subagents are done and you
   have produced whatever final artifact the task asks for.

Keep your own context for coordinating: after the first wave, any hands-on exploration that would
take more than a handful of tool calls should be a subagent that writes its findings to a file, and
you read only its summary.
"""
