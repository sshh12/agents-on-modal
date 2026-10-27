"""Dynamic swarms: an orchestrator agent that spawns *other* agents through a shared-volume CLI.

A naive :class:`~agents_on_modal.swarm.Swarm` fans out N identical agents. A *dynamic* swarm runs
one **orchestrator** agent (claude/codex in a Modal sandbox) that, given a natural-language prompt,
researches as needed and then drives spawning of **subagents** -- in ordered *waves* (stages) and/or
bounded-concurrency *pools* -- by talking to the ``aom-swarm`` CLI (:mod:`agents_on_modal.swarmctl`).

Everything flows over the shared Modal Volume mounted at ``/shared`` in every agent. A single
reserved namespace, ``/shared/_swarm/``, is the control plane:

    _swarm/meta.json              {swarm_id, model, harness, orchestrator_id, default_timeout}
    _swarm/requests/<reqid>.json  CLI -> controller commands   {id, ts, verb, payload}
    _swarm/ack/<reqid>.json       controller -> CLI replies    {id, ok, errors, warnings, result}
    _swarm/workflows/<name>.json  controller-published plan + live stage/agent state
    _swarm/agents/<agent_id>.json controller-published per-agent status
    _swarm/traces/<agent_id>.jsonl durable per-agent trace (parsed events; survives teardown)
    _swarm/outputs/<agent_id>.json structured outputs for agents given a `schema` (soft contract)

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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .config import AomError
from .models import EFFORTS, normalize_effort

#: Every subagent needs an explicit deadline (Modal sandboxes always take a timeout); this is the
#: fallback when a plan omits one. One hour is enough for a single /goal pass without stranding GPUs.
DEFAULT_TIMEOUT = 3600.0

#: The one reserved directory under ``/shared``. Everything else there is free-form.
RESERVED = "_swarm"

TERMINAL_STATES = ("done", "timeout", "failed", "cancelled")


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

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"prompt": self.prompt, "goal": self.goal, "timeout": self.timeout}
        if self.effort:
            data["effort"] = self.effort
        if self.schema is not None:
            data["schema"] = self.schema
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
        errors.append(f"{where}: 'schema' must be a JSON object")
        schema = None

    for key in data:
        if key not in ("prompt", "goal", "timeout", "effort", "schema"):
            warnings.append(f"{where}: unknown field {key!r} ignored")

    return AgentSpec(prompt=prompt, goal=goal, timeout=timeout, effort=effort, schema=schema)


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
        return None, errors, warnings
    return Plan(stages=stages), errors, warnings


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

    def __init__(self, volume_name: str) -> None:
        self.volume_name = volume_name
        self._vol: Any = None

    def _volume(self) -> Any:
        if self._vol is None:
            import modal

            self._vol = modal.Volume.from_name(self.volume_name, version=2, create_if_missing=True)
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
    "\n\n---\nWhen you have finished, also write your final result as a single JSON document that "
    "conforms to this JSON schema to the file `{path}` (create parent directories if needed):\n"
    "```json\n{schema}\n```\n"
)


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
        # Orchestrator trace tailing (its sandbox is not a subagent; tail it so its trace persists
        # to the volume too and survives teardown).
        self._orch_run: Any = None
        self._orch_offset = 0

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
            time.sleep(interval)

    def tick(self) -> None:
        """One pass: write meta, ingest new requests, advance workflows, publish state."""

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

        for name in self.control.listdir(f"{RESERVED}/requests"):
            if not name.endswith(".json"):
                continue
            reqid = name[: -len(".json")]
            if self.control.exists(f"{RESERVED}/ack/{reqid}.json"):
                continue
            request = self.control.get_json(f"{RESERVED}/requests/{name}")
            if isinstance(request, dict):
                self._handle_request(reqid, request)

        for wf in list(self._workflows.values()):
            self._advance(wf)
            self._publish_workflow(wf)

        self._tail_orchestrator()

    # -- request handling --------------------------------------------------------------------- #
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
        name = self._unique_name(str(payload.get("name") or reqid))
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
        name = self._unique_name(str(payload.get("workflow") or "spawn"))
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
        for stage_runs in wf.runs.values():
            for ar in stage_runs.values():
                self._refresh(ar)
        finished_by_stage = [self._finished_indices(wf, si) for si in range(len(wf.plan.stages))]
        if wf.paused:
            return
        active = active_stage_index(wf.plan, finished_by_stage)
        if active is None:
            return  # workflow complete
        stage = wf.plan.stages[active]
        stage_runs = wf.runs.setdefault(active, {})
        started = set(stage_runs.keys())
        finished = finished_by_stage[active]
        for idx in stage_next_indices(stage, started, finished):
            stage_runs[idx] = self._spawn_agent(wf, active, idx)

    def _finished_indices(self, wf: _WorkflowRuntime, stage_index: int) -> set[int]:
        done = set()
        for idx, ar in wf.runs.get(stage_index, {}).items():
            if ar.state in TERMINAL_STATES:
                done.add(idx)
        return done

    def _spawn_agent(self, wf: _WorkflowRuntime, stage_index: int, idx: int) -> _AgentRun:
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
            prompt += _SCHEMA_INSTRUCTION.format(path=out, schema=json.dumps(spec.schema, indent=2))
        run = box.start(
            self.harness,
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

    def _refresh(self, ar: _AgentRun) -> None:
        try:
            ar.state = str(ar.run.status().get("state", ar.state))
        except Exception:  # noqa: BLE001 - mid-flight read; keep the last known state
            pass
        self._publish_trace(ar)
        if ar.state in TERMINAL_STATES and ar.spec.schema is not None and ar.schema_ok is None:
            self._check_schema(ar)
        self._publish_agent(ar)

    def _check_schema(self, ar: _AgentRun) -> None:
        produced = self.control.get_json(f"{RESERVED}/outputs/{ar.id}.json")
        if produced is None:
            ar.schema_ok = False
            return
        ar.schema_ok = not validate_against_schema(produced, ar.spec.schema or {})

    def _terminate(self, ar: _AgentRun) -> None:
        try:
            ar.box.stop()
        except Exception:  # noqa: BLE001 - best effort
            pass
        ar.state = "cancelled"
        self._publish_agent(ar)

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
        self._append_events(self.orchestrator_id, self._orch_run, envelopes)

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
                "state": ar.state,
                "goal": ar.spec.goal,
                "timeout": ar.spec.timeout,
                "effort": ar.spec.effort,
                "schema": ar.spec.schema is not None,
                "schema_ok": ar.schema_ok,
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
                    "agents": [
                        {
                            "idx": idx,
                            "id": ar.id,
                            "run_id": ar.run_id,
                            "state": ar.state,
                            "goal": ar.spec.goal,
                            "timeout": ar.spec.timeout,
                            "effort": ar.spec.effort,
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
        effort: str | None = None,
        proxy: str | None = None,
        shared_volume: str | None = None,
        default_subagent_timeout: float = DEFAULT_TIMEOUT,
        control: "ControlPlane | None" = None,
    ) -> "DynamicSwarm":
        from . import models, sandbox
        from .models import lifecycle as model_ops

        resolved = models.get(model)
        # Ensure the shared model is deployed (idempotent); do NOT scale it per swarm. The model is
        # an account-level shared resource -- Modal's concurrency autoscaler sizes it from aggregate
        # load across the orchestrator, this swarm's subagents, and any other consumer, under the
        # conservative catalog policy (``Scaling``).
        endpoint = model_ops.deploy(resolved)

        swarm_id = sandbox.new_id("swarm-")
        shared = shared_volume or f"aom-shared-{swarm_id}"
        allowed = _model_efforts(resolved)
        instructions = build_orchestrator_instructions(
            default_timeout=default_subagent_timeout,
            model_name=resolved.name,
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


def _model_efforts(model: Any) -> list[str]:
    """The distinct engine-side effort levels this model actually exercises, cheapest first."""

    efforts = getattr(getattr(model, "reasoning", None), "efforts", {}) or {}
    distinct = {v for v in efforts.values() if isinstance(v, str)}
    ordered = [e for e in EFFORTS if e in distinct]
    return ordered or ["low", "medium", "high"]


def _install_cli_shim(box: Any) -> None:
    """Best-effort: put an ``aom-swarm`` wrapper on PATH in the orchestrator sandbox. If it fails,
    the injected instructions also give the ``python3 -m agents_on_modal.swarmctl`` form."""

    shim = '#!/bin/sh\nexec python3 -m agents_on_modal.swarmctl "$@"\n'
    try:
        box.write_file("/tmp/aom-swarm", shim)
        box.exec(["install", "-m", "0755", "/tmp/aom-swarm", "/usr/local/bin/aom-swarm"], root=True)
    except Exception:  # noqa: BLE001 - sugar only
        pass


# --------------------------------------------------------------------------------------------- #
# Injected orchestrator instructions (shown verbatim in the dashboard)                           #
# --------------------------------------------------------------------------------------------- #
def build_orchestrator_instructions(
    *, default_timeout: float, model_name: str, allowed_efforts: list[str]
) -> str:
    """Markdown prepended to the orchestrator's prompt (and surfaced by the dashboard so the user
    sees exactly what was injected)."""

    efforts = ", ".join(allowed_efforts)
    timeout = int(default_timeout)
    return f"""# You are a dynamic-swarm orchestrator

You can spawn and manage a fleet of **subagents** (each its own Modal sandbox running the same
harness and model, `{model_name}`) to accomplish the task below. You do this with the `aom-swarm`
command. Spawn only what the task needs; subagents are expensive.

## Shared workspace
Every agent — you and all subagents — shares a live directory at `/shared`. The subdirectory
`/shared/_swarm/` is reserved for the `aom-swarm` control plane; **do not write there yourself.**
Everything else under `/shared` is yours to organize however you like. *You* decide how subagents
coordinate — e.g. a message-board file they append findings to, a `reports/` folder where each
writes its own markdown, a conventions file — and you must spell that out in the prompt you give
each subagent (they only know what you tell them). These are your choices, not fixed rules.

## The `aom-swarm` CLI
Run it in your shell. (If `aom-swarm` is not found, use `python3 -m agents_on_modal.swarmctl`.)

- `aom-swarm submit --file plan.json [--name NAME]` — run a whole plan (waves + pools). Validates
  the plan and prints errors; fix them and resubmit. You can submit several plans.
- `aom-swarm spawn --prompt "..." [--goal] --timeout N [--effort E] [--schema s.json] [--workflow W]`
  — spawn one subagent ad hoc.
- `aom-swarm status [--workflow W] [--json]` — see every workflow, stage and subagent's state.
- `aom-swarm trace AGENT_ID [--tail N]` — read a subagent's recent activity.
- `aom-swarm wait [--all|--any] [--workflow W] [--timeout N]` — block until subagents finish. Use
  this to sequence work: submit a wave, `wait --all`, read results from `/shared`, then spawn the
  next wave or a final synthesis agent.
- `aom-swarm stop TARGET` — pause a workflow (by name) or terminate one subagent (by id).
- `aom-swarm resume --workflow W` — un-pause a workflow.

## Plan format (`submit`)
```json
{{
  "name": "research",
  "stages": [
    {{"name": "wave1", "max_concurrent": 6, "agents": [
        {{"prompt": "…full instructions incl. where to write…", "goal": true,
          "timeout": 1800, "effort": "medium"}}
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
  model maps any level to its nearest supported), and optional `schema` (a JSON Schema; the subagent
  is asked to also emit matching JSON, which is validated — a soft contract, not a hard guarantee).

## How to work
1. Do any research you need yourself (browser, web search) to design the workflow.
2. Write `plan.json` and `aom-swarm submit` it (or `spawn`/`wait` reactively).
3. `aom-swarm wait` for stages, inspect `/shared`, and continue — add stages, spawn a final
   synthesis agent, etc. The job is not done until the subagents are done and you have produced
   whatever final artifact the task asks for.
"""
