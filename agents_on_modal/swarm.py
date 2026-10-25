"""Swarm orchestrator: N Modal sandboxes running the same harness/model/prompt.

A *swarm* is a fan-out. :meth:`Swarm.launch` deploys the model, pre-warms it
(``models.scale``), then creates ``count`` Modal sandboxes tagged ``swarm=<id>`` and starts one
headless run in each. Everything a reader needs lives in the sandboxes themselves, so
:meth:`Swarm.reconnect` can rebuild the whole swarm from its id alone, in a fresh process (the
dashboard webapp after a restart): it lists the live sandboxes (``sandbox.list_sandboxes("modal",
swarm=<id>)``) and reattaches to their runs.

Sharing is a Modal **Volume v2** named ``aom-shared-<swarm_id>``, mounted at ``/shared`` in every
agent (the Modal sandbox backend does the mount). It is eventually-consistent, last-writer-wins,
committed periodically and at exit -- the instructions handed to each agent already explain that.
There is no separate hub sandbox and no sync daemon. :meth:`Swarm.read_shared` /
:meth:`Swarm.list_shared` read that Volume from outside Modal, with no container.

Only ``cli.py`` prints; this module raises :class:`~agents_on_modal.config.AomError` subclasses.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import agents_on_modal.models.deploy as model_ops  # the submodule; see note below

from . import models, sandbox
from .config import AomError
from .sandbox.base import Run, Sandbox, SandboxError

# ``models.deploy``/``scale`` are callables in ``agents_on_modal.models.deploy``; reach them via the
# dotted-import ``model_ops`` alias (the package's lazy re-export recurses under ``from``-import).
# ``models.get`` is a real function on the package.

#: Bounded concurrency and retry policy for the create/start fan-out.
MAX_CONCURRENCY = 16
CREATE_RETRIES = 5
RETRY_BACKOFF = 2.0
#: Pre-warm at most this many containers eagerly (the rest scale up on load).
PREWARM_MIN = 4


@dataclass
class Agent:
    """One member of a swarm: a sandbox and the run it is driving."""

    sandbox: Sandbox
    run: Run | None = None
    idx: int | None = None

    @property
    def id(self) -> str:
        return self.sandbox.id

    @property
    def run_id(self) -> str | None:
        return self.run.run_id if self.run is not None else None

    def state(self) -> str:
        if self.run is None:
            return "unknown"
        return str(self.run.status().get("state", "starting"))


@dataclass
class Swarm:
    """A handle on a fan-out. Reconstructible from its id via :meth:`reconnect`."""

    id: str
    model: str
    shared_volume: str
    _agents: list[Agent] = field(default_factory=list)

    # ---- launch -----------------------------------------------------------------------------
    @classmethod
    def launch(cls, harness: str, model: str | Any, prompt: str, *, count: int,
               goal: bool = False, timeout: float | None = None, proxy: str | None = None,
               effort: str | None = None, shared_volume: str | None = None) -> "Swarm":
        """Deploy + pre-warm the model, then create ``count`` modal sandboxes and start a run in each."""

        if count < 1:
            raise AomError("swarm count must be >= 1.")

        resolved = models.get(model)
        endpoint = model_ops.deploy(resolved)
        # Pre-warm so the first wave of agents does not all hit a scaled-to-zero server (503s).
        model_ops.scale(
            resolved,
            min_containers=min(count, PREWARM_MIN),
            max_containers=max(count, PREWARM_MIN),
        )

        swarm_id = sandbox.new_id("swarm-")
        shared = shared_volume or f"aom-shared-{swarm_id}"

        def _launch_one(idx: int) -> Agent:
            last: Exception | None = None
            for attempt in range(CREATE_RETRIES):
                try:
                    box = sandbox.create(
                        "modal", endpoint=endpoint, proxy=proxy,
                        swarm=swarm_id, shared_volume=shared,
                    )
                    run = box.start(
                        harness, endpoint, prompt, goal=goal, timeout=timeout,
                        effort=effort, meta={"swarm_id": swarm_id, "idx": idx},
                    )
                    return Agent(sandbox=box, run=run, idx=idx)
                except SandboxError as error:
                    last = error
                    if attempt < CREATE_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF * (attempt + 1))
            raise AomError(f"swarm {swarm_id}: agent {idx} failed to launch: {last}")

        workers = max(1, min(count, MAX_CONCURRENCY))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            agents = list(pool.map(_launch_one, range(count)))
        agents.sort(key=lambda a: (a.idx if a.idx is not None else 0))
        return cls(id=swarm_id, model=resolved.name, shared_volume=shared, _agents=agents)

    # ---- rediscovery (stateless) ------------------------------------------------------------
    @classmethod
    def reconnect(cls, swarm_id: str) -> "Swarm":
        """Rebuild a swarm from its id by listing live sandboxes and their runs (no local state)."""

        boxes = sandbox.list_sandboxes("modal", swarm=swarm_id)
        agents: list[Agent] = []
        model_name = ""
        for box in boxes:
            runs = box.runs()
            run = runs[-1] if runs else None
            if run is not None and not model_name:
                try:
                    model_name = str(run.spec.get("model", "")) or model_name
                except Exception:  # noqa: BLE001 - spec may be unreadable mid-flight
                    pass
            idx = None
            labels = box.labels()
            if labels.get("idx", "").isdigit():
                idx = int(labels["idx"])
            agents.append(Agent(sandbox=box, run=run, idx=idx))
        agents.sort(key=lambda a: (a.idx if a.idx is not None else 0, a.id))
        return cls(id=swarm_id, model=model_name,
                   shared_volume=f"aom-shared-{swarm_id}", _agents=agents)

    def agents(self) -> list[Agent]:
        return list(self._agents)

    # ---- shared volume (read from outside Modal) --------------------------------------------
    def _volume(self) -> Any:
        import modal

        return modal.Volume.from_name(self.shared_volume, version=2)

    def list_shared(self, path: str = "/", *, recursive: bool = True) -> list[str]:
        """Paths committed to the shared Volume (read from outside, no container)."""

        try:
            return [getattr(entry, "path", str(entry))
                    for entry in self._volume().listdir(path, recursive=recursive)]
        except Exception as error:  # noqa: BLE001 - missing volume / not committed yet
            raise AomError(f"Could not list shared volume {self.shared_volume!r}: {error}") from error

    def read_shared(self, path: str) -> bytes:
        """Bytes of one file on the shared Volume (read from outside, no container)."""

        try:
            return b"".join(self._volume().read_file(path))
        except FileNotFoundError:
            raise
        except Exception as error:  # noqa: BLE001
            raise AomError(f"Could not read {path!r} from {self.shared_volume!r}: {error}") from error

    # ---- teardown ---------------------------------------------------------------------------
    def stop(self) -> None:
        """Terminate every agent sandbox and let the model scale back to zero (idempotent)."""

        for agent in self.agents():
            try:
                agent.sandbox.stop()
            except SandboxError:
                pass
        if self.model:
            try:
                model_ops.scale(models.get(self.model), min_containers=0)
            except AomError:
                pass

    def __repr__(self) -> str:
        return f"<Swarm {self.id} model={self.model} agents={len(self._agents)}>"
