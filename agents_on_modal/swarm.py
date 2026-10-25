"""Swarm orchestrator: N Modal sandboxes running the same harness/model/prompt.

A *swarm* is a fan-out. :meth:`Swarm.launch` ensures the (shared, account-level) model is deployed
(idempotent; it does not scale it -- Modal autoscales the shared endpoint from aggregate load under
the catalog policy), then creates ``count`` Modal sandboxes tagged ``swarm=<id>`` and starts one
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

import agents_on_modal.models.lifecycle as model_ops

from . import models, sandbox
from .config import AomError
from .sandbox.base import Run, Sandbox, SandboxError

# ``model_ops`` is the lifecycle submodule (deploy/scale/endpoint/...); aliased so tests can
# monkeypatch it. ``models.get`` is the catalog lookup on the package.

#: Bounded concurrency and retry policy for the create/start fan-out.
MAX_CONCURRENCY = 16
CREATE_RETRIES = 5
RETRY_BACKOFF = 2.0


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
        """Ensure the shared model is deployed, then create ``count`` modal sandboxes and start a run in each."""

        if count < 1:
            raise AomError("swarm count must be >= 1.")

        resolved = models.get(model)
        # Just ensure the shared model is deployed (idempotent). We do NOT scale it for this swarm:
        # the model is an account-level shared resource and Modal's concurrency autoscaler sizes it
        # from AGGREGATE load across every consumer, under the conservative catalog policy
        # (``Scaling``). Scaling per launch would stomp that global setting and provision GPUs by
        # agent count instead of by real request load.
        endpoint = model_ops.deploy(resolved)

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
        """Terminate every agent sandbox (idempotent).

        The shared model is left alone: it is an account-level resource other swarms/apps may be
        using, and it scales back to zero on its own once AGGREGATE demand is idle (``min=0`` +
        ``scaledown_window``). Forcing its ``min_containers`` here would wrongly override the global
        policy and could tear capacity out from under another consumer.
        """

        for agent in self.agents():
            try:
                agent.sandbox.stop()
            except SandboxError:
                pass

    def __repr__(self) -> str:
        return f"<Swarm {self.id} model={self.model} agents={len(self._agents)}>"
