"""The model catalog: one ``catalog.json`` entry fully describes one Modal deployment.

    from agents_on_modal import models
    model = models.get("qwen3.8-27b")
    endpoint = models.deploy(model)          # idempotent; weights prepared on CPU first
    models.wait_ready(endpoint)
    models.scale(model, min_containers=2, max_containers=16)   # live, no redeploy

``deploy``/``endpoint``/``wait_ready``/``scale``/``stop``/``status`` live in ``deploy.py`` and are
re-exported here lazily so importing the catalog never imports Modal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import AomError

CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog.json"

#: Normalized effort scale every harness effort is mapped onto. ``ultra`` (Codex) is ``max``.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
ENGINES = ("sglang", "llamacpp")
REGIONS = ("us-east", "us-west", "ca-central", "eu-west", "ap-south", "ap-southeast-2")
NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,38}[a-z0-9])?$")
GPU_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_-]{0,31})(?::([1-8]))?$")


def normalize_effort(value: str | None) -> str | None:
    """Map any harness effort spelling onto ``EFFORTS`` (``None`` for unknown/absent)."""

    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered == "ultra":
        return "max"
    return lowered if lowered in EFFORTS else None


@dataclass(frozen=True)
class Reasoning:
    """How a model reasons and how normalized efforts reach the engine.

    ``efforts[e]`` is a string (send ``reasoning_effort=<value>`` with thinking on), ``True``
    (thinking on, no effort parameter) or ``False`` (thinking off). ``kwarg`` is the chat-template
    switch for thinking (``thinking`` / ``enable_thinking``) or ``None`` when the template has none.
    """

    default: str = "high"
    efforts: dict[str, str | bool] = field(default_factory=dict)
    kwarg: str | None = None

    def engine_value(self, effort: str | None) -> str | bool:
        """The engine-side value for a (possibly unnormalized) harness effort."""

        normalized = normalize_effort(effort) or self.default
        if normalized in self.efforts:
            return self.efforts[normalized]
        return self.efforts.get(self.default, True)


@dataclass(frozen=True)
class Scaling:
    """Modal autoscaler settings for the model's GPU server."""

    min_containers: int = 0
    max_containers: int = 1
    buffer_containers: int = 0
    scaledown_window: int = 900
    target_concurrency: float = 32.0

    def replace(self, **overrides: Any) -> "Scaling":
        values = {**self.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
        return Scaling(**values)


@dataclass(frozen=True)
class Model:
    """One deployable model. ``name`` is the catalog key, the served model id, and the app suffix."""

    name: str
    description: str
    repo: str
    engine: str
    gpu: str
    context_window: int
    max_output_tokens: int
    reasoning: Reasoning
    scaling: Scaling = Scaling()
    revision: str | None = None
    image: str | None = None
    engine_args: tuple[str, ...] = ()
    engine_env: dict[str, str] = field(default_factory=dict)
    serving_pip: tuple[str, ...] = ()
    download_subdir: str | None = None
    sampling: dict[str, float] = field(default_factory=dict)
    region: str = "us-west"
    hf_token_env: str | None = None
    vision: bool = False
    startup_timeout: int = 3600
    cpu: int = 8
    memory: int = 98_304

    @property
    def app_name(self) -> str:
        return f"aom-{self.name.replace('.', '-')}"

    @property
    def volume_name(self) -> str:
        return f"{self.app_name}-weights"

    @property
    def gpu_count(self) -> int:
        match = GPU_PATTERN.fullmatch(self.gpu)
        return int(match.group(2) or "1") if match else 1

    def to_json(self) -> dict[str, Any]:
        """A JSON-safe dict (used to ship the model spec into the Modal app and the gateway)."""

        data = dict(self.__dict__)
        data["reasoning"] = dict(self.reasoning.__dict__)
        data["scaling"] = dict(self.scaling.__dict__)
        data["engine_args"] = list(self.engine_args)
        data["serving_pip"] = list(self.serving_pip)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Model":
        values = dict(data)
        values["reasoning"] = Reasoning(**values.get("reasoning", {}))
        values["scaling"] = Scaling(**values.get("scaling", {}))
        values["engine_args"] = tuple(values.get("engine_args", ()))
        values["serving_pip"] = tuple(values.get("serving_pip", ()))
        known = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise AomError(f"Model {values.get('name')!r} has unknown fields: {', '.join(unknown)}")
        model = cls(**values)
        _validate(model)
        return model


def _validate(model: Model) -> None:
    where = f"catalog model {model.name!r}"
    if not NAME_PATTERN.fullmatch(model.name):
        raise AomError(f"{where}: names are lowercase [a-z0-9.-], at most 40 chars.")
    if model.engine not in ENGINES:
        raise AomError(f"{where}: engine must be one of {', '.join(ENGINES)}.")
    if not GPU_PATTERN.fullmatch(model.gpu):
        raise AomError(f"{where}: gpu must look like H100 or B200:2.")
    if model.gpu_count & (model.gpu_count - 1):
        raise AomError(f"{where}: GPU count must be a power of two (tensor parallelism).")
    if model.region not in REGIONS:
        raise AomError(f"{where}: region must be one of {', '.join(REGIONS)}.")
    if model.context_window < 8192 or model.max_output_tokens >= model.context_window:
        raise AomError(f"{where}: need context_window >= 8192 and max_output_tokens < context_window.")
    if model.reasoning.default not in EFFORTS:
        raise AomError(f"{where}: reasoning.default must be one of {', '.join(EFFORTS)}.")
    bad = sorted(set(model.reasoning.efforts) - set(EFFORTS))
    if bad:
        raise AomError(f"{where}: unknown reasoning efforts {bad}.")
    if model.engine == "llamacpp" and not model.download_subdir:
        raise AomError(f"{where}: llamacpp models need download_subdir (the GGUF quant folder).")
    for arg in model.engine_args:
        if not arg.startswith("--"):
            raise AomError(f"{where}: engine_args entries must be --flag or --flag=value ({arg!r}).")


def load_catalog(path: str | Path | None = None) -> dict[str, Model]:
    """Parse a catalog file (default: the packaged ``catalog.json``) into ``{name: Model}``."""

    source = Path(path) if path else CATALOG_PATH
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AomError(f"Cannot read model catalog {source}: {error}") from error
    entries = document.get("models") if isinstance(document, dict) else None
    if not isinstance(entries, dict):
        raise AomError(f"{source} must be an object with a 'models' mapping.")
    return {name: Model.from_json({"name": name, **spec}) for name, spec in entries.items()}


def list_models(path: str | Path | None = None) -> list[Model]:
    return list(load_catalog(path).values())


def get(name: str | Model, path: str | Path | None = None) -> Model:
    """Look up a model by catalog name (a ``Model`` passes through)."""

    if isinstance(name, Model):
        return name
    catalog = load_catalog(path)
    if name not in catalog:
        raise AomError(f"Unknown model {name!r}. Known: {', '.join(catalog)}")
    return catalog[name]


@dataclass(frozen=True)
class Endpoint:
    """A deployed model as seen from outside Modal: gateway base URL + Modal proxy token."""

    model: Model
    url: str
    token: str

    @property
    def openai_base_url(self) -> str:
        return f"{self.url}/v1"


def __getattr__(name: str) -> Any:
    if name in {"deploy", "endpoint", "wait_ready", "scale", "stop", "status"}:
        from . import deploy as _deploy

        return getattr(_deploy, name)
    raise AttributeError(name)
