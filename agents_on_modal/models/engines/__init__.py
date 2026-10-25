"""Engine registry: ``get(name) -> Engine``.

An engine bundles the three per-engine serving decisions (image, download, launch)
behind one small interface, so adding an engine is a new ``Engine`` subclass here,
not an ``if`` threaded through ``serve.py`` and ``deploy.py``.
"""

from __future__ import annotations

from ...config import AomError
from .base import BASE_RUNTIME_ENV, DownloadSpec, Engine, gpu_count
from .llamacpp import LlamaCppEngine
from .sglang import SGLangEngine

#: One shared, stateless instance per engine (engines hold no mutable state).
_ENGINES: dict[str, Engine] = {
    SGLangEngine.name: SGLangEngine(),
    LlamaCppEngine.name: LlamaCppEngine(),
}


def get(name: str) -> Engine:
    """Look up an engine by its registry key (``sglang`` / ``llamacpp``)."""

    try:
        return _ENGINES[str(name).lower()]
    except KeyError:
        raise AomError(
            f"Unknown engine {name!r}. Known engines: {', '.join(sorted(_ENGINES))}."
        ) from None


def names() -> tuple[str, ...]:
    return tuple(sorted(_ENGINES))


__all__ = [
    "BASE_RUNTIME_ENV",
    "DownloadSpec",
    "Engine",
    "SGLangEngine",
    "LlamaCppEngine",
    "get",
    "gpu_count",
    "names",
]
