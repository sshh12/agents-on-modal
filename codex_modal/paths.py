"""Project-local paths used by codex-modal."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _state_root() -> Path:
    """Allow the Docker sandbox to place wrapper state on a writable volume."""

    override = os.environ.get("CODEX_MODAL_STATE_ROOT")
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / ".codex-modal"


STATE_ROOT = _state_root()
CODEX_HOME = STATE_ROOT / "codex-home"
RUNTIME_ROOT = STATE_ROOT / "runtime"
CATALOG_PATH = PROJECT_ROOT / "catalog.json"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
CREDENTIALS_PATH = STATE_ROOT / "credentials.json"


def caller_cwd() -> Path:
    """Return the directory from which the shell launcher was invoked."""

    raw = os.environ.get("CODEX_MODAL_CALLER_CWD")
    if raw:
        path = Path(raw).expanduser()
        if path.is_dir():
            return path.resolve()
    return Path.cwd().resolve()


def ensure_state_dirs() -> None:
    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
