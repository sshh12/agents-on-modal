#!/usr/bin/env python3
"""Zero-setup launch for the examples: everything one needs before it imports aom.

Stdlib only, so it runs on whatever ``python`` is on PATH. :func:`prepare` is idempotent; each step
is a no-op once satisfied, so a configured machine pays nothing:

  1. **.env** -- load the repo-root ``.env`` (``HF_TOKEN`` for deploying a model,
     ``AOM_SWARM_PRESET_KEY``, ...) without overriding variables already set (non-empty) in the
     environment.
  2. **interpreter** -- if this Python lacks ``modal`` at the version pyproject.toml pins (e.g. an
     older global install), re-launch on the repo's ``.venv``, creating it and ``pip install -e .``-ing
     on first run.
  3. **Modal login** -- no Modal credentials yet -> run ``modal setup`` (opens a browser).
  4. **proxy token** -- no aom proxy token stored yet -> mint one, exactly as ``aom setup`` does.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv"
#: Set on the re-launched child so a venv that still cannot satisfy the deps fails loudly, not in a loop.
_RELAUNCHED = "AOM_EXAMPLE_RELAUNCHED"


def _say(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- #
# 1. .env                                                                                       #
# --------------------------------------------------------------------------------------------- #
def load_dotenv(path: Path = REPO / ".env") -> list[str]:
    """Export each ``KEY=value`` in ``path`` that is not already set; return the keys exported.

    A variable set but empty counts as unset, so a blank ``HF_TOKEN`` doesn't hide the file's.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    loaded: list[str] = []
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or key.startswith("#"):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


# --------------------------------------------------------------------------------------------- #
# 2. interpreter                                                                                #
# --------------------------------------------------------------------------------------------- #
def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def deps_ok() -> bool:
    """True when this interpreter has ``modal`` inside pyproject.toml's pin (the dashboard's one dep)."""

    from importlib import metadata

    try:
        have = _version(metadata.version("modal"))
    except metadata.PackageNotFoundError:
        return False
    pin = re.search(r'"modal>=([\d.]+),<([\d.]+)"', (REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return pin is None or _version(pin[1]) <= have < _version(pin[2])


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_python(*, interactive: bool = False) -> None:
    """Return if this interpreter can run aom; otherwise re-launch on the repo ``.venv`` (no return).

    ``interactive``: the child is a TUI that reads Ctrl+C as input (interrupt the agent, not quit),
    so this pass-through parent ignores it and just waits, instead of killing a slow-to-exit child.
    """

    if deps_ok():
        return
    if Path(sys.prefix).resolve() == VENV.resolve():  # on .venv already, it just lacks the deps
        _say("installing agents-on-modal into .venv (first run, takes a minute) ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)])
        return
    if os.environ.get(_RELAUNCHED):
        raise SystemExit(f"{_venv_python()} still lacks a compatible `modal`; "
                         f"run `{_venv_python()} -m pip install -e {REPO}` and retry.")
    python = _venv_python()
    if not python.exists():
        _say(f"creating {VENV} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    _say(f"re-launching on {python}")
    os.environ[_RELAUNCHED] = "1"
    proc = subprocess.Popen([str(python), *sys.argv])
    if interactive:  # ignored only after Popen, so the child doesn't inherit SIG_IGN (POSIX)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise SystemExit(proc.wait())
    try:
        code = proc.wait()
    except KeyboardInterrupt:  # the child got the same Ctrl+C; let it shut its server down cleanly
        try:
            code = proc.wait(timeout=10)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            proc.kill()
            code = 130
    raise SystemExit(code)


# --------------------------------------------------------------------------------------------- #
# 3-4. Modal login + aom proxy token                                                            #
# --------------------------------------------------------------------------------------------- #
def _modal_logged_in() -> bool:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    config = Path(os.environ.get("MODAL_CONFIG_PATH") or Path.home() / ".modal.toml")
    try:
        return "token_id" in config.read_text(encoding="utf-8")
    except OSError:
        return False


def ensure_modal_login() -> None:
    if _modal_logged_in():
        return
    _say("no Modal credentials found -- running `modal setup` (opens a browser) ...")
    if subprocess.call([sys.executable, "-m", "modal", "setup"]) != 0 or not _modal_logged_in():
        raise SystemExit("Modal login did not complete; run `python -m modal setup` and retry.")


def ensure_proxy_token() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from agents_on_modal.config import AomError, load_token

    try:
        load_token()
        return
    except AomError:
        pass
    _say("no aom proxy token found -- minting one (same as `aom setup`) ...")
    from agents_on_modal.cli import main as aom

    if aom(["setup"]) != 0:
        raise SystemExit("could not mint a Modal proxy token; run `aom setup` and retry.")


def prepare(*, interactive: bool = False) -> None:
    """Run every setup step, in order (see the module docstring); ``interactive`` for TUI examples."""

    loaded = load_dotenv()
    if loaded:
        _say(f"loaded {', '.join(loaded)} from {REPO / '.env'}")
    ensure_python(interactive=interactive)
    ensure_modal_login()
    ensure_proxy_token()
