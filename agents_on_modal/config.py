"""User-level state: where aom keeps tokens, local sandbox homes, and exported logs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class AomError(RuntimeError):
    """Base class for every error the library raises on purpose."""


def ensure_utf8_io() -> None:
    """Make stdout/stderr UTF-8 so printing non-ASCII (``✓``, ``→``) never crashes a command.

    On Windows the console defaults to a legacy code page (cp1252); emitting a non-ASCII character
    then raises ``UnicodeEncodeError: 'charmap'`` and kills the process mid-output (issue S10). We
    reconfigure the streams to UTF-8 with ``errors="replace"`` (so a stray byte degrades to ``?``
    rather than aborting) and also export ``PYTHONIOENCODING`` so child processes inherit it. Called
    from every CLI entrypoint; cheap and idempotent, and a no-op where the streams are already UTF-8.
    """

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best effort; never let I/O setup break a command
            pass


def home() -> Path:
    """The aom state directory (``AOM_HOME`` or ``~/.agents-on-modal``), created on demand."""

    root = Path(os.environ.get("AOM_HOME") or Path.home() / ".agents-on-modal").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class ProxyToken:
    """A Modal workspace proxy token (``wk-…`` id + ``ws-…`` secret)."""

    token_id: str
    token_secret: str

    @property
    def bearer(self) -> str:
        return f"{self.token_id}.{self.token_secret}"

    @classmethod
    def parse(cls, value: str) -> "ProxyToken":
        token_id, _, secret = value.strip().partition(".")
        if not token_id.startswith("wk-") or not secret.startswith("ws-"):
            raise AomError("Expected a Modal proxy token of the form wk-<id>.ws-<secret>.")
        return cls(token_id, secret)


def _token_path() -> Path:
    return home() / "token.json"


def load_token() -> ProxyToken:
    """The user's Modal proxy token: ``MODAL_PROXY_TOKEN`` env, else ``~/.agents-on-modal/token.json``."""

    if os.environ.get("MODAL_PROXY_TOKEN"):
        return ProxyToken.parse(os.environ["MODAL_PROXY_TOKEN"])
    path = _token_path()
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProxyToken(data["token_id"], data["token_secret"])
    raise AomError("No Modal proxy token. Run `aom setup` or set MODAL_PROXY_TOKEN=wk-<id>.ws-<secret>.")


def save_token(token: ProxyToken) -> Path:
    path = _token_path()
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"token_id": token.token_id, "token_secret": token.token_secret}, stream)
    os.replace(temporary, path)
    return path
