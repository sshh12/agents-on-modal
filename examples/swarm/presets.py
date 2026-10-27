#!/usr/bin/env python3
"""Swarm dashboard presets: load ``presets.json`` and (de)obfuscate encrypted prompts.

This is a tiny, stdlib-only config layer for the swarm dashboard. Presets live in
``presets.json`` next to this file. A preset's prompt can be stored two ways:

  * **plaintext** -- a ``"prompt"`` string, readable by anyone with the repo.
  * **encrypted** -- an ``"enc"`` object holding ciphertext plus the scheme and the
    *key source* used to decrypt it. The cleartext is never in the file.

Encrypted-entry shape::

    {
      "name": "osint",
      "label": "OSINT footprint",
      "mode": "dynamic",
      "enc": {
        "scheme": "sha256-xor-b64-v1",   # the (de)obfuscation algorithm
        "key": "env:AOM_SWARM_PRESET_KEY", # where the passphrase comes from
        "ciphertext": "<base64>"          # the obfuscated UTF-8 prompt
      }
    }

Key sources are expressed as a ``"<source>:<ref>"`` string so the schema is
extensible. Today only ``env:<VAR>`` is implemented (the passphrase lives in a
gitignored ``.env`` / the process environment, never in the repo). A *future*
variant where the password sits right next to the ciphertext -- trivially
decryptable, no env needed -- slots in as e.g. ``"key": "inline:"`` with the
passphrase carried in a sibling field (``"password": "..."``); ``_resolve_key``
would learn that one new prefix and everything else stays the same.

.. warning::
   The ``sha256-xor-b64-v1`` scheme is **obfuscation, not real cryptography**.
   It is a reversible XOR keystream, meant only to keep a prompt out of plain
   sight in the repo. Do not use it to protect anything that actually matters.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESETS_FILE = HERE / "presets.json"

#: Env var that carries the passphrase for ``env:``-sourced encrypted presets.
ENV_KEY_VAR = "AOM_SWARM_PRESET_KEY"

SCHEME = "sha256-xor-b64-v1"


class PresetError(Exception):
    """A preset could not be loaded or decrypted."""


# --------------------------------------------------------------------------------------------- #
# obfuscation (NOT real crypto): a sha256-derived XOR keystream, base64-wrapped.                #
# --------------------------------------------------------------------------------------------- #
def _keystream(key: str, n: int) -> bytes:
    """Deterministic ``n``-byte keystream from ``key``: sha256(key || counter) blocks."""

    key_bytes = key.encode("utf-8")
    out = bytearray()
    counter = 0
    while len(out) < n:
        block = hashlib.sha256(key_bytes + counter.to_bytes(8, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:n])


def _xor(data: bytes, key: str) -> bytes:
    stream = _keystream(key, len(data))
    return bytes(b ^ k for b, k in zip(data, stream))


def encrypt(text: str, key: str) -> str:
    """Obfuscate ``text`` with ``key`` and return base64 ciphertext (reverse of :func:`decrypt`)."""

    if not key:
        raise PresetError("encrypt requires a non-empty key")
    return base64.b64encode(_xor(text.encode("utf-8"), key)).decode("ascii")


def decrypt(entry: dict, key: str) -> str:
    """Return the cleartext prompt for an encrypted ``entry`` using ``key``.

    ``entry`` is the ``"enc"`` object (``scheme``/``ciphertext``). Raises
    :class:`PresetError` on an unknown scheme, a missing/empty key, or ciphertext
    that does not decode as UTF-8 (i.e. the wrong key).
    """

    scheme = entry.get("scheme")
    if scheme != SCHEME:
        raise PresetError(f"unsupported scheme: {scheme!r}")
    if not key:
        raise PresetError("decrypt requires a non-empty key")
    try:
        raw = base64.b64decode(entry["ciphertext"], validate=True)
    except Exception as exc:  # noqa: BLE001
        raise PresetError(f"invalid ciphertext: {exc}") from exc
    try:
        return _xor(raw, key).decode("utf-8")
    except UnicodeDecodeError as exc:
        # Wrong key -> XOR produces non-UTF-8 bytes. Surface as a decrypt failure.
        raise PresetError("could not decrypt (wrong key?)") from exc


# --------------------------------------------------------------------------------------------- #
# key resolution: "<source>:<ref>" -> passphrase                                                #
# --------------------------------------------------------------------------------------------- #
def _resolve_key(enc: dict, env: dict | None = None) -> str | None:
    """Resolve the passphrase for an ``enc`` object, or ``None`` if unavailable.

    Supported sources:
      * ``env:<VAR>``  -- read ``<VAR>`` from ``env`` (defaults to ``os.environ``).

    Future (documented, not yet implemented): ``inline:`` with the passphrase in a
    sibling ``"password"`` field, for trivially-decryptable presets.
    """

    env = os.environ if env is None else env
    source = enc.get("key") or ""
    if source.startswith("env:"):
        return env.get(source[len("env:") :]) or None
    # Unknown / not-yet-implemented source (e.g. a future "inline:").
    return None


# --------------------------------------------------------------------------------------------- #
# loading                                                                                       #
# --------------------------------------------------------------------------------------------- #
def load_presets(path: Path | None = None) -> list[dict]:
    """Return the raw preset entries from ``presets.json`` (no decryption)."""

    path = PRESETS_FILE if path is None else Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    presets = data.get("presets", []) if isinstance(data, dict) else data
    if not isinstance(presets, list):
        raise PresetError("presets.json must contain a list of presets")
    return presets


def resolve_presets(path: Path | None = None, env: dict | None = None) -> list[dict]:
    """Load presets and resolve each prompt for the UI.

    Returns a list of ``{name, label, mode, prompt?, locked}`` dicts:
      * plaintext entries carry their ``prompt`` and ``locked=False``;
      * encrypted entries we can decrypt carry the cleartext ``prompt`` and ``locked=False``;
      * encrypted entries we cannot decrypt (no/invalid key) carry ``locked=True`` and
        **no** ``prompt`` -- the endpoint never errors just because a key is missing.
    """

    out: list[dict] = []
    for entry in load_presets(path):
        item = {
            "name": entry.get("name") or entry.get("label") or "preset",
            "label": entry.get("label") or entry.get("name") or "preset",
            "mode": (entry.get("mode") or "naive").lower(),
        }
        if "enc" in entry:
            key = _resolve_key(entry["enc"], env=env)
            try:
                if not key:
                    raise PresetError("no key available")
                item["prompt"] = decrypt(entry["enc"], key)
                item["locked"] = False
            except PresetError:
                item["locked"] = True
        else:
            item["prompt"] = entry.get("prompt", "")
            item["locked"] = False
        out.append(item)
    return out
