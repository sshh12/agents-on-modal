"""Offline unit tests for the swarm dashboard presets module.

No network, no Modal: encrypt/decrypt round-trip, loading presets.json, and that an
encrypted preset is unreadable without the right key and readable with it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SWARM = Path(__file__).resolve().parent.parent / "examples" / "swarm"


def _load_presets_module():
    """Import examples/swarm/presets.py in isolation (it is not an installed package)."""

    spec = importlib.util.spec_from_file_location("swarm_presets", SWARM / "presets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_presets_module()


def test_encrypt_decrypt_roundtrip():
    key = "correct horse battery staple"
    text = "Research something with émojis 🚀 and unicode ünïcödé."
    ct = P.encrypt(text, key)
    assert ct != text
    assert P.decrypt({"scheme": P.SCHEME, "ciphertext": ct}, key) == text


def test_decrypt_wrong_key_fails():
    ct = P.encrypt("secret prompt", "the-right-key")
    entry = {"scheme": P.SCHEME, "ciphertext": ct}
    with pytest.raises(P.PresetError):
        P.decrypt(entry, "the-wrong-key-entirely-different")


def test_decrypt_unknown_scheme():
    with pytest.raises(P.PresetError):
        P.decrypt({"scheme": "aes-256-gcm", "ciphertext": "x"}, "k")


def test_encrypt_requires_key():
    with pytest.raises(P.PresetError):
        P.encrypt("x", "")


def test_presets_json_loads_and_has_both_forms():
    entries = P.load_presets()
    assert isinstance(entries, list) and entries
    assert any("prompt" in e and "enc" not in e for e in entries), "need a plaintext preset"
    assert any("enc" in e for e in entries), "need an encrypted preset"
    for e in entries:
        assert e.get("name") and e.get("mode") in ("naive", "dynamic")
    for e in entries:
        if "enc" in e:
            assert e["enc"]["scheme"] == P.SCHEME
            assert e["enc"]["key"].startswith("env:")
            assert e["enc"]["ciphertext"]


def test_plaintext_preset_resolves_without_key():
    resolved = P.resolve_presets(env={})
    plain = [p for p in resolved if not p["locked"]]
    assert plain, "at least one plaintext preset should resolve with no key"
    assert all("prompt" in p for p in plain)


def test_encrypted_preset_locked_without_key_readable_with_it(tmp_path):
    key = "a-test-passphrase"
    text = "The secret OSINT workflow prompt."
    config = {
        "version": 1,
        "presets": [
            {"name": "plain", "label": "Plain", "mode": "naive", "prompt": "hello"},
            {
                "name": "sealed",
                "label": "Sealed",
                "mode": "dynamic",
                "enc": {
                    "scheme": P.SCHEME,
                    "key": "env:AOM_SWARM_PRESET_KEY",
                    "ciphertext": P.encrypt(text, key),
                },
            },
        ],
    }
    path = tmp_path / "presets.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    # Without the key: encrypted entry is locked and carries no cleartext.
    without = {p["name"]: p for p in P.resolve_presets(path=path, env={})}
    assert without["sealed"]["locked"] is True
    assert "prompt" not in without["sealed"]
    assert without["plain"]["locked"] is False and without["plain"]["prompt"] == "hello"

    # With the key: it decrypts.
    with_key = {p["name"]: p for p in P.resolve_presets(
        path=path, env={"AOM_SWARM_PRESET_KEY": key})}
    assert with_key["sealed"]["locked"] is False
    assert with_key["sealed"]["prompt"] == text


def test_real_osint_preset_decrypts_with_env_key(monkeypatch):
    """If the local .env key is present in the environment, the shipped OSINT preset decrypts."""

    import os

    key = os.environ.get("AOM_SWARM_PRESET_KEY")
    # Fall back to reading the gitignored .env so the test is meaningful in a fresh shell.
    if not key:
        env_file = SWARM.parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("AOM_SWARM_PRESET_KEY="):
                    key = line.split("=", 1)[1]
                    break
    if not key:
        pytest.skip("AOM_SWARM_PRESET_KEY not available (no .env)")

    resolved = {p["name"]: p for p in P.resolve_presets(env={"AOM_SWARM_PRESET_KEY": key})}
    assert "osint" in resolved
    assert resolved["osint"]["locked"] is False
    assert "Shrivu Shankar" in resolved["osint"]["prompt"]
