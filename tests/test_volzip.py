"""Offline unit tests for the swarm dashboard's volume export (``examples/swarm``).

No Modal. ``volzip`` (the walker) runs against a temp tree, both in-process and the way
``export`` ships it into the sandbox: its source passed to ``python -c``, the zip streamed to stdout.
The ``export`` tests swap in a fake ``modal`` whose sandbox runs that same command locally.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from agents_on_modal.config import AomError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "swarm"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import export  # noqa: E402 - examples/swarm/export.py
import volzip  # noqa: E402 - examples/swarm/volzip.py

SOURCE = Path(volzip.__file__).read_text(encoding="utf-8")


def _tree(root: Path) -> None:
    files = {
        "README.md": "# hi",
        "board/agent-1.MD": "upper-case suffix",
        "data/results.csv": "a,b\n1,2",
        "data/results.tsv": "a\tb",
        "data/state.json": "{}",
        "data/notes.txt": "not an output",
        "_swarm/traces/x.jsonl": "{}\n",
        "app/node_modules/pkg/package.json": "{}",
        "app/.git/config.json": "{}",
        "app/src/__pycache__/m.json": "{}",
        "app/src/report.md": "deep",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _names(blob: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.namelist()


def test_default_keeps_outputs_and_prunes_dependency_dirs(tmp_path):
    _tree(tmp_path)
    out = io.BytesIO()
    stats = volzip.zip_tree(str(tmp_path), out)

    assert _names(out.getvalue()) == [
        "README.md", "app/src/report.md", "board/agent-1.MD",
        "data/results.csv", "data/results.tsv", "data/state.json",
    ]
    assert stats["files"] == 6
    assert stats["skipped_dirs"] == 3   # node_modules, .git, __pycache__
    assert stats["errors"] == 0
    with zipfile.ZipFile(out) as zf:
        assert zf.read("app/src/report.md") == b"deep"
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in zf.infolist())


def test_all_files_keeps_everything_but_pruned_dirs(tmp_path):
    _tree(tmp_path)
    out = io.BytesIO()
    stats = volzip.zip_tree(str(tmp_path), out, exts=None)
    names = _names(out.getvalue())

    assert "data/notes.txt" in names and "_swarm/traces/x.jsonl" in names
    assert not any(part in n for n in names for part in ("node_modules", ".git/", "__pycache__"))
    assert stats["files"] == len(names) == 8


def test_large_files_stream_instead_of_prefetch(tmp_path, monkeypatch):
    monkeypatch.setattr(volzip, "PREFETCH_MAX_BYTES", 4)
    (tmp_path / "big.md").write_bytes(b"x" * 1000)
    (tmp_path / "tiny.md").write_bytes(b"ok")
    out = io.BytesIO()
    stats = volzip.zip_tree(str(tmp_path), out)
    with zipfile.ZipFile(out) as zf:
        assert zf.read("big.md") == b"x" * 1000
        assert zf.read("tiny.md") == b"ok"
    assert stats == {"files": 2, "bytes": 1002, "skipped_dirs": 0, "errors": 0}


def test_symlinks_are_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("no", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.md").write_text("yes", encoding="utf-8")
    try:
        os.symlink(outside, root / "linkdir", target_is_directory=True)
        os.symlink(outside / "secret.md", root / "link.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable here")
    out = io.BytesIO()
    volzip.zip_tree(str(root), out)
    assert _names(out.getvalue()) == ["real.md"]


@pytest.mark.parametrize("extra, expected", [([], 6), (["--all"], 8)])
def test_shipped_source_streams_zip_to_stdout(tmp_path, extra, expected):
    # Exactly how export.zip_shared_volume runs it in the sandbox: source via -c, zip on stdout.
    _tree(tmp_path)
    proc = subprocess.run([sys.executable, "-c", SOURCE, str(tmp_path), "-", *extra],
                          capture_output=True, check=True)
    stats = json.loads(proc.stderr.decode().strip().splitlines()[-1])
    assert stats["files"] == expected
    assert len(_names(proc.stdout)) == expected


def _fake_export_modal(root, *, missing=False, exit_code=None):
    """A fake ``modal`` whose sandbox runs the shipped walker locally, with ``/v`` mapped to ``root``."""

    import contextlib
    import subprocess
    import sys
    import types

    state = {"created": [], "terminated": 0, "runs": []}

    class FakeVolume:
        def __init__(self, name, read_only=False):
            self.name, self.read_only = name, read_only

        def hydrate(self):
            if missing:
                raise RuntimeError("Volume not found")
            return self

        def with_mount_options(self, read_only=None):
            return FakeVolume(self.name, read_only=bool(read_only))

    class FakeStream:
        def __init__(self, data):
            self._data = data

        def __iter__(self):
            return iter([self._data[i:i + 1000] for i in range(0, len(self._data), 1000)])

        def read(self):
            return self._data

    class FakeProc:
        def __init__(self, argv):
            argv = [sys.executable if a == "python3" else (str(root) if a == "/v" else a) for a in argv]
            done = subprocess.run(argv, capture_output=True)
            self.stdout, self.stderr = FakeStream(done.stdout), FakeStream(done.stderr)
            self._code = done.returncode if exit_code is None else exit_code

        def wait(self):
            return self._code

    class FakeSandbox:
        def exec(self, *argv, text=True):
            assert text is False  # the zip is binary
            state["runs"].append(argv)
            return FakeProc(argv)

        def terminate(self):
            state["terminated"] += 1

    class FakeApp:
        def __init__(self, name):
            self.name = name

        @contextlib.contextmanager
        def run(self):
            state["app_running"] = True
            yield self
            state["app_running"] = False

    def create(*args, **kwargs):
        assert state.get("app_running"), "sandbox must run under the live ephemeral app"
        state["created"].append(kwargs)
        return FakeSandbox()

    fake = types.SimpleNamespace(
        Volume=types.SimpleNamespace(from_name=lambda name, version=2: FakeVolume(name)),
        App=FakeApp, Image=types.SimpleNamespace(debian_slim=lambda: "slim"),
        Sandbox=types.SimpleNamespace(create=create),
    )
    return fake, state


def _shared_tree(root):
    for rel in ("report.md", "data/rows.csv", "notes.txt", "node_modules/p/package.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(rel, encoding="utf-8")


@pytest.mark.parametrize("all_files, names", [
    (False, ["data/rows.csv", "report.md"]),
    (True, ["data/rows.csv", "notes.txt", "report.md"]),
])
def test_zip_shared_runs_the_walker_in_a_read_only_sandbox(monkeypatch, tmp_path, all_files, names):
    import io
    import zipfile

    _shared_tree(tmp_path)
    fake, state = _fake_export_modal(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "modal", fake)

    out = io.BytesIO()
    stats = export.zip_shared_volume("aom-shared-s1", out, all_files=all_files)

    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == names
    assert stats["files"] == len(names)
    assert stats["zip_bytes"] == len(out.getvalue())
    (kwargs,) = state["created"]
    assert kwargs["block_network"] is True and "gpu" not in kwargs
    assert kwargs["volumes"]["/v"].name == "aom-shared-s1"
    assert kwargs["volumes"]["/v"].read_only is True
    assert state["runs"][0][:2] == ("python3", "-c")
    assert ("--all" in state["runs"][0]) is all_files
    assert state["terminated"] == 1 and state["app_running"] is False


def test_zip_shared_missing_volume_raises_before_any_sandbox(monkeypatch, tmp_path):
    import io

    fake, state = _fake_export_modal(tmp_path, missing=True)
    monkeypatch.setitem(__import__("sys").modules, "modal", fake)
    with pytest.raises(AomError, match="No shared volume"):
        export.zip_shared_volume("aom-shared-gone", io.BytesIO())
    assert state["created"] == []


def test_zip_shared_failed_walk_raises_and_still_terminates(monkeypatch, tmp_path):
    import io

    _shared_tree(tmp_path)
    fake, state = _fake_export_modal(tmp_path, exit_code=1)
    monkeypatch.setitem(__import__("sys").modules, "modal", fake)
    with pytest.raises(AomError, match="failed"):
        export.zip_shared_volume("aom-shared-s1", io.BytesIO())
    assert state["terminated"] == 1
