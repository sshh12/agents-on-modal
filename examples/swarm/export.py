"""Zip a swarm's shared Volume on Modal's side, for the dashboard's "download .zip" button.

Listing and reading a Volume from the host costs one RPC per directory and per file (minutes for a
few thousand files), so this zips next to the data instead: a short-lived, CPU-only,
network-blocked sandbox under an ephemeral app mounts the Volume read-only and runs ``volzip.py``
(stdlib only, shipped as source) over it, streaming the zip back on stdout. It needs only the
Volume, so it works after the swarm's sandboxes (and the ``aom-sandboxes`` app) are gone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

from agents_on_modal.config import AomError

#: The ephemeral app an export sandbox runs under; it stops when the export returns.
EXPORT_APP = "aom-volume-export"
#: Wall-clock cap for one export sandbox (an ``all_files`` export of a large volume).
EXPORT_TIMEOUT = 30 * 60
_EXPORT_MOUNT = "/v"
_WALKER = Path(__file__).with_name("volzip.py")


def zip_shared_volume(name: str, dest: BinaryIO, *, all_files: bool = False) -> dict:
    """Zip the shared Volume ``name`` on Modal's side and write the zip into ``dest``.

    By default only ``volzip.DEFAULT_EXTS`` files are kept; ``all_files`` keeps every file.
    Dependency/cache dirs are pruned either way. Returns volzip's counts plus ``zip_bytes``; raises
    :class:`AomError` on a missing Volume or a failed walk.
    """

    import modal

    try:
        volume = modal.Volume.from_name(name, version=2)
        volume.hydrate()  # a missing volume fails here, clearly, not inside the sandbox
    except Exception as error:  # noqa: BLE001
        raise AomError(f"No shared volume {name!r}: {error}") from error

    source = _WALKER.read_text(encoding="utf-8")
    argv = ["python3", "-c", source, _EXPORT_MOUNT, "-"] + (["--all"] if all_files else [])
    written = 0
    app = modal.App(EXPORT_APP)
    with app.run():
        box = modal.Sandbox.create(
            app=app, image=modal.Image.debian_slim(), timeout=EXPORT_TIMEOUT, block_network=True,
            cpu=2.0, memory=2048, volumes={_EXPORT_MOUNT: volume.with_mount_options(read_only=True)},
        )
        try:
            proc = box.exec(*argv, text=False)
            for chunk in proc.stdout:
                dest.write(chunk)
                written += len(chunk)
            err = proc.stderr.read()
            code = proc.wait()
        finally:
            box.terminate()
    lines = (err.decode("utf-8", "replace") if isinstance(err, bytes) else str(err)).strip().splitlines()
    if code != 0 or not lines:
        raise AomError(f"Zipping {name!r} failed (exit {code}): {lines[-1] if lines else 'no output'}")
    stats = json.loads(lines[-1])
    stats["zip_bytes"] = written
    return stats
