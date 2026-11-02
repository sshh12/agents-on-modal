"""Zip a directory tree (a mounted shared Volume) -- stdlib only, so it runs anywhere.

:func:`zip_tree` is the walker. ``export.zip_shared_volume`` ships this module's source into a
short-lived CPU sandbox that mounts the Volume read-only and runs it there
(``python3 -c <source> <root> -``), so the walk happens next to the data instead of as one Volume RPC
per directory and per file from the host. The zip streams to stdout; one JSON stats line goes to
stderr.

On a Volume mount every directory listing and first file read is a network round trip, so a plain
``os.walk`` + sequential reads is latency-bound (minutes for a few thousand dirs). Both phases are
therefore fanned out over threads: a parallel ``scandir`` walk, then reads prefetched through a
bounded window while the zip is written, in order, on the main thread.

Keep this module free of non-stdlib and package-relative imports: it is executed as a bare script.
"""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

#: The default export: the human/agent-readable outputs a swarm writes.
DEFAULT_EXTS = (".md", ".tsv", ".csv", ".json")
#: Vendored dependency and cache trees, pruned from every export (``all`` included): they hold
#: most of a swarm volume's files and none of its results.
SKIP_DIRS = frozenset({
    "node_modules", ".git", ".venv", "venv", "__pycache__", "site-packages", ".cache",
    ".npm", ".pnpm-store", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})
#: Threads for the walk and the read prefetch (both latency-bound on a Volume mount).
WORKERS = 64
#: Files above this are streamed from disk by the zip writer instead of prefetched, so the read
#: window (2 x WORKERS files) holds at most ~512 MiB.
PREFETCH_MAX_BYTES = 4 * 1024 * 1024
#: Zip timestamps cannot predate 1980; clamp to a day after, so no timezone tips it under.
_ZIP_EPOCH = 315619200


def _scan(path, skip, suffixes):
    """One directory: ``(subdirs, files, skipped_dirs, errors)``. Symlinks are never followed."""

    subdirs, files, skipped = [], [], 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in skip:
                        skipped += 1
                    else:
                        subdirs.append(entry.path)
                elif entry.is_file(follow_symlinks=False) and (
                        suffixes is None or entry.name.lower().endswith(suffixes)):
                    files.append(entry.path)
    except OSError:
        return subdirs, files, skipped, 1
    return subdirs, files, skipped, 0


def _walk(pool, root, skip, suffixes, stats):
    """Every kept file under ``root``, sorted, listing directories concurrently."""

    found = []
    pending = {pool.submit(_scan, root, skip, suffixes)}
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            subdirs, files, skipped, errors = future.result()
            found.extend(files)
            stats["skipped_dirs"] += skipped
            stats["errors"] += errors
            pending.update(pool.submit(_scan, d, skip, suffixes) for d in subdirs)
    return sorted(found)


def _read(path):
    """``(data or None, stat)``; ``None`` data means too big to prefetch (stream it instead)."""

    with open(path, "rb") as handle:
        st = os.fstat(handle.fileno())
        return (handle.read() if st.st_size <= PREFETCH_MAX_BYTES else None), st


def zip_tree(root, dest, *, exts=DEFAULT_EXTS, skip=SKIP_DIRS, compresslevel=6, workers=WORKERS):
    """Write a ZIP_DEFLATED zip of the files under ``root`` to ``dest`` (a path or binary file).

    ``exts`` keeps only files with those (case-insensitive) suffixes; ``None`` keeps every file.
    Directories named in ``skip`` are pruned. Symlinks are skipped, so nothing outside ``root``
    is followed. ``dest`` may be unseekable (stdout). Archive paths are relative to ``root`` and
    sorted. Returns ``{"files", "bytes", "skipped_dirs", "errors"}`` -- counts only, never names.
    """

    suffixes = tuple(e.lower() for e in exts) if exts is not None else None
    stats = {"files": 0, "bytes": 0, "skipped_dirs": 0, "errors": 0}
    with ThreadPoolExecutor(max(1, workers)) as pool, \
            zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        paths = iter(_walk(pool, root, skip, suffixes, stats))
        window = deque()

        def top_up():
            while len(window) < 2 * workers:
                path = next(paths, None)
                if path is None:
                    return
                window.append((path, pool.submit(_read, path)))

        top_up()
        while window:
            path, future = window.popleft()
            top_up()
            arcname = os.path.relpath(path, root).replace(os.sep, "/")
            try:
                data, st = future.result()
                if data is None:
                    zf.write(path, arcname)
                else:
                    info = zipfile.ZipInfo(arcname, time.localtime(max(st.st_mtime, _ZIP_EPOCH))[:6])
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (st.st_mode & 0xFFFF) << 16
                    zf.writestr(info, data, compresslevel=compresslevel)
            except OSError:  # vanished or unreadable mid-walk
                stats["errors"] += 1
                continue
            stats["files"] += 1
            stats["bytes"] += st.st_size
    return stats


def main(argv):
    """``<root> <out.zip|-> [--all]``: zip ``root`` (``-`` streams to stdout), stats JSON to stderr."""

    args = [a for a in argv if a != "--all"]
    root, out = args[0], args[1]
    exts = None if "--all" in argv else DEFAULT_EXTS
    stats = zip_tree(root, sys.stdout.buffer if out == "-" else out, exts=exts)
    sys.stdout.flush()
    sys.stderr.write(json.dumps(stats) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
