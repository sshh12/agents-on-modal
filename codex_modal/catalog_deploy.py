"""Deploy a self-managed model straight from the JSON catalog.

    python -m codex_modal.catalog_deploy 0
    python -m codex_modal.catalog_deploy glm-5.3-flash-uncensored
    python -m codex_modal.catalog_deploy 0 --gpu H200:4 --dry-run
    python -m codex_modal.catalog_deploy custom-catalog.json my-model   # explicit file

The catalog is the single source of truth for how a model is brought up; this
module turns one entry into the equivalent ``codex-modal --modal-self-managed``
invocation and runs it, then prints the served endpoint URL. Adding a model is a
catalog entry, not code.

Catalog shape - the unified ``catalog.json`` has a ``self_managed`` section that
holds the ``models`` array (its managed-endpoint presets live under
``modal_managed``). An explicit path argument may instead point at a bare
``models`` array or a ``{"models": [...]}`` object, so custom catalogs still
work. Each entry:

    name              label for selection by name                (required)
    source_repo       Hugging Face repo to serve                 (required)
    endpoint_name     Modal app / DNS label; also the attach name (required)
    gpu               e.g. "H100:8" (``--gpu`` overrides)         (required)
    region            Modal routing region        (default us-west)
    hf_token_env      env var holding the HF token, if the repo is gated
    context_window    served context length            (default 262144)
    reasoning_effort  default effort                       (default high)
    reasoning_levels  advertised effort ladder      (default low/high/max)
    scaledown_window  idle seconds before scale-to-zero  (default 900)
    keep_endpoint     leave the endpoint up after deploy  (default true)
    runtime           serving engine "sglang" (default) or "llamacpp"
    download_subdir   restrict the repo download to this subdir (e.g. a single
                      GGUF quant folder like "Mix-256G")
    server_args       list of extra "--flag[=value]" args for the serving engine
                      (SGLang flags, or llama-server flags when runtime=llamacpp);
                      legacy alias: ``sglang_args``
    serving_image     base registry image for the runtime; for sglang this is the
                      SGLang image, for llamacpp a CUDA base is used by default.
                      legacy alias: ``sglang_image``
    serving_pip       list of extra pip packages for the serving image
                      (e.g. ["transformers==5.16.0"] for a new architecture)
    cpu, memory, target_inputs                 optional overrides

Runtimes differ only in image, download/readiness, and launch; see
``codex_modal/runtimes.py``. llama.cpp entries omit ``serving_image`` (the
runtime compiles llama-server from source) and name the GGUF quant folder via
``download_subdir``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .custom_deploy import custom_app_base_url, custom_app_name
from .modal_cli import current_workspace_slug
from .paths import CATALOG_PATH, PROJECT_ROOT


def _load_models(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: cannot read catalog {path}: {error}")
    # Unified catalog: the self-managed specs live under 'self_managed'.
    if isinstance(document, dict) and isinstance(document.get("self_managed"), dict):
        document = document["self_managed"]
    if isinstance(document, list):
        models = document
    elif isinstance(document, dict) and isinstance(document.get("models"), list):
        models = document["models"]
    else:
        raise SystemExit(
            f"error: {path} must be the unified catalog (a 'self_managed.models' array), "
            "a JSON array, or an object with a 'models' array."
        )
    if not all(isinstance(entry, dict) for entry in models):
        raise SystemExit(f"error: every entry in {path} must be a JSON object.")
    return models


def _select(models: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    if selector.lstrip("-").isdigit():
        index = int(selector)
        if not -len(models) <= index < len(models):
            raise SystemExit(f"error: index {index} is out of range (0..{len(models) - 1}).")
        return models[index]
    for entry in models:
        if str(entry.get("name", "")) == selector or str(entry.get("endpoint_name", "")) == selector:
            return entry
    names = ", ".join(str(entry.get("name", "?")) for entry in models) or "<none>"
    raise SystemExit(f"error: no catalog entry named {selector!r}. Available: {names}")


def _codex_modal_argv(entry: dict[str, Any], gpu: str) -> list[str]:
    argv = [
        sys.executable, "-m", "codex_modal",
        "--modal-self-managed",
        "--modal-gpu", gpu,
        "--modal-custom-hf-repo", str(entry["source_repo"]),
        "--modal-endpoint-name", str(entry["endpoint_name"]),
        "--modal-routing-region", str(entry.get("region", "us-west")),
        "--modal-context-window", str(int(entry.get("context_window", 262144))),
        "--modal-reasoning-effort", str(entry.get("reasoning_effort", "high")),
        "--modal-reasoning-levels", ",".join(str(x) for x in entry.get("reasoning_levels", ["low", "high", "max"])),
        "--modal-scaledown-window", str(int(entry.get("scaledown_window", 900))),
    ]
    if entry.get("keep_endpoint", True):
        argv.append("--modal-keep-endpoint")
    # Optional base model + shard-reuse (fine-tunes/abliterations that share most
    # shards with a base checkpoint already on a Modal Volume).
    if entry.get("base_model"):
        argv += ["--modal-model", str(entry["base_model"])]
    if entry.get("base_model_revision"):
        argv += ["--modal-model-revision", str(entry["base_model_revision"])]
    if entry.get("custom_hf_revision"):
        argv += ["--modal-custom-hf-revision", str(entry["custom_hf_revision"])]
    if entry.get("base_volume"):
        argv += ["--modal-base-volume", str(entry["base_volume"])]
    if entry.get("base_volume_path"):
        argv += ["--modal-base-volume-path", str(entry["base_volume_path"])]
    if entry.get("hf_token_env"):
        argv += ["--modal-hf-token-env", str(entry["hf_token_env"])]
    if entry.get("runtime"):
        argv += ["--modal-runtime", str(entry["runtime"])]
    if entry.get("download_subdir"):
        argv += ["--modal-download-subdir", str(entry["download_subdir"])]
    # Serving base image: generic `serving_image`, or legacy `sglang_image`.
    serving_image = entry.get("serving_image") or entry.get("sglang_image")
    if serving_image:
        argv += ["--modal-sglang-image", str(serving_image)]
    for optional, flag in (("cpu", "--modal-cpu"), ("memory", "--modal-memory"),
                           ("target_inputs", "--modal-target-inputs"),
                           ("startup_timeout", "--modal-startup-timeout")):
        if entry.get(optional) is not None:
            argv += [flag, str(entry[optional])]
    # Serving-engine args: generic `server_args`, or legacy `sglang_args`.
    for server_arg in entry.get("server_args") or entry.get("sglang_args", []):
        argv += ["--modal-sglang-arg", str(server_arg)]
    for package in entry.get("serving_pip", []):
        argv += ["--modal-serving-pip", str(package)]
    argv += ["--", "exec", "--skip-git-repo-check", "Reply with exactly OK."]
    return argv


def _served_url(endpoint_name: str, region: str) -> str | None:
    try:
        slug = current_workspace_slug()
        return custom_app_base_url(slug, custom_app_name(endpoint_name, slug), region)
    except Exception as error:  # noqa: BLE001 - slug lookup can fail many ways; non-fatal
        print(f"warning: could not compute the served URL yet ({error}); read it from the "
              "logs below.", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modal-deploy", description="Deploy a self-managed model from a JSON catalog."
    )
    parser.add_argument(
        "catalog", nargs="?", default=str(CATALOG_PATH),
        help=f"path to the catalog JSON (default: {CATALOG_PATH.name})",
    )
    parser.add_argument("selector", help="entry index (e.g. 0) or name")
    parser.add_argument("--gpu", help="override the entry's gpu, e.g. H100:8")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = parser.parse_args(argv)

    entry = _select(_load_models(Path(args.catalog).expanduser()), args.selector)
    for required in ("source_repo", "endpoint_name", "gpu"):
        if not entry.get(required):
            raise SystemExit(f"error: catalog entry is missing required field {required!r}.")
    gpu = args.gpu or str(entry["gpu"])
    endpoint_name = str(entry["endpoint_name"])
    region = str(entry.get("region", "us-west"))

    command = _codex_modal_argv(entry, gpu)
    url = _served_url(endpoint_name, region)

    print(f"catalog   : {args.catalog}")
    print(f"model     : {entry.get('name', endpoint_name)}  ({entry['source_repo']})")
    print(f"gpu       : {gpu}   region: {region}   endpoint: {endpoint_name}")
    if url:
        print(f"served URL: {url}")
    print(f"command   : python -m {' '.join(command[2:])}\n")

    if args.dry_run:
        return 0

    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode == 0:
        print(f"\nEndpoint {endpoint_name!r} is up. Attach by name, e.g.:")
        print(f"  codex-modal --modal-use-app {endpoint_name} -- exec 'hi'")
        print(f"  blue-green-red:  --model <role>=modal:{endpoint_name}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
