"""Deploy, attach, scale and tear down a model's persistent Modal app.

These are the functions re-exported from ``agents_on_modal.models`` (``deploy``, ``endpoint``,
``wait_ready``, ``scale``, ``stop``, ``status``). Deploy is idempotent: it ``modal deploy``\\ s
``agents_on_modal.models.serve`` with the model spec in ``AOM_MODEL_JSON``, runs the CPU
``prepare`` function so weights are downloaded before any GPU is requested, and returns an
``Endpoint``.

``modal`` is imported inside the functions, never at module top, so importing this module (and
therefore the catalog) off-host never pulls in Modal. The URL / app / volume-name derivation and
the DNS-label shortening are pure and need no Modal.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from ..config import AomError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import ProxyToken
    from . import Endpoint, Model, Scaling

# Env var names the serve module reads (duplicated here so importing deploy never imports serve,
# which imports modal). Kept in sync with agents_on_modal.models.serve.
AOM_MODEL_JSON_ENV = "AOM_MODEL_JSON"
AOM_APP_NAME_ENV = "AOM_APP_NAME"

SERVE_MODULE = "agents_on_modal.models.serve"
#: Pin the image builder so deploys are reproducible (A5).
MODAL_IMAGE_BUILDER_VERSION = "2025.06"
#: A Modal Server URL is one DNS label: ``{workspace}--{app}-server`` (RFC 1035, A2).
MAX_DNS_LABEL_LENGTH = 63
#: Autoscaler settings ``scale()`` accepts (maps onto Server.autoscaler.update).
AUTOSCALER_KEYS = frozenset(
    {"min_containers", "max_containers", "buffer_containers",
     "scaledown_window", "scaleup_window", "target_concurrency"}
)


# --------------------------------------------------------------------------- #
# Pure name/URL derivation (no Modal)                                          #
# --------------------------------------------------------------------------- #
def shorten_app_name(app_name: str, workspace_slug: str) -> str:
    """Fit ``{workspace}--{app}-server`` inside a 63-char DNS label, deterministically."""

    fixed = len(workspace_slug) + len("--") + len("-server")
    maximum = MAX_DNS_LABEL_LENGTH - fixed
    if maximum < 10:
        raise AomError(
            f"Modal workspace slug {workspace_slug!r} leaves no room for a Server app name."
        )
    if len(app_name) <= maximum:
        return app_name
    digest = hashlib.sha256(app_name.encode("utf-8")).hexdigest()[:8]
    prefix = app_name[: maximum - len(digest) - 1].rstrip("-")
    shortened = f"{prefix}-{digest}"
    if not prefix or len(shortened) > maximum:
        raise AomError(f"Could not derive a DNS-safe Modal app name from {app_name!r}.")
    return shortened


def deployed_app_name(model: "Model", workspace_slug: str) -> str:
    """The Modal app name actually deployed for a model (possibly shortened for DNS)."""

    return shorten_app_name(model.app_name, workspace_slug)


def base_url(app_name: str, workspace_slug: str, region: str) -> str:
    """The gateway's base URL (no ``/v1``; ``Endpoint.openai_base_url`` appends it)."""

    label = f"{workspace_slug}--{app_name}-server"
    if len(label) > MAX_DNS_LABEL_LENGTH:
        raise AomError(
            f"Modal Server DNS label is {len(label)} chars; the limit is {MAX_DNS_LABEL_LENGTH}."
        )
    return f"https://{label}.{region}.modal.direct"


# --------------------------------------------------------------------------- #
# Modal seams (imported lazily; monkeypatched in tests)                        #
# --------------------------------------------------------------------------- #
def _workspace_slug() -> str:
    """The workspace DNS slug used in ``*.modal.direct`` URLs.

    ``Workspace.from_context()`` is lazy; its ``name`` is ``None`` until hydrated. Hydrate it, and
    fall back to parsing ``modal profile current`` if the SDK path is unavailable.
    """

    import modal

    try:
        workspace = modal.Workspace.from_context()
        hydrate = getattr(workspace, "hydrate", None)
        if callable(hydrate):
            hydrate()
        slug = getattr(workspace, "name", None)
        if slug:
            return str(slug)
    except Exception:  # noqa: BLE001 - fall through to the CLI probe
        pass

    result = subprocess.run(
        [sys.executable, "-m", "modal", "profile", "current"],
        text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    slug = (result.stdout or "").strip().splitlines()
    if result.returncode == 0 and slug and slug[-1].strip():
        return slug[-1].strip()
    raise AomError(
        "Could not determine the active Modal workspace slug. Run `modal token new` / `aom setup`."
    )


def _modal_deploy(app_name: str, env: dict[str, str]) -> None:
    """Run ``modal deploy --name <app_name> -m agents_on_modal.models.serve`` with live output."""

    command = [sys.executable, "-m", "modal", "deploy", "--name", app_name, "-m", SERVE_MODULE]
    process = subprocess.Popen(
        command, env=env, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    assert process.stdout is not None
    chunks: list[str] = []
    for line in process.stdout:
        chunks.append(line)
        # The console encoding (cp1252 on Windows) may not cover modal's progress glyphs; never
        # let streaming the child's output crash the deploy.
        try:
            print(line, end="", flush=True)
        except UnicodeEncodeError:
            sys.stdout.write(line.encode("utf-8", "replace").decode("utf-8", "replace"))
            sys.stdout.flush()
    code = process.wait()
    if code != 0:
        detail = "".join(chunks).strip()
        raise AomError(f"modal deploy failed (exit {code})" + (f": {detail}" if detail else "."))


def _run_prepare(app_name: str) -> Any:
    """Invoke the deployed CPU ``prepare`` function (downloads weights before any GPU)."""

    import modal

    function = modal.Function.from_name(app_name, "prepare")
    with modal.enable_output():
        return function.remote()


def _apply_autoscaler(server: Any, settings: dict[str, Any]) -> None:
    """Update a Server's autoscaler (modal >= 1.6.2 ``autoscaler.update``; 1.5.x fallback)."""

    autoscaler = getattr(server, "autoscaler", None)
    if autoscaler is not None and hasattr(autoscaler, "update"):
        autoscaler.update(**settings)
    else:  # modal 1.5.x
        server.update_autoscaler(**settings)


def _autoscaler_settings(server: Any) -> dict[str, Any] | None:
    autoscaler = getattr(server, "autoscaler", None)
    if autoscaler is not None and hasattr(autoscaler, "settings"):
        value = autoscaler.settings()
    elif hasattr(server, "get_autoscaler_settings"):
        value = server.get_autoscaler_settings()
    else:
        return None
    if isinstance(value, dict):
        return value
    return {k: getattr(value, k) for k in AUTOSCALER_KEYS if hasattr(value, k)}


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def deploy(model: "Model", *, scaling: "Scaling | None" = None, token: "ProxyToken | None" = None) -> "Endpoint":
    """Idempotently deploy the model's Modal app and prepare its weights; return an Endpoint."""

    from ..config import load_token
    from . import Endpoint

    proxy = token or load_token()
    model_json = model.to_json()
    if scaling is not None:
        model_json["scaling"] = dict(scaling.__dict__)

    workspace = _workspace_slug()
    app_name = deployed_app_name(model, workspace)

    env = os.environ.copy()
    env[AOM_MODEL_JSON_ENV] = json.dumps(model_json)
    env[AOM_APP_NAME_ENV] = app_name
    env["MODAL_IMAGE_BUILDER_VERSION"] = MODAL_IMAGE_BUILDER_VERSION
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    _modal_deploy(app_name, env)
    # Prepare weights on CPU before any GPU allocates (cost-critical, A11).
    _run_prepare(app_name)

    url = base_url(app_name, workspace, str(model_json.get("region", "us-west")))
    return Endpoint(model=model, url=url, token=proxy.bearer)


def endpoint(model: "Model", *, token: "ProxyToken | None" = None) -> "Endpoint":
    """Attach to an already-deployed model without deploying; compute its URL and token."""

    from ..config import load_token
    from . import Endpoint

    proxy = token or load_token()
    workspace = _workspace_slug()
    app_name = deployed_app_name(model, workspace)
    url = base_url(app_name, workspace, model.region)
    return Endpoint(model=model, url=url, token=proxy.bearer)


def wait_ready(endpoint: "Endpoint", *, timeout: float = 1800) -> None:
    """Poll ``endpoint.url + "/health"`` with the proxy token until it returns 200."""

    health_url = endpoint.url + "/health"
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(
        health_url, headers={"Authorization": f"Bearer {endpoint.token}"}
    )
    last: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if getattr(response, "status", response.getcode()) == 200:
                    return
                last = f"HTTP {response.status}"
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise AomError(
                    f"{health_url} rejected the proxy token ({error.code}); "
                    "check the token is valid for this workspace/environment."
                ) from error
            last = f"HTTP {error.code}"  # 404/503/5xx: still cold-starting
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as error:
            last = str(error)
        time.sleep(5)
    raise AomError(f"{health_url} did not become ready within {timeout:g}s (last: {last}).")


def scale(model: "Model", **settings: Any) -> None:
    """Update the server's autoscaler live (no redeploy), e.g. ``scale(m, min_containers=4)``."""

    unknown = sorted(set(settings) - AUTOSCALER_KEYS)
    if unknown:
        raise AomError(
            f"Unknown scale settings {unknown}; allowed: {', '.join(sorted(AUTOSCALER_KEYS))}."
        )
    if not settings:
        raise AomError("scale() needs at least one setting to change.")

    import modal

    workspace = _workspace_slug()
    app_name = deployed_app_name(model, workspace)
    server = modal.Server.from_name(app_name, "Server")
    _apply_autoscaler(server, settings)


def stop(model: "Model") -> None:
    """Stop the model's deployed app (idempotent: a missing app is treated as already stopped)."""

    import modal

    workspace = _workspace_slug()
    app_name = deployed_app_name(model, workspace)
    try:
        app = modal.App.lookup(app_name)
    except Exception as error:  # noqa: BLE001 - NotFoundError etc.
        if "not found" in str(error).lower() or "does not exist" in str(error).lower():
            return
        raise AomError(f"Could not look up Modal app {app_name!r}: {error}") from error

    stop_fn = getattr(app, "stop", None)
    if callable(stop_fn):
        stop_fn()
        return
    # modal 1.5.x has no App.stop; stop running containers by zeroing the autoscaler.
    server = modal.Server.from_name(app_name, "Server")
    _apply_autoscaler(server, {"min_containers": 0, "max_containers": 0})


def status(model: "Model") -> dict[str, Any]:
    """A best-effort snapshot: app name, URL, volume, and autoscaler settings if reachable."""

    workspace: str | None = None
    try:
        workspace = _workspace_slug()
    except Exception as error:  # noqa: BLE001
        return {"model": model.name, "app_name": model.app_name, "error": str(error)}

    app_name = deployed_app_name(model, workspace)
    info: dict[str, Any] = {
        "model": model.name,
        "app_name": app_name,
        "volume": model.volume_name,
        "url": base_url(app_name, workspace, model.region),
    }
    try:
        import modal

        server = modal.Server.from_name(app_name, "Server")
        settings = _autoscaler_settings(server)
        if settings is not None:
            info["autoscaler"] = settings
        get_url = getattr(server, "get_url", None)
        if callable(get_url):
            info["server_url"] = get_url()
    except Exception as error:  # noqa: BLE001 - not deployed / unreachable
        info["server_error"] = str(error)
    return info
