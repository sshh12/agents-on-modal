"""Deploy, attach, scale and tear down a model's persistent Modal app.

These are the functions re-exported from ``agents_on_modal.models`` (``deploy``, ``endpoint``,
``wait_ready``, ``scale``, ``stop``, ``status``). Deploy is idempotent: it ``modal deploy``\\ s
``agents_on_modal.models.serve`` with the model spec in ``AOM_MODEL_JSON``, runs the CPU
``prepare`` function so weights are downloaded before any GPU is requested, and returns an
``Endpoint``. ``ensure_deployed`` is the shared-consumer variant: it attaches to an app that is
already deployed rather than redeploying it out from under whoever is using it.

The endpoint's GPU container cap (``max_containers``) is not in the model spec; the app using the
endpoint passes it. ``deploy`` sets it exactly (a redeploy restarts the endpoint anyway), while
``ensure_deployed`` only ever raises it, so two apps asking for 1 share one endpoint at 1, and
an app asking for 2 bumps it live without disturbing the other. Only ``scale`` lowers it.

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
import threading
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
AOM_MAX_CONTAINERS_ENV = "AOM_MAX_CONTAINERS"

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
        # The console encoding (cp1252 on Windows) may not cover modal's progress glyphs, and a
        # programmatic caller (Swarm.launch, the dashboard) never ran cli._configure_stdio, so its
        # stdout can be cp1252. Re-encode to the console's OWN codec -- not utf-8, which can encode
        # the glyph and so would raise again -- turning unrepresentable glyphs into '?'. Never let
        # streaming the child's output crash the deploy.
        try:
            print(line, end="", flush=True)
        except UnicodeEncodeError:
            try:
                enc = getattr(sys.stdout, "encoding", None) or "utf-8"
                sys.stdout.write(line.encode(enc, "replace").decode(enc, "replace"))
                sys.stdout.flush()
            except Exception:  # noqa: BLE001 - output is best-effort; the deploy must not die for it
                pass
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


def _server_deployed(app_name: str) -> bool:
    """True when the app's Server resolves, i.e. it is deployed (scaled to zero still counts)."""

    import modal

    try:
        modal.Server.from_name(app_name, "Server").hydrate()
    except Exception:  # noqa: BLE001 - not found / stopped -> not deployed (same signal as status())
        return False
    return True


def _live_max_containers(app_name: str) -> int | None:
    """The Server's cap in effect right now; ``None`` when Modal reports none (no Server-level cap)."""

    import modal

    settings = _autoscaler_settings(modal.Server.from_name(app_name, "Server")) or {}
    return settings.get("max_containers")


def _set_max_containers(app_name: str, n: int) -> None:
    import modal

    _apply_autoscaler(modal.Server.from_name(app_name, "Server"), {"max_containers": n})


#: One lock per app, so concurrent ``ensure_deployed`` calls in this process (say, two dashboard
#: spawns) neither race two deploys of one app nor interleave their cap checks.
_APP_LOCKS: dict[str, threading.Lock] = {}
_APP_LOCKS_GUARD = threading.Lock()


def _app_lock(app_name: str) -> threading.Lock:
    with _APP_LOCKS_GUARD:
        return _APP_LOCKS.setdefault(app_name, threading.Lock())


def _check_max_containers(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise AomError(f"max_containers must be a positive integer, got {n!r}.")
    return n


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
def deploy(
    model: "Model", *, max_containers: int, scaling: "Scaling | None" = None,
    token: "ProxyToken | None" = None,
) -> "Endpoint":
    """Deploy (or redeploy) the model's Modal app capped at ``max_containers``; prepare its weights.

    Redeploying a live app restarts it (see ``ensure_deployed``, the attach-if-live variant apps
    should use). The cap is set exactly, statically and as a live override right after the deploy,
    so the cap read back from the endpoint is the one this deploy chose.
    """

    from ..config import load_token
    from . import Endpoint

    cap = _check_max_containers(max_containers)
    proxy = token or load_token()
    model_json = model.to_json()
    if scaling is not None:
        model_json["scaling"] = dict(scaling.__dict__)

    workspace = _workspace_slug()
    app_name = deployed_app_name(model, workspace)

    env = os.environ.copy()
    env[AOM_MODEL_JSON_ENV] = json.dumps(model_json)
    env[AOM_APP_NAME_ENV] = app_name
    env[AOM_MAX_CONTAINERS_ENV] = str(cap)
    env["MODAL_IMAGE_BUILDER_VERSION"] = MODAL_IMAGE_BUILDER_VERSION
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    _modal_deploy(app_name, env)
    _set_max_containers(app_name, cap)
    # Prepare weights on CPU before any GPU allocates (cost-critical, A11).
    _run_prepare(app_name)

    url = base_url(app_name, workspace, str(model_json.get("region", "us-west")))
    return Endpoint(model=model, url=url, token=proxy.bearer)


def ensure_deployed(
    model: "Model", *, max_containers: int, token: "ProxyToken | None" = None,
) -> "Endpoint":
    """Attach to the model's app if it is already deployed, ``deploy`` it if not; return an Endpoint.

    The entry point for every app that *uses* a model (swarm launches, sessions, the dashboard).
    ``deploy`` always ``modal deploy``\\ s, and a redeploy that changes anything in the app -- a
    catalog edit (the spec is baked into the image) or a different ``agents_on_modal`` source
    upload -- makes Modal replace the running containers: every in-flight request from every other
    app (another swarm, a local session) is cut and the next one waits out a full cold start.
    Attaching leaves the endpoint running; rolling out a change stays an explicit ``deploy``
    (``aom models deploy``). ``prepare`` still runs (resumable, a no-op once the weights are
    complete), so an interrupted first download gets finished instead of served from a partial
    volume.

    ``max_containers`` is this app's ask for the endpoint's GPU container cap. A first deploy uses
    it; a live endpoint is raised to it when lower (live, no restart) and never lowered, since other
    apps on it may have asked for more. A cap is a ceiling, not a reservation: Modal only adds a
    container once in-flight requests exceed the model's ``target_concurrency`` per container for
    ``scaleup_window``, and scales idle ones to zero, so a raised cap costs nothing until there is
    load for it. Lowering is an explicit ``scale``.
    """

    from ..config import load_token

    cap = _check_max_containers(max_containers)
    proxy = token or load_token()
    app_name = deployed_app_name(model, _workspace_slug())
    with _app_lock(app_name):
        if not _server_deployed(app_name):
            return deploy(model, max_containers=cap, token=proxy)
        _run_prepare(app_name)
        current = _live_max_containers(app_name)
        if current is None or current < cap:  # None: no cap in effect -- establish this one
            _set_max_containers(app_name, cap)
            was = "none" if current is None else current
            print(f"[aom] {model.name}: max containers {was} -> {cap}", flush=True)
    return endpoint(model, token=proxy)


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


def active_containers(model: "Model") -> int | None:
    """Point-in-time count of containers currently serving this model's endpoint.

    This is the *active* signal, distinct from *deployed*: a deployed app that is scaled to zero
    returns ``0`` (cold -- no GPU cost until the first request), while ``>= 1`` means live containers
    are up (warm, billing). Returns ``None`` when the app isn't deployed/reachable, or if Modal's
    experimental listing API is unavailable -- callers then fall back to the deployed-only view.
    """

    import modal

    try:
        workspace = _workspace_slug()
        app_name = deployed_app_name(model, workspace)
        server = modal.Server.from_name(app_name, "Server")
        # ``_experimental_list_containers`` returns the containers registered to serve requests right
        # now; its length is the live replica count. It's experimental, so guard against it vanishing.
        lister = getattr(server, "_experimental_list_containers", None)
        if lister is None:
            return None
        return len(lister())
    except Exception:  # noqa: BLE001 - not deployed / unreachable / experimental API drift
        return None


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
    # No App.stop in this SDK (1.5/1.6), and 1.6 rejects the old "max_containers=0" fallback; the
    # CLI's `app stop` works on every version and drains the running containers too.
    _modal_app_stop(app_name)


def _modal_app_stop(app_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "modal", "app", "stop", app_name, "--yes"],
        text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode != 0:
        raise AomError(f"modal app stop {app_name} failed: {(result.stdout or '').strip()[-400:]}")


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
