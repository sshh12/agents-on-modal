"""The Modal sandbox backend: an agent in a Modal Sandbox built from the batteries image.

A Modal Sandbox has full outbound internet, starts in well under a second on the v2 backend, and
scales to many thousands of concurrent instances, so this backend is the one used for massively
parallel swarms. Sandboxes are attached to a persistent, deployed App (``modal.App.lookup(...,
create_if_missing=True)``) so they outlive the launching process and can be rediscovered from any
process with :meth:`ModalSandbox.list` / :meth:`ModalSandbox.get` after a restart.

The model endpoint is reached **directly** (no egress container); each sandbox gets its own
*run-scoped* Modal proxy token, created at :meth:`create` and deleted at :meth:`stop`, so a crashed
or leaked sandbox never leaves a long-lived credential behind. When an upstream residential proxy is
configured, a loopback :mod:`agents_on_modal.sandbox.net.relay` is started inside the sandbox and the
``*_PROXY`` env is exported so code and the browser egress through the residential IP while model
inference stays direct (the model host is in ``NO_PROXY``).

``modal`` is imported lazily inside methods so this module imports with no Modal installed and with no
network, which keeps the catalog/registry import path offline. The image is published once as a named
image (``aom-sandbox:<tag>``) and only looked up on the hot path; building is an explicit, online
step (``with modal.enable_output(): image.build(app)``) that this module performs only on a cache miss
or when ``build=True`` is passed.

Modal specifics this backend relies on (modal 1.6.2): ``Sandbox.create`` blocks until scheduled;
there is no ``user=`` on ``exec`` (containers run as uid 0), so the agent user is entered with
``gosu``; the only filesystem API is ``sb.filesystem`` and it writes as root, so written files are
chowned to the agent; under gVisor there is no usable iptables, hence the loopback relay rather than a
packet filter.
"""

from __future__ import annotations

import hashlib
import shlex
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, ClassVar
from urllib.parse import urlsplit

from ..harness.base import McpServer, ModelAccess, SandboxProfile
from .base import ExecResult, Sandbox, SandboxError, new_id
from .instructions import build_instructions, load_skills
from .net import parse_proxy, relay

#: Persistent, deployed App that owns every aom Modal sandbox (so they are rediscoverable).
APP_NAME = "aom-sandboxes"
#: Published image name; the tag is a content hash of the Dockerfile + init.sh + this version.
IMAGE_NAME = "aom-sandbox"
#: Bump to force a new image tag even when the Dockerfile/init.sh bytes are unchanged.
IMAGE_VERSION = "1"

#: uid:gid of the ``agent`` user baked into the image (Dockerfile creates uid/gid 10001).
AGENT_UID_GID = "10001:10001"

#: The one browser MCP server every batteries sandbox advertises. All of its configuration
#: (headless, isolated, the system Chromium path, the output dir) is baked into the image env, so no
#: args are needed here; this matches the docker backend's browser recipe.
PLAYWRIGHT_MCP = McpServer(command="mcp-server-playwright")

_IMAGE_RESOURCES = files("agents_on_modal.sandbox").joinpath("image")


def _image_tag() -> str:
    """Short content hash of the image inputs. Offline (reads packaged bytes only)."""

    digest = hashlib.sha256()
    digest.update(IMAGE_VERSION.encode("utf-8"))
    for name in ("Dockerfile", "init.sh"):
        digest.update(_IMAGE_RESOURCES.joinpath(name).read_bytes())
    return digest.hexdigest()[:12]


def _image_ref() -> str:
    return f"{IMAGE_NAME}:{_image_tag()}"


def _is_not_found(modal: Any, error: BaseException) -> bool:
    """True when ``error`` is Modal's "named object does not exist" error."""

    not_found = getattr(getattr(modal, "exception", None), "NotFoundError", None)
    if not_found is not None and isinstance(error, not_found):
        return True
    return "NotFound" in type(error).__name__


def _is_fs_not_found(error: BaseException) -> bool:
    """True when a filesystem read failed because the path is missing."""

    if isinstance(error, FileNotFoundError):
        return True
    return "NotFound" in type(error).__name__


class ModalSandbox(Sandbox):
    """One agent in a Modal Sandbox from the batteries image (full internet, run-scoped token)."""

    kind: ClassVar[str] = "modal"

    def __init__(
        self,
        sb: Any,
        *,
        id: str,
        proxy: Any = None,
        proxy_url: str | None = None,
        token: str | None = None,
        token_id: str = "",
        shared: bool = False,
        endpoint_host: str = "",
        relay_env: Mapping[str, str] | None = None,
    ) -> None:
        self._sb = sb
        self.id = id
        self.workdir = "/work"
        self.home = "/home/agent"
        self.runs_root = "/home/agent/.aom/runs"
        self.posix = True
        self.python = ("gosu", "agent", "python3")
        #: Parsed upstream proxy (``UpstreamProxy`` | ``None``); the raw URL is kept for the relay.
        self.proxy = proxy
        self._proxy_url = proxy_url
        #: Run-scoped bearer token (``wk-….ws-…``) or ``None`` on a rediscovered handle.
        self.token = token
        self._proxy_token_id = token_id
        self._shared = shared
        self._endpoint_host = endpoint_host
        self._relay_env = dict(relay_env or {})

    # ---- image + app ------------------------------------------------------------------------
    @staticmethod
    def _app() -> Any:
        import modal

        return modal.App.lookup(APP_NAME, create_if_missing=True)

    @classmethod
    def ensure_image(cls, build: bool = False) -> Any:
        """Return the published sandbox image, building+publishing it only on a miss or ``build``.

        Hot path is a pure ``Image.from_name`` lookup (never triggers a build). A build requires
        network and output streaming; this method performs it as ``image.build(app)`` under
        ``modal.enable_output()`` and then publishes it under :data:`IMAGE_NAME`.
        """

        import modal

        ref = _image_ref()
        if not build:
            try:
                return modal.Image.from_name(ref)
            except Exception as error:  # noqa: BLE001 - re-raise anything that is not "not found"
                if not _is_not_found(modal, error):
                    raise
        image = modal.Image.from_dockerfile(
            str(_IMAGE_RESOURCES.joinpath("Dockerfile"))
        ).add_local_dir(str(files("agents_on_modal")), "/opt/aom/agents_on_modal", copy=True)
        with modal.enable_output():
            image = image.build(cls._app())
        image.publish(ref)
        # The built handle is usable as ``image=`` directly; it is now also published under ``ref``
        # so subsequent (and concurrent) creates take the pure ``Image.from_name`` hot path above.
        return image

    # ---- lifecycle --------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        endpoint: Any,
        *,
        proxy: str | None = None,
        timeout: float = 6 * 3600,
        idle_timeout: float = 15 * 60,
        cpu: Any = (1.0, 4.0),
        memory: Any = (2048, 8192),
        encrypted_ports: Sequence[int] = (),
        shared_volume: str | None = None,
        swarm: str | None = None,
        token: str | None = None,
        region: str | None = None,
        build: bool = False,
        **_: Any,
    ) -> "ModalSandbox":
        """Create and start one Modal Sandbox, with a fresh run-scoped proxy token."""

        import modal

        app = cls._app()
        image = cls.ensure_image(build=build)
        sid = new_id("aom-")

        # Mint a run-scoped proxy token so the model credential dies with the sandbox. Fall back to
        # the endpoint's own token if the workspace API is unavailable.
        bearer = token
        token_id = ""
        if bearer is None:
            try:
                created = modal.Workspace.from_context().proxy_tokens.create()
                token_id = getattr(created, "token_id", "")
                bearer = f"{token_id}.{getattr(created, 'token_secret', '')}"
            except Exception:  # noqa: BLE001 - any failure falls back to the shared endpoint token
                bearer = endpoint.token
                token_id = ""

        endpoint_host = (urlsplit(endpoint.url).hostname or "").strip()

        env: dict[str, str] = {"IS_SANDBOX": "1", "AOM_SANDBOX_ID": sid}
        upstream = parse_proxy(proxy, session=sid)
        relay_env: dict[str, str] = {}
        if upstream is not None:
            relay_env = relay.env(upstream, endpoint_host)
            env.update(relay_env)

        volumes: dict[str, Any] = {}
        shared = False
        if shared_volume:
            volumes["/shared"] = modal.Volume.from_name(
                shared_volume, version=2, create_if_missing=True
            )
            shared = True

        tags = {
            "aom": "1",
            "aom_id": sid,
            "harness": "",
            "swarm": swarm or "",
            "proxy_token_id": token_id,
        }

        try:
            sb = modal.Sandbox.create(
                "sleep",
                "infinity",
                app=app,
                image=image,
                name=sid,
                tags=tags,
                timeout=int(timeout),
                idle_timeout=int(idle_timeout) if idle_timeout else None,
                cpu=cpu,
                memory=memory,
                workdir="/work",
                encrypted_ports=list(encrypted_ports),
                volumes=volumes,
                env=env,
                region=region,
                verbose=False,
            )
        except Exception as error:  # noqa: BLE001 - on failure, do not leak the token we just made
            if token_id:
                try:
                    modal.Workspace.from_context().proxy_tokens.delete(token_id)
                except Exception:  # noqa: BLE001
                    pass
            raise SandboxError(f"could not create Modal sandbox {sid!r}: {error}") from error

        box = cls(
            sb,
            id=sid,
            proxy=upstream,
            proxy_url=proxy,
            token=bearer,
            token_id=token_id,
            shared=shared,
            endpoint_host=endpoint_host,
            relay_env=relay_env,
        )
        if upstream is not None:
            box._start_relay(proxy or "", sid, endpoint_host)
        return box

    @classmethod
    def _reconstruct(cls, sb: Any) -> "ModalSandbox":
        """Wrap a live Modal sandbox discovered by ``list``/``get`` (tags recover what they can)."""

        try:
            tags = dict(sb.get_tags())
        except Exception:  # noqa: BLE001
            tags = {}
        sid = tags.get("aom_id") or str(getattr(sb, "object_id", "") or "")
        return cls(
            sb,
            id=sid,
            token=None,
            token_id=tags.get("proxy_token_id", ""),
            shared=False,
            endpoint_host="",
            relay_env={},
        )

    @classmethod
    def list(cls, *, swarm: str | None = None, **_: Any) -> list["ModalSandbox"]:
        """Live aom Modal sandboxes (optionally only one swarm's), rediscovered via tags."""

        import modal

        try:
            app = modal.App.lookup(APP_NAME)
        except Exception as error:  # noqa: BLE001
            if _is_not_found(modal, error):
                return []
            raise
        tags = {"aom": "1"}
        if swarm:
            tags["swarm"] = swarm
        return [cls._reconstruct(sb) for sb in modal.Sandbox.list(app_id=app.app_id, tags=tags)]

    @classmethod
    def get(cls, sandbox_id: str) -> "ModalSandbox":
        """Reattach to a Modal sandbox by its aom id (its Modal name) or raw sandbox id."""

        import modal

        sb = None
        try:
            sb = modal.Sandbox.from_name(APP_NAME, sandbox_id)
        except Exception as error:  # noqa: BLE001
            if not _is_not_found(modal, error):
                raise
        if sb is None:
            try:
                sb = modal.Sandbox.from_id(sandbox_id)
            except Exception as error:  # noqa: BLE001
                raise SandboxError(f"no Modal sandbox named or identified by {sandbox_id!r}") from error
        return cls._reconstruct(sb)

    def stop(self) -> None:
        """Terminate the sandbox and delete its run-scoped proxy token (idempotent)."""

        import modal

        try:
            self._sb.terminate()
        except Exception:  # noqa: BLE001 - already gone is fine
            pass
        token_id = self._proxy_token_id
        if token_id:
            try:
                modal.Workspace.from_context().proxy_tokens.delete(token_id)
            except Exception:  # noqa: BLE001
                pass
            self._proxy_token_id = ""

    def labels(self) -> dict[str, str]:
        try:
            return dict(self._sb.get_tags())
        except Exception:  # noqa: BLE001
            return {}

    # ---- primitives -------------------------------------------------------------------------
    def _as_agent(self, argv: Sequence[str], *, root: bool) -> list[str]:
        """Wrap ``argv`` in ``gosu agent`` unless running as root or already wrapped."""

        argv = list(argv)
        if root or (argv and argv[0] == "gosu"):
            return argv
        return ["gosu", "agent", *argv]

    def exec(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 120,
        root: bool = False,
    ) -> ExecResult:
        wrapped = self._as_agent(argv, root=root)
        proc = self._sb.exec(
            *wrapped,
            env=dict(env) if env else None,
            workdir=cwd or self.workdir,
            timeout=int(timeout) if timeout else None,
        )
        out = proc.stdout.read()
        err = proc.stderr.read()
        proc.wait()
        return ExecResult(proc.returncode or 0, _text(out), _text(err))

    def spawn(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Launch a detached, session-leader background process and return immediately."""

        wrapped = self._as_agent(argv, root=False)
        command = (
            "setsid nohup "
            + " ".join(shlex.quote(part) for part in wrapped)
            + " >/tmp/aom-spawn.log 2>&1 &"
        )
        # The trailing ``&`` backgrounds the detached process; the launcher shell returns at once.
        self._sb.exec(
            "bash",
            "-lc",
            command,
            env=dict(env) if env else None,
            workdir=cwd or self.workdir,
        )

    def write_file(self, path: str, data: str | bytes) -> None:
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        self._sb.filesystem.write_bytes(raw, path)
        # The filesystem API runs as root, so hand the file (and any root-owned parent dirs it had
        # to create) to the agent, stopping at the first dir the agent already owns.
        quoted = shlex.quote(path)
        script = (
            f"chown {AGENT_UID_GID} {quoted} 2>/dev/null || true; "
            f"d=$(dirname {quoted}); "
            'while [ "$d" != "/" ] && [ "$d" != "." ] && '
            '[ "$(stat -c %u "$d" 2>/dev/null)" = 0 ]; do '
            f'chown {AGENT_UID_GID} "$d" 2>/dev/null || true; d=$(dirname "$d"); done'
        )
        self._sb.exec("bash", "-lc", script).wait()

    def read_file(self, path: str, offset: int = 0) -> bytes:
        try:
            data = self._sb.filesystem.read_bytes(path)
        except Exception as error:  # noqa: BLE001
            if _is_fs_not_found(error):
                raise FileNotFoundError(path) from error
            raise
        return bytes(data)[offset:]

    def attach(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> int:
        """Run an interactive command on the caller's terminal (used by ``aom shell``)."""

        wrapped = self._as_agent(argv, root=False)
        proc = self._sb.exec(
            *wrapped,
            env=dict(env) if env else None,
            workdir=cwd or self.workdir,
            pty=True,
        )
        try:
            proc.attach()
        except Exception:  # noqa: BLE001 - best effort; still report the exit code
            pass
        if proc.returncode is not None:
            return int(proc.returncode)
        return int(proc.wait() or 0)

    # ---- model access + profile -------------------------------------------------------------
    def model_access(self, endpoint: Any) -> ModelAccess:
        """Direct gateway URL + the run-scoped bearer token (no egress indirection)."""

        return ModelAccess.from_endpoint(endpoint, token=self.token)

    def profile(self) -> SandboxProfile:
        instructions = build_instructions(
            "modal",
            workdir=self.workdir,
            home=self.home,
            shared_dir="/shared" if self._shared else None,
            network="full",
            proxy=bool(self.proxy),
        )
        return SandboxProfile(
            kind="modal",
            instructions=instructions,
            posix=True,
            skills=load_skills(),
            mcp_servers={"playwright": PLAYWRIGHT_MCP},
            env=dict(self._relay_env),
        )

    # ---- internals --------------------------------------------------------------------------
    def _start_relay(self, proxy_url: str, session: str, model_host: str) -> None:
        """Start the loopback residential-proxy relay inside the sandbox (detached)."""

        self.spawn(
            [*self.python, "-m", "agents_on_modal.sandbox.net.relay"],
            env={
                "AOM_UPSTREAM_PROXY": proxy_url,
                "AOM_PROXY_SESSION": session,
                "AOM_MODEL_HOST": model_host,
            },
        )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return "" if value is None else str(value)


__all__ = ["APP_NAME", "IMAGE_NAME", "IMAGE_VERSION", "PLAYWRIGHT_MCP", "ModalSandbox"]
