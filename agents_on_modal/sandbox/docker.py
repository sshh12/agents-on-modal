"""The Docker sandbox backend: a throwaway local container with full outbound internet, no host or
LAN access, the batteries-included image, and an optional residential proxy.

Topology (per sandbox, two containers + two Docker networks)::

    +-----------------------------+        +----------------------------------+
    | <id>-egress                 |        | <id>-agent                       |
    | networks: <id>-inner + edge |<------>| network: <id>-inner (--internal) |
    | holds the model credential  |  3128  | runs the agent, firewalled       |
    | and the upstream proxy url  |  8081  | (AOM_INIT_FIREWALL=1)            |
    +-----------------------------+        +----------------------------------+
            |                                       no route anywhere else
            v  (edge bridge has a default route)
        public internet

``<id>-inner`` is created with ``--internal`` so it has no default route: the agent cannot reach the
internet, the host, or the LAN directly. Its only peer is the egress broker, which validates every
destination (public IPs only, allow-listed ports) and injects the model credential the agent never
holds. A default-deny iptables policy inside the agent's own netns (applied by ``init.sh`` when
``AOM_INIT_FIREWALL=1``) is the second, independent layer.

Nothing from the host filesystem is mounted; the container is thrown away on :meth:`stop`.

The image is built in two stages: a cached batteries base (``aom-sandbox-base:<hash>`` from
``image/Dockerfile``) and a thin derived image that copies the ``agents_on_modal`` package to
``/opt/aom`` so ``python3 -m agents_on_modal.*`` resolves inside the sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..harness.base import McpServer, ModelAccess, SandboxProfile
from .base import ExecResult, Sandbox, SandboxError, new_id
from .instructions import build_instructions, load_skills

# Ports the broker listens on (see net/egress.py and image/init.sh).
PROXY_PORT = 3128
MODEL_PORT = 8081

# Label set on every container so list()/get() can find sandboxes aom created.
SANDBOX_LABEL = "aom.sandbox"
ID_LABEL = "aom.id"

# A placeholder bearer token: the broker strips it and injects the real credential, so the agent
# never holds a usable key.
DUMMY_MODEL_TOKEN = "aom-sandbox-no-credential"

_HERE = Path(__file__).resolve()
_IMAGE_DIR = _HERE.parent / "image"
_REPO_ROOT = _HERE.parents[2]
_DOCKERFILE = _IMAGE_DIR / "Dockerfile"
_INIT_SCRIPT = _IMAGE_DIR / "init.sh"


class DockerError(SandboxError):
    """A Docker CLI call failed, or the daemon is unreachable."""


def _docker() -> str:
    executable = shutil.which("docker")
    if not executable:
        raise DockerError("The `docker` CLI is not on PATH; the docker sandbox needs it.")
    return executable


def _run(
    args: Sequence[str],
    *,
    input: bytes | None = None,
    text: bool = True,
    check: bool = False,
    timeout: float | None = 300.0,
) -> subprocess.CompletedProcess:
    """Run ``docker <args>`` and return the completed process (never raising on a nonzero exit unless
    ``check``). Everything the backend does goes through here so tests can patch one seam."""

    result = subprocess.run(  # noqa: S603 - args are built by us, not user input
        [_docker(), *args],
        input=input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        detail = ""
        if text:
            detail = ((result.stderr or "") + (result.stdout or "")).strip()
        raise DockerError(
            f"docker {' '.join(str(a) for a in args[:3])} failed (exit {result.returncode})"
            + (f": {detail}" if detail else ".")
        )
    return result


def _run_interactive(args: Sequence[str]) -> int:
    """Run ``docker <args>`` attached to the caller's terminal; return its exit code."""

    return subprocess.run([_docker(), *args], check=False).returncode  # noqa: S603


def _image_exists(reference: str) -> bool:
    return _run(["image", "inspect", reference], check=False, timeout=120.0).returncode == 0


def _base_tag() -> str:
    """``aom-sandbox-base:<short hash of Dockerfile + init.sh>`` — the cached batteries image."""

    digest = hashlib.sha256()
    digest.update(_DOCKERFILE.read_bytes())
    digest.update(_INIT_SCRIPT.read_bytes())
    return f"aom-sandbox-base:{digest.hexdigest()[:12]}"


def ensure_image(build: bool = False) -> str:
    """Ensure both image stages exist and return the derived image tag to run containers from.

    The base is skipped when its content-hash tag already exists (``build=True`` forces a rebuild).
    The derived stage is a thin ``FROM <base>`` + ``COPY agents_on_modal`` built with the repo root
    as context; Docker's layer cache keeps it cheap, so it is always (re)built.
    """

    base = _base_tag()
    if build or not _image_exists(base):
        _run(
            ["build", "-t", base, "-f", str(_DOCKERFILE), str(_IMAGE_DIR)],
            check=True,
            timeout=3600.0,
        )
    derived = f"aom-sandbox:{base.split(':', 1)[1]}"
    dockerfile_text = f"FROM {base}\nCOPY agents_on_modal /opt/aom/agents_on_modal\n"
    handle, derived_dockerfile = tempfile.mkstemp(prefix="aom-derived-", suffix=".Dockerfile")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(dockerfile_text)
        _run(
            ["build", "-t", derived, "-f", derived_dockerfile, str(_REPO_ROOT)],
            check=True,
            timeout=3600.0,
        )
    finally:
        try:
            os.unlink(derived_dockerfile)
        except OSError:
            pass
    return derived


def _network_ip(container: str, network: str) -> str:
    """The container's IPv4 address on ``network`` (``docker inspect``)."""

    template = '{{(index .NetworkSettings.Networks "%s").IPAddress}}' % network
    result = _run(["inspect", "-f", template, container], check=True, timeout=120.0)
    ip = result.stdout.strip()
    if not ip:
        raise DockerError(f"Could not discover the IP of {container} on network {network}.")
    return ip


def _container_env(container: str) -> dict[str, str]:
    """The container's environment as a dict (``docker inspect``)."""

    result = _run(["inspect", "-f", "{{json .Config.Env}}", container], check=False, timeout=120.0)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        entries = json.loads(result.stdout)
    except ValueError:
        return {}
    out: dict[str, str] = {}
    for entry in entries or []:
        key, _, value = str(entry).partition("=")
        out[key] = value
    return out


class DockerSandbox(Sandbox):
    """A throwaway local Docker container pair driven through ``docker exec``."""

    kind = "docker"

    def __init__(self, *, sandbox_id: str, egress_ip: str, proxy: str | None = None) -> None:
        self.id = sandbox_id
        self.egress_ip = egress_ip
        self.proxy = proxy
        self.agent_container = f"{sandbox_id}-agent"
        self.egress_container = f"{sandbox_id}-egress"
        self.inner_network = f"{sandbox_id}-inner"
        self.edge_network = f"{sandbox_id}-edge"
        self.workdir = "/work"
        self.home = "/home/agent"
        self.runs_root = "/home/agent/.aom/runs"
        self.posix = True
        self.python = ("python3",)

    # ---- lifecycle --------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        endpoint: Any = None,
        proxy: str | None = None,
        build: bool = False,
        memory: str = "4g",
        cpus: str = "2",
        pids: int = 2048,
        allow_ports: Sequence[int] = (80, 443),
        **_: Any,
    ) -> "DockerSandbox":
        image = ensure_image(build)
        sandbox_id = new_id("cmsbx-")
        agent = f"{sandbox_id}-agent"
        egress = f"{sandbox_id}-egress"
        inner = f"{sandbox_id}-inner"
        edge = f"{sandbox_id}-edge"
        labels = ["--label", f"{SANDBOX_LABEL}=1", "--label", f"{ID_LABEL}={sandbox_id}"]

        created_networks: list[str] = []
        created_containers: list[str] = []
        try:
            _run(["network", "create", "--internal", inner], check=True)
            created_networks.append(inner)
            _run(["network", "create", edge], check=True)
            created_networks.append(edge)

            egress_env: dict[str, str] = {
                "AOM_MODEL_URL": endpoint.openai_base_url if endpoint is not None else "",
                "AOM_MODEL_TOKEN": endpoint.token if endpoint is not None else "",
                "AOM_PROXY_SESSION": sandbox_id,
                "AOM_EGRESS_ALLOW_PORTS": ",".join(str(p) for p in allow_ports),
            }
            if proxy:
                egress_env["AOM_UPSTREAM_PROXY"] = proxy
            egress_args = ["run", "-d", "--name", egress, "--network", inner, *labels]
            for key, value in egress_env.items():
                egress_args += ["-e", f"{key}={value}"]
            egress_args += [image, "python3", "-m", "agents_on_modal.sandbox.net.egress"]
            _run(egress_args, check=True)
            created_containers.append(egress)
            # The broker needs a second interface with a route out; the agent never joins it.
            _run(["network", "connect", edge, egress], check=True)

            egress_ip = _network_ip(egress, inner)

            proxy_url = f"http://{egress_ip}:{PROXY_PORT}"
            no_proxy = f"{egress_ip},localhost,127.0.0.1"
            agent_env = {
                "AOM_INIT_FIREWALL": "1",
                "AOM_EGRESS_IP": egress_ip,
                "AOM_EGRESS_PROXY_PORT": str(PROXY_PORT),
                "AOM_EGRESS_MODEL_PORT": str(MODEL_PORT),
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "NO_PROXY": no_proxy,
                "no_proxy": no_proxy,
            }
            agent_args = [
                "run",
                "-d",
                "--name",
                agent,
                "--hostname",
                "aom-sandbox",
                "--network",
                inner,
                "--cap-add",
                "NET_ADMIN",
                "--memory",
                str(memory),
                "--cpus",
                str(cpus),
                "--pids-limit",
                str(pids),
                *labels,
            ]
            for key, value in agent_env.items():
                agent_args += ["-e", f"{key}={value}"]
            # No command: the image CMD (`sleep infinity`) idles the container; init.sh applies the
            # firewall and drops to the agent user first.
            agent_args.append(image)
            _run(agent_args, check=True)
            created_containers.append(agent)
        except Exception:
            for container in created_containers:
                _run(["rm", "-f", container], check=False, timeout=120.0)
            for network in created_networks:
                _run(["network", "rm", network], check=False, timeout=120.0)
            raise

        return cls(sandbox_id=sandbox_id, egress_ip=egress_ip, proxy=proxy)

    @classmethod
    def list(cls, **filters: Any) -> list["DockerSandbox"]:
        result = _run(
            ["ps", "--filter", f"label={SANDBOX_LABEL}=1", "--format", f'{{{{.Label "{ID_LABEL}"}}}}'],
            check=False,
        )
        seen: list[str] = []
        for line in result.stdout.splitlines():
            sandbox_id = line.strip()
            if sandbox_id and sandbox_id not in seen:
                seen.append(sandbox_id)
        boxes: list["DockerSandbox"] = []
        for sandbox_id in seen:
            try:
                boxes.append(cls.get(sandbox_id))
            except SandboxError:
                continue
        return boxes

    @classmethod
    def get(cls, sandbox_id: str) -> "DockerSandbox":
        agent = f"{sandbox_id}-agent"
        egress = f"{sandbox_id}-egress"
        inner = f"{sandbox_id}-inner"
        if _run(["inspect", "-f", "{{.Id}}", agent], check=False, timeout=120.0).returncode != 0:
            raise SandboxError(f"No docker sandbox {sandbox_id!r} (container {agent} not found).")
        egress_ip = _network_ip(egress, inner)
        proxy = _container_env(egress).get("AOM_UPSTREAM_PROXY") or None
        return cls(sandbox_id=sandbox_id, egress_ip=egress_ip, proxy=proxy)

    def stop(self) -> None:
        for container in (self.agent_container, self.egress_container):
            _run(["rm", "-f", container], check=False, timeout=120.0)
        for network in (self.inner_network, self.edge_network):
            _run(["network", "rm", network], check=False, timeout=120.0)

    def labels(self) -> dict[str, str]:
        return {SANDBOX_LABEL: "1", ID_LABEL: self.id}

    # ---- primitives -------------------------------------------------------------------------
    def _exec_argv(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None,
        cwd: str | None,
        root: bool,
        flags: Sequence[str] = (),
    ) -> list[str]:
        args = ["exec", *flags]
        if not root:
            args += ["-u", "10001:10001"]
        if cwd:
            args += ["-w", cwd]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        args.append(self.agent_container)
        args.extend(str(item) for item in argv)
        return args

    def exec(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 120,
        root: bool = False,
    ) -> ExecResult:
        args = self._exec_argv(argv, env=env, cwd=cwd, root=root)
        result = _run(args, check=False, timeout=timeout)
        return ExecResult(result.returncode, result.stdout or "", result.stderr or "")

    def spawn(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        # `-d` returns immediately; `setsid` gives the process its own session so it outlives the
        # exec client and keeps running detached.
        args = self._exec_argv(["setsid", *[str(a) for a in argv]], env=env, cwd=cwd, root=False, flags=["-d"])
        _run(args, check=True, timeout=120.0)

    def write_file(self, path: str, data: str | bytes) -> None:
        payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        parent = os.path.dirname(path) or "/"
        # Write as root (parents may not exist yet), then hand the file to the agent user.
        args = self._exec_argv(
            ["sh", "-c", f'mkdir -p "{parent}" && cat > "{path}"'],
            env=None,
            cwd=None,
            root=True,
            flags=["-i"],
        )
        result = _run(args, input=payload, text=False, check=False, timeout=120.0)
        if result.returncode != 0:
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()
            raise SandboxError(f"{self.agent_container}: could not write {path}: {detail}")
        chown = self._exec_argv(["chown", "10001:10001", path], env=None, cwd=None, root=True)
        _run(chown, check=False, timeout=60.0)

    def read_file(self, path: str, offset: int = 0) -> bytes:
        test = self._exec_argv(["test", "-f", path], env=None, cwd=None, root=False)
        if _run(test, check=False, timeout=60.0).returncode != 0:
            raise FileNotFoundError(path)
        args = self._exec_argv(
            ["tail", "-c", f"+{offset + 1}", path], env=None, cwd=None, root=False
        )
        result = _run(args, text=False, check=False, timeout=120.0)
        if result.returncode != 0:
            # The file existed a moment ago; treat a race as "not there".
            raise FileNotFoundError(path)
        return result.stdout or b""

    def attach(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> int:
        args = self._exec_argv(argv, env=env, cwd=cwd, root=False, flags=["-it"])
        return _run_interactive(args)

    # ---- model + profile --------------------------------------------------------------------
    def model_access(self, endpoint: Any) -> ModelAccess:
        return ModelAccess.from_endpoint(
            endpoint,
            base_url=f"http://{self.egress_ip}:{MODEL_PORT}",
            token=DUMMY_MODEL_TOKEN,
        )

    def profile(self) -> SandboxProfile:
        instructions = build_instructions(
            "docker",
            workdir=self.workdir,
            home=self.home,
            shared_dir=None,
            network="internet",
            proxy=bool(self.proxy),
        )
        playwright = McpServer(
            "playwright-mcp",
            (
                "--headless",
                "--isolated",
                "--browser",
                "chromium",
                "--executable-path",
                "/usr/bin/chromium",
                "--image-responses",
                "omit",
                "--output-dir",
                "/tmp/playwright-mcp",
            ),
        )
        return SandboxProfile(
            kind="docker",
            instructions=instructions,
            posix=True,
            skills=load_skills(),
            mcp_servers={"playwright": playwright},
            env={},
        )


__all__ = ["DockerError", "DockerSandbox", "ensure_image"]
