"""Offline unit tests for the Modal sandbox backend.

A fake ``modal`` module is injected into ``sys.modules`` so nothing touches the real Modal service,
no Docker/CLIs run, and no credits are spent. The backend imports ``modal`` lazily inside each
method, so the injected fake is what every method sees.
"""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

import pytest

import agents_on_modal.sandbox.modal as modal_backend
from agents_on_modal.sandbox.modal import APP_NAME, ModalSandbox


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------
class _NotFoundError(Exception):
    pass


class _FsNotFound(Exception):
    pass


class _Reader:
    def __init__(self, data: str) -> None:
        self._data = data

    def read(self) -> str:
        return self._data


class FakeProc:
    def __init__(self, out: str = "", err: str = "", returncode: int = 0) -> None:
        self.stdout = _Reader(out)
        self.stderr = _Reader(err)
        self.returncode = returncode
        self.attached = False

    def wait(self) -> int:
        return self.returncode

    def attach(self) -> None:
        self.attached = True


class FakeFilesystem:
    def __init__(self) -> None:
        self.writes: list[tuple[bytes, str]] = []
        self.files: dict[str, bytes] = {}

    def write_bytes(self, data: bytes, path: str) -> None:
        self.writes.append((bytes(data), path))
        self.files[path] = bytes(data)

    def read_bytes(self, path: str) -> bytes:
        if path not in self.files:
            raise _FsNotFound(path)
        return self.files[path]


class FakeSandbox:
    def __init__(self, name: str, tags: dict, object_id: str, create_kwargs: dict) -> None:
        self.name = name
        self._tags = dict(tags)
        self.object_id = object_id
        self.create_kwargs = create_kwargs
        self.exec_calls: list[dict] = []
        self.filesystem = FakeFilesystem()
        self.terminated = False

    def exec(self, *args, env=None, workdir=None, timeout=None, pty=False):
        self.exec_calls.append(
            {"args": args, "env": env, "workdir": workdir, "timeout": timeout, "pty": pty}
        )
        return FakeProc()

    def get_tags(self) -> dict:
        return dict(self._tags)

    def terminate(self, wait: bool = False) -> int:
        self.terminated = True
        return 137


class FakeImage:
    def __init__(self, ref: str | None = None, dockerfile: str | None = None) -> None:
        self.ref = ref
        self.dockerfile = dockerfile
        self.added: list[tuple] = []
        self.built = False
        self.published: str | None = None

    def add_local_dir(self, local_path, remote_path, copy=False):
        self.added.append((local_path, remote_path, copy))
        return self

    def build(self, app):
        self.built = True
        return self

    def publish(self, ref, **_):
        self.published = ref


class FakeVolume:
    def __init__(self, name, version, create_if_missing) -> None:
        self.name = name
        self.version = version
        self.create_if_missing = create_if_missing


def build_fake_modal(*, image_present: bool = True):
    """Return a fake ``modal`` module plus a shared ``state`` object for assertions."""

    state = SimpleNamespace(
        created=[],            # list of (args, kwargs) for Sandbox.create
        registry={},           # name -> FakeSandbox
        tokens_created=[],     # names passed to proxy_tokens.create
        tokens_deleted=[],     # ids passed to proxy_tokens.delete
        volumes=[],            # FakeVolume instances
        list_calls=[],         # tags dicts passed to Sandbox.list
        from_dockerfile=[],    # dockerfile paths
        context_dirs=[],       # context_dir passed to from_dockerfile
        images_built=[],
        app_lookups=[],
        image_present=image_present,
    )

    token_counter = {"n": 0}

    # ---- App
    def app_lookup(name, create_if_missing=False):
        state.app_lookups.append((name, create_if_missing))
        return SimpleNamespace(app_id=f"ap-{name}", name=name)

    # ---- Image
    def image_from_name(ref):
        if not state.image_present:
            raise _NotFoundError(ref)
        return FakeImage(ref=ref)

    def image_from_dockerfile(path, *, context_dir=None, **_):
        state.from_dockerfile.append(path)
        state.context_dirs.append(context_dir)
        img = FakeImage(dockerfile=path)
        state.images_built.append(img)
        return img

    # ---- Sandbox
    def sandbox_create(*args, **kwargs):
        state.created.append((args, kwargs))
        sb = FakeSandbox(
            name=kwargs.get("name"),
            tags=kwargs.get("tags") or {},
            object_id=f"sb-{kwargs.get('name')}",
            create_kwargs=kwargs,
        )
        if sb.name:
            state.registry[sb.name] = sb
        return sb

    def sandbox_list(app_id=None, tags=None):
        state.list_calls.append(dict(tags or {}))
        want = dict(tags or {})
        for sb in state.registry.values():
            if all(sb._tags.get(k) == v for k, v in want.items()):
                yield sb

    def sandbox_from_name(app_name, name, **_):
        if name in state.registry:
            return state.registry[name]
        raise _NotFoundError(name)

    def sandbox_from_id(sandbox_id, **_):
        for sb in state.registry.values():
            if sb.object_id == sandbox_id:
                return sb
        raise _NotFoundError(sandbox_id)

    # ---- Volume
    def volume_from_name(name, version=None, create_if_missing=False):
        vol = FakeVolume(name, version, create_if_missing)
        state.volumes.append(vol)
        return vol

    # ---- Workspace / proxy tokens
    class FakeProxyTokens:
        def create(self):
            token_counter["n"] += 1
            n = token_counter["n"]
            state.tokens_created.append(f"wk-{n}")
            return SimpleNamespace(token_id=f"wk-{n}", token_secret=f"ws-{n}")

        def delete(self, token_id):
            state.tokens_deleted.append(token_id)

    def workspace_from_context():
        return SimpleNamespace(proxy_tokens=FakeProxyTokens())

    @contextlib.contextmanager
    def enable_output():
        yield

    modal = SimpleNamespace(
        App=SimpleNamespace(lookup=app_lookup),
        Image=SimpleNamespace(from_name=image_from_name, from_dockerfile=image_from_dockerfile),
        Sandbox=SimpleNamespace(
            create=sandbox_create,
            list=sandbox_list,
            from_name=sandbox_from_name,
            from_id=sandbox_from_id,
        ),
        Volume=SimpleNamespace(from_name=volume_from_name),
        Workspace=SimpleNamespace(from_context=workspace_from_context),
        enable_output=enable_output,
        exception=SimpleNamespace(
            NotFoundError=_NotFoundError,
            SandboxFilesystemNotFoundError=_FsNotFound,
        ),
    )
    return modal, state


def make_endpoint():
    model = SimpleNamespace(
        name="qwen3-27b",
        context_window=32768,
        max_output_tokens=4096,
        reasoning=SimpleNamespace(default="high"),
        vision=False,
    )
    return SimpleNamespace(
        model=model,
        url="https://ws--aom-qwen.us-west.modal.direct",
        token="wk-user.ws-usersecret",
    )


@pytest.fixture
def fake_modal(monkeypatch):
    modal, state = build_fake_modal()
    monkeypatch.setitem(sys.modules, "modal", modal)
    return state


# --------------------------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------------------------
def test_create_mints_token_and_sets_sandbox_args(fake_modal):
    endpoint = make_endpoint()
    box = ModalSandbox.create(endpoint)

    # exactly one run-scoped token minted (1.6.2 create() takes no name)
    assert fake_modal.tokens_created == ["wk-1"]
    assert box.token == "wk-1.ws-1"

    # exactly one sandbox created with the right shape
    assert len(fake_modal.created) == 1
    args, kwargs = fake_modal.created[0]
    assert args == ("sleep", "infinity")
    assert kwargs["name"] == box.id
    assert kwargs["workdir"] == "/work"
    assert kwargs["timeout"] == 6 * 3600
    assert isinstance(kwargs["image"], FakeImage)
    assert kwargs["encrypted_ports"] == []
    assert kwargs["volumes"] == {}
    tags = kwargs["tags"]
    assert tags["aom"] == "1"
    assert tags["aom_id"] == box.id
    assert tags["proxy_token_id"] == "wk-1"
    assert tags["swarm"] == ""
    assert kwargs["env"]["IS_SANDBOX"] == "1"


def test_python_prefix_and_attrs(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    assert box.python == ("gosu", "agent", "python3")
    assert box.workdir == "/work"
    assert box.home == "/home/agent"
    assert box.runs_root == "/home/agent/.aom/runs"
    assert box.posix is True
    assert box.kind == "modal"


def test_explicit_token_skips_minting(fake_modal):
    box = ModalSandbox.create(make_endpoint(), token="wk-x.ws-y")
    assert fake_modal.tokens_created == []
    assert box.token == "wk-x.ws-y"
    assert box._proxy_token_id == ""


def test_swarm_tag_recorded(fake_modal):
    box = ModalSandbox.create(make_endpoint(), swarm="swarm-42")
    _, kwargs = fake_modal.created[0]
    assert kwargs["tags"]["swarm"] == "swarm-42"
    assert box._shared is False


# --------------------------------------------------------------------------------------------
# exec / spawn / attach
# --------------------------------------------------------------------------------------------
def test_exec_wraps_gosu_and_root_does_not(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    box._sb.exec_calls.clear()

    result = box.exec(["echo", "hi"])
    assert box._sb.exec_calls[-1]["args"] == ("gosu", "agent", "echo", "hi")
    assert box._sb.exec_calls[-1]["workdir"] == "/work"
    assert result.returncode == 0

    box.exec(["id"], root=True)
    assert box._sb.exec_calls[-1]["args"] == ("id",)

    box.exec(["gosu", "agent", "whoami"])
    assert box._sb.exec_calls[-1]["args"] == ("gosu", "agent", "whoami")


def test_exec_passes_cwd_and_timeout(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    box._sb.exec_calls.clear()
    box.exec(["ls"], cwd="/tmp", timeout=30)
    call = box._sb.exec_calls[-1]
    assert call["workdir"] == "/tmp"
    assert call["timeout"] == 30


def test_spawn_detached_setsid_nohup(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    box._sb.exec_calls.clear()
    result = box.spawn(["gosu", "agent", "python3", "-m", "agents_on_modal.runner", "/run/x"])
    assert result is None
    call = box._sb.exec_calls[-1]
    assert call["args"][0:2] == ("bash", "-lc")
    cmd = call["args"][2]
    assert cmd.startswith("setsid nohup ")
    assert cmd.endswith(">/tmp/aom-spawn.log 2>&1 &")
    assert "gosu agent python3 -m agents_on_modal.runner /run/x" in cmd


def test_attach_uses_pty_and_returns_code(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    box._sb.exec_calls.clear()
    code = box.attach(["bash"])
    call = box._sb.exec_calls[-1]
    assert call["pty"] is True
    assert call["args"] == ("gosu", "agent", "bash")
    assert code == 0


# --------------------------------------------------------------------------------------------
# write_file / read_file
# --------------------------------------------------------------------------------------------
def test_write_file_writes_then_chowns(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    box._sb.exec_calls.clear()
    path = "/home/agent/.aom/runs/run-1/spec.json"
    box.write_file(path, "{}")
    assert box._sb.filesystem.writes[-1] == (b"{}", path)
    chown = box._sb.exec_calls[-1]
    assert chown["args"][0:2] == ("bash", "-lc")
    assert "chown 10001:10001" in chown["args"][2]
    assert path in chown["args"][2]


def test_write_file_accepts_bytes(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    box.write_file("/tmp/x.bin", b"\x00\x01")
    assert box._sb.filesystem.files["/tmp/x.bin"] == b"\x00\x01"


def test_read_file_offset(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    box._sb.filesystem.files["/tmp/t"] = b"0123456789"
    assert box.read_file("/tmp/t") == b"0123456789"
    assert box.read_file("/tmp/t", offset=4) == b"456789"


def test_read_file_missing_maps_to_filenotfound(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    with pytest.raises(FileNotFoundError):
        box.read_file("/tmp/does-not-exist")


# --------------------------------------------------------------------------------------------
# model access
# --------------------------------------------------------------------------------------------
def test_model_access_is_direct_with_scoped_token(fake_modal):
    endpoint = make_endpoint()
    box = ModalSandbox.create(endpoint)
    access = box.model_access(endpoint)
    assert access.base_url == endpoint.url          # direct gateway root, no egress
    assert access.token == box.token == "wk-1.ws-1"  # the run-scoped bearer
    assert access.model_id == "qwen3-27b"
    assert access.context_window == 32768


# --------------------------------------------------------------------------------------------
# proxy / relay
# --------------------------------------------------------------------------------------------
PROXY = "socks5h://user-{session}:pw@residential.example:1080"


def test_proxy_sets_relay_env_and_keeps_model_direct(fake_modal):
    endpoint = make_endpoint()
    box = ModalSandbox.create(endpoint, proxy=PROXY)
    _, kwargs = fake_modal.created[0]
    env = kwargs["env"]
    assert env["HTTP_PROXY"] == "http://127.0.0.1:3128"
    assert env["ALL_PROXY"] == "socks5://127.0.0.1:1080"
    # the model host stays direct
    assert "ws--aom-qwen.us-west.modal.direct" in env["NO_PROXY"]
    assert box.proxy is not None


def test_proxy_starts_relay_in_sandbox(fake_modal):
    endpoint = make_endpoint()
    box = ModalSandbox.create(endpoint, proxy=PROXY)
    # the last exec during create launches the relay
    relay_call = box._sb.exec_calls[-1]
    assert relay_call["args"][0:2] == ("bash", "-lc")
    assert "agents_on_modal.sandbox.net.relay" in relay_call["args"][2]
    assert relay_call["env"]["AOM_UPSTREAM_PROXY"] == PROXY          # raw, {session} unsubstituted
    # the session is sanitized to [A-Za-z0-9] before the relay sees it: residential providers split
    # the username on '-', so a sandbox id's dashes would corrupt the sticky session. It stays a
    # deterministic function of box.id (dashes stripped) so each sandbox keeps a stable distinct IP.
    import re as _re
    assert relay_call["env"]["AOM_PROXY_SESSION"] == _re.sub(r"[^A-Za-z0-9]", "", box.id)[:48]
    assert "-" not in relay_call["env"]["AOM_PROXY_SESSION"]
    assert relay_call["env"]["AOM_MODEL_HOST"] == "ws--aom-qwen.us-west.modal.direct"


def test_no_proxy_means_no_relay(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    assert box.proxy is None
    assert box._sb.exec_calls == []  # nothing exec'd during a plain create


def test_profile_advertises_proxy_and_browser(fake_modal):
    box = ModalSandbox.create(make_endpoint(), proxy=PROXY)
    profile = box.profile()
    assert profile.kind == "modal"
    assert "playwright" in profile.mcp_servers
    assert profile.env["HTTP_PROXY"] == "http://127.0.0.1:3128"
    assert "upstream proxy" in profile.instructions.lower()
    assert profile.skills  # shipped SKILL.md files


# --------------------------------------------------------------------------------------------
# shared volume
# --------------------------------------------------------------------------------------------
def test_shared_volume_mounts_and_profile_advertises(fake_modal):
    box = ModalSandbox.create(make_endpoint(), shared_volume="aom-swarm-vol")
    _, kwargs = fake_modal.created[0]
    assert "/shared" in kwargs["volumes"]
    vol = fake_modal.volumes[-1]
    assert vol.name == "aom-swarm-vol"
    assert vol.version == 2
    assert vol.create_if_missing is True
    assert box._shared is True
    assert "/shared" in box.profile().instructions


# --------------------------------------------------------------------------------------------
# list / get / stop
# --------------------------------------------------------------------------------------------
def test_list_filters_by_swarm(fake_modal):
    ModalSandbox.create(make_endpoint(), swarm="alpha")
    ModalSandbox.create(make_endpoint(), swarm="beta")

    all_boxes = ModalSandbox.list()
    assert len(all_boxes) == 2
    assert fake_modal.list_calls[-1] == {"aom": "1"}

    alpha = ModalSandbox.list(swarm="alpha")
    assert len(alpha) == 1
    assert alpha[0].labels()["swarm"] == "alpha"
    assert fake_modal.list_calls[-1] == {"aom": "1", "swarm": "alpha"}


def test_list_empty_when_app_missing(monkeypatch):
    modal, _ = build_fake_modal()

    def missing_lookup(name, create_if_missing=False):
        if not create_if_missing:
            raise _NotFoundError(name)
        return SimpleNamespace(app_id=f"ap-{name}", name=name)

    modal.App = SimpleNamespace(lookup=missing_lookup)
    monkeypatch.setitem(sys.modules, "modal", modal)
    assert ModalSandbox.list() == []


def test_get_reconstructs_handle(fake_modal):
    created = ModalSandbox.create(make_endpoint())
    got = ModalSandbox.get(created.id)
    assert got.id == created.id
    assert got.token is None                     # secret is not recoverable from tags
    assert got._proxy_token_id == "wk-1"         # but the token id is, for teardown


def test_stop_terminates_and_deletes_token(fake_modal):
    box = ModalSandbox.create(make_endpoint())
    sb = box._sb
    box.stop()
    assert sb.terminated is True
    assert fake_modal.tokens_deleted == ["wk-1"]
    # idempotent: a second stop neither errors nor double-deletes
    box.stop()
    assert fake_modal.tokens_deleted == ["wk-1"]


def test_rediscovered_handle_can_delete_token(fake_modal):
    created = ModalSandbox.create(make_endpoint())
    got = ModalSandbox.get(created.id)
    got.stop()
    assert got._sb.terminated is True
    assert fake_modal.tokens_deleted == ["wk-1"]


# --------------------------------------------------------------------------------------------
# image
# --------------------------------------------------------------------------------------------
def test_ensure_image_hot_path_no_build(fake_modal):
    image = ModalSandbox.ensure_image()
    assert isinstance(image, FakeImage)
    assert fake_modal.from_dockerfile == []       # never built on the hot path
    assert fake_modal.images_built == []


def test_ensure_image_builds_when_missing(monkeypatch):
    modal, state = build_fake_modal(image_present=False)
    monkeypatch.setitem(sys.modules, "modal", modal)
    image = ModalSandbox.ensure_image()
    assert len(state.from_dockerfile) == 1
    built = state.images_built[-1]
    assert built.built is True
    assert built.published is not None
    assert built.added and built.added[-1][1] == "/opt/aom/agents_on_modal"
    assert built.added[-1][2] is True             # copy=True bakes the package into a layer


def test_ensure_image_build_flag_forces_build(fake_modal):
    ModalSandbox.ensure_image(build=True)
    assert len(fake_modal.from_dockerfile) == 1
    assert fake_modal.images_built[-1].built is True


def test_app_lookup_creates_if_missing(fake_modal):
    ModalSandbox.create(make_endpoint())
    assert (APP_NAME, True) in fake_modal.app_lookups
