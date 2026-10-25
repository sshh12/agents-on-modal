"""The Docker sandbox backend, fully offline.

Every ``docker`` invocation goes through ``agents_on_modal.sandbox.docker._run`` (and interactive
sessions through ``_run_interactive``); the tests patch that one seam with a recording fake that
returns canned output, so nothing here touches a Docker daemon, the network, or the filesystem of a
container. The assertions check the exact docker argv the backend builds.
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest

from agents_on_modal.harness.base import ModelAccess, SandboxProfile
from agents_on_modal.sandbox import docker as dock
from agents_on_modal.sandbox.base import ExecResult, SandboxError


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------
def make_endpoint(url: str = "https://gw.example.com", token: str = "real-secret") -> types.SimpleNamespace:
    model = types.SimpleNamespace(
        name="qwen3.8-27b",
        context_window=32768,
        max_output_tokens=8192,
        reasoning=types.SimpleNamespace(default="high"),
        vision=False,
    )
    return types.SimpleNamespace(
        model=model,
        url=url,
        token=token,
        openai_base_url=f"{url}/v1",
    )


class FakeDocker:
    """Records every ``docker`` call and answers inspect/ps/read queries from canned state."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.inputs: list[bytes] = []
        self.built: list[dict] = []
        self.image_exists = True
        self.egress_ip = "10.88.0.2"
        self.ps_ids: list[str] = []
        self.env_json: dict[str, list[str]] = {}
        self.missing_containers: set[str] = set()
        self.file_exists = True
        self.read_payload = b"trace-bytes"
        self.exec_stdout = "ok"

    def __call__(self, args, *, input=None, text=True, check=False, timeout=None):
        args = [str(a) for a in args]
        self.calls.append(args)
        if input is not None:
            self.inputs.append(input)
        rc, out = self._dispatch(args, text)
        if check and rc != 0:
            raise dock.DockerError(f"fake docker {' '.join(args[:3])} exit {rc}")
        if text:
            return subprocess.CompletedProcess(args, rc, out if isinstance(out, str) else out.decode(), "")
        return subprocess.CompletedProcess(args, rc, out if isinstance(out, bytes) else out.encode(), b"")

    def _dispatch(self, args, text):
        head = args[0]
        if head == "image" and args[1:2] == ["inspect"]:
            return (0 if self.image_exists else 1), ""
        if head == "build":
            record = {"args": args, "context": args[-1]}
            if "-t" in args:
                record["tag"] = args[args.index("-t") + 1]
            if "-f" in args:
                path = args[args.index("-f") + 1]
                try:
                    record["dockerfile"] = Path(path).read_text(encoding="utf-8")
                except OSError:
                    record["dockerfile"] = None
            self.built.append(record)
            return 0, ""
        if head == "inspect":
            template = args[args.index("-f") + 1] if "-f" in args else ""
            container = args[-1]
            if "IPAddress" in template:
                return 0, self.egress_ip
            if "json .Config.Env" in template:
                return 0, json.dumps(self.env_json.get(container, []))
            if template == "{{.Id}}":
                return (1 if container in self.missing_containers else 0), "deadbeef"
            return 0, ""
        if head == "ps":
            return 0, "\n".join(self.ps_ids)
        if head == "exec":
            if "test" in args:
                return (0 if self.file_exists else 1), ""
            if "tail" in args:
                return (0 if self.file_exists else 1), (self.read_payload if not text else "")
            return 0, self.exec_stdout
        # run / network create|connect|rm / rm -f ...
        return 0, ""


@pytest.fixture
def fake(monkeypatch):
    recorder = FakeDocker()
    monkeypatch.setattr(dock, "_run", recorder)
    def fake_interactive(args):
        recorder([*args, "<interactive>"])
        return 0

    monkeypatch.setattr(dock, "_run_interactive", fake_interactive)
    return recorder


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
def env_of(args: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for index, token in enumerate(args):
        if token == "-e" and index + 1 < len(args):
            key, _, value = args[index + 1].partition("=")
            out[key] = value
    return out


def find(fake: FakeDocker, predicate) -> list[str]:
    for call in fake.calls:
        if predicate(call):
            return call
    raise AssertionError(f"no docker call matched; calls were:\n" + "\n".join(map(str, fake.calls)))


def make_box(sandbox_id: str = "cmsbx-test", egress_ip: str = "10.88.0.2", proxy=None) -> dock.DockerSandbox:
    return dock.DockerSandbox(sandbox_id=sandbox_id, egress_ip=egress_ip, proxy=proxy)


# --------------------------------------------------------------------------------------------
# ensure_image
# --------------------------------------------------------------------------------------------
def test_ensure_image_builds_base_and_derived_when_absent(fake):
    fake.image_exists = False
    tag = dock.ensure_image(build=False)

    base = dock._base_tag()
    assert tag == "aom-sandbox:" + base.split(":", 1)[1]
    assert len(fake.built) == 2

    base_build, derived_build = fake.built
    assert base_build["tag"] == base
    assert base_build["context"] == str(dock._IMAGE_DIR)
    assert base_build["args"][base_build["args"].index("-f") + 1] == str(dock._DOCKERFILE)

    assert derived_build["tag"] == tag
    assert derived_build["context"] == str(dock._REPO_ROOT)
    assert derived_build["dockerfile"] == f"FROM {base}\nCOPY agents_on_modal /opt/aom/agents_on_modal\n"


def test_ensure_image_skips_base_when_present(fake):
    fake.image_exists = True
    dock.ensure_image(build=False)
    # Only the thin derived image is built; the base is reused.
    assert [b["tag"] for b in fake.built] == ["aom-sandbox:" + dock._base_tag().split(":", 1)[1]]
    assert all(call[:2] != ["build", "-t"] or call[2].startswith("aom-sandbox:") for call in fake.calls)


def test_ensure_image_force_rebuilds_base(fake):
    fake.image_exists = True
    dock.ensure_image(build=True)
    tags = [b["tag"] for b in fake.built]
    assert tags[0] == dock._base_tag()
    assert len(tags) == 2


def test_ensure_image_removes_temp_dockerfile(fake):
    fake.image_exists = True
    dock.ensure_image()
    path = fake.built[-1]["args"][fake.built[-1]["args"].index("-f") + 1]
    assert not Path(path).exists()


# --------------------------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------------------------
def test_create_wires_networks(fake):
    box = dock.DockerSandbox.create(endpoint=make_endpoint())

    inner = f"{box.id}-inner"
    edge = f"{box.id}-edge"
    internal = find(fake, lambda c: c[:2] == ["network", "create"] and "--internal" in c and inner in c)
    assert internal[-1] == inner
    edge_create = find(fake, lambda c: c[:2] == ["network", "create"] and "--internal" not in c and edge in c)
    assert edge_create[-1] == edge
    # Egress joins the edge network as a second interface.
    find(fake, lambda c: c[:2] == ["network", "connect"] and edge in c and f"{box.id}-egress" in c)


def test_create_starts_egress_with_model_env(fake):
    endpoint = make_endpoint(url="https://gw.example.com", token="SECRET-KEY")
    box = dock.DockerSandbox.create(endpoint=endpoint)

    egress = find(
        fake,
        lambda c: c[:2] == ["run", "-d"] and f"{box.id}-egress" in c and "agents_on_modal.sandbox.net.egress" in c,
    )
    env = env_of(egress)
    assert env["AOM_MODEL_URL"] == "https://gw.example.com/v1"
    assert env["AOM_MODEL_TOKEN"] == "SECRET-KEY"
    assert env["AOM_PROXY_SESSION"] == box.id
    assert env["AOM_EGRESS_ALLOW_PORTS"] == "80,443"
    assert "AOM_UPSTREAM_PROXY" not in env
    # Egress runs on the inner network and is labelled.
    assert "--internal" not in egress
    assert egress[egress.index("--network") + 1] == f"{box.id}-inner"
    assert "aom.sandbox=1" in egress


def test_create_starts_agent_firewalled(fake):
    fake.egress_ip = "10.88.0.2"
    box = dock.DockerSandbox.create(endpoint=make_endpoint(), memory="8g", cpus="4", pids=1024)

    agent = find(fake, lambda c: c[:2] == ["run", "-d"] and f"{box.id}-agent" in c)
    env = env_of(agent)
    assert env["AOM_INIT_FIREWALL"] == "1"
    assert env["AOM_EGRESS_IP"] == "10.88.0.2"
    assert env["AOM_EGRESS_PROXY_PORT"] == "3128"
    assert env["AOM_EGRESS_MODEL_PORT"] == "8081"
    assert env["HTTP_PROXY"] == "http://10.88.0.2:3128"
    assert env["HTTPS_PROXY"] == "http://10.88.0.2:3128"
    assert env["NO_PROXY"] == "10.88.0.2,localhost,127.0.0.1"
    assert "NET_ADMIN" in agent
    assert agent[agent.index("--network") + 1] == f"{box.id}-inner"
    assert agent[agent.index("--memory") + 1] == "8g"
    assert agent[agent.index("--cpus") + 1] == "4"
    assert agent[agent.index("--pids-limit") + 1] == "1024"
    # No explicit command: the image CMD (sleep infinity) idles it.
    assert agent[-1].startswith("aom-sandbox:")
    assert box.egress_ip == "10.88.0.2"


def test_create_allow_ports_formatting(fake):
    box = dock.DockerSandbox.create(endpoint=make_endpoint(), allow_ports=(80, 443, 8080))
    egress = find(fake, lambda c: f"{box.id}-egress" in c and "agents_on_modal.sandbox.net.egress" in c)
    assert env_of(egress)["AOM_EGRESS_ALLOW_PORTS"] == "80,443,8080"


def test_create_threads_proxy_into_egress(fake):
    box = dock.DockerSandbox.create(endpoint=make_endpoint(), proxy="socks5h://user:pass@res.example:1080")
    egress = find(fake, lambda c: f"{box.id}-egress" in c and "agents_on_modal.sandbox.net.egress" in c)
    assert env_of(egress)["AOM_UPSTREAM_PROXY"] == "socks5h://user:pass@res.example:1080"
    assert box.proxy == "socks5h://user:pass@res.example:1080"


def test_create_cleans_up_on_failure(monkeypatch):
    recorder = FakeDocker()
    original = recorder._dispatch

    def boom(args, text):
        if args[:2] == ["run", "-d"] and any("-agent" in a for a in args):
            return 1, "no space left"  # agent run fails
        return original(args, text)

    recorder._dispatch = boom
    monkeypatch.setattr(dock, "_run", recorder)

    with pytest.raises(dock.DockerError):
        dock.DockerSandbox.create(endpoint=make_endpoint())
    # The egress container and both networks must be torn down.
    removed = [c for c in recorder.calls if c[:1] == ["rm"] or c[:2] == ["network", "rm"]]
    assert any("-egress" in " ".join(c) for c in removed)
    assert any(c[:2] == ["network", "rm"] for c in removed)


# --------------------------------------------------------------------------------------------
# model_access + attributes
# --------------------------------------------------------------------------------------------
def test_model_access_points_at_egress_with_dummy_token():
    box = make_box(egress_ip="10.5.0.9")
    access = box.model_access(make_endpoint(token="real-secret"))
    assert isinstance(access, ModelAccess)
    assert access.base_url == "http://10.5.0.9:8081"
    assert access.token == dock.DUMMY_MODEL_TOKEN
    assert access.token != "real-secret"
    assert access.model_id == "qwen3.8-27b"


def test_instance_attributes():
    box = make_box()
    assert box.kind == "docker"
    assert box.workdir == "/work"
    assert box.home == "/home/agent"
    assert box.runs_root == "/home/agent/.aom/runs"
    assert box.posix is True
    assert box.python == ("python3",)
    assert box.labels() == {"aom.sandbox": "1", "aom.id": "cmsbx-test"}


# --------------------------------------------------------------------------------------------
# exec / spawn / write_file / read_file / attach
# --------------------------------------------------------------------------------------------
def test_exec_builds_argv_as_agent(fake):
    box = make_box()
    result = box.exec(["echo", "hi"], env={"FOO": "bar"}, cwd="/work", timeout=30)
    assert isinstance(result, ExecResult)
    call = find(fake, lambda c: c[:1] == ["exec"] and "echo" in c)
    assert call[:2] == ["exec", "-u"]
    assert "10001:10001" in call
    assert call[call.index("-w") + 1] == "/work"
    assert "FOO=bar" in call
    assert call[call.index(box.agent_container) + 1:] == ["echo", "hi"]


def test_exec_root_omits_user(fake):
    box = make_box()
    box.exec(["apt-get", "update"], root=True)
    call = find(fake, lambda c: c[:1] == ["exec"] and "apt-get" in c)
    assert "-u" not in call


def test_spawn_is_detached_setsid(fake):
    box = make_box()
    box.spawn(["python3", "-m", "agents_on_modal.runner", "/run/dir"], cwd="/work")
    call = find(fake, lambda c: c[:1] == ["exec"] and "setsid" in c)
    assert "-d" in call
    assert "10001:10001" in call  # runs as the agent user
    tail = call[call.index(box.agent_container) + 1:]
    assert tail[0] == "setsid"
    assert tail[1:] == ["python3", "-m", "agents_on_modal.runner", "/run/dir"]


def test_write_file_tees_as_root_then_chowns(fake):
    box = make_box()
    box.write_file("/home/agent/.aom/runs/r/spec.json", "hello")
    write = find(fake, lambda c: c[:1] == ["exec"] and any("cat >" in a for a in c))
    assert "-i" in write
    assert "-u" not in write  # written as root so parents + chown work
    shell = write[-1]
    assert 'mkdir -p "/home/agent/.aom/runs/r"' in shell
    assert fake.inputs[-1] == b"hello"
    # chown runs as root and walks up chowning root-owned parent dirs to the agent, so the
    # agent-user runner can write status.json/trace.jsonl into the run dir it did not create.
    chown = find(fake, lambda c: c[:1] == ["exec"] and any("chown 10001:10001" in a for a in c))
    assert "-u" not in chown
    script = chown[-1]
    assert "chown 10001:10001" in script
    assert "stat -c %u" in script


def test_write_file_accepts_bytes(fake):
    box = make_box()
    box.write_file("/work/data.bin", b"\x00\x01\x02")
    assert fake.inputs[-1] == b"\x00\x01\x02"


def test_read_file_tails_from_offset(fake):
    fake.file_exists = True
    fake.read_payload = b"abcdef"
    box = make_box()
    data = box.read_file("/home/agent/.aom/runs/r/trace.jsonl", offset=10)
    assert data == b"abcdef"
    find(fake, lambda c: c[:1] == ["exec"] and "test" in c and "-f" in c)
    tail = find(fake, lambda c: c[:1] == ["exec"] and "tail" in c)
    assert tail[tail.index("-c") + 1] == "+11"  # offset 10 -> 1-based +11


def test_read_file_offset_zero_reads_whole_file(fake):
    box = make_box()
    box.read_file("/work/f", offset=0)
    tail = find(fake, lambda c: "tail" in c)
    assert tail[tail.index("-c") + 1] == "+1"


def test_read_file_missing_raises(fake):
    fake.file_exists = False
    box = make_box()
    with pytest.raises(FileNotFoundError):
        box.read_file("/work/missing")


def test_attach_uses_interactive_tty(fake):
    box = make_box()
    code = box.attach(["bash"], cwd="/work")
    assert code == 0
    call = find(fake, lambda c: c[:1] == ["exec"] and "-it" in c and "bash" in c)
    assert "10001:10001" in call
    assert call[call.index("-w") + 1] == "/work"


# --------------------------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------------------------
def test_profile_has_playwright_and_internet():
    box = make_box()
    profile = box.profile()
    assert isinstance(profile, SandboxProfile)
    assert profile.kind == "docker"
    assert "playwright" in profile.mcp_servers
    server = profile.mcp_servers["playwright"]
    assert server.command == "playwright-mcp"
    assert "--headless" in server.args
    assert "--executable-path" in server.args and "/usr/bin/chromium" in server.args
    assert "--output-dir" in server.args and "/tmp/playwright-mcp" in server.args
    assert "Full outbound internet" in profile.instructions
    assert "host machine and its LAN are" in profile.instructions
    assert profile.skills  # shipped skills are loaded


def test_profile_proxy_line_toggles():
    assert "routed through an upstream proxy" not in make_box(proxy=None).profile().instructions
    assert "routed through an upstream proxy" in make_box(proxy="socks5h://h:1").profile().instructions


# --------------------------------------------------------------------------------------------
# list / get / stop
# --------------------------------------------------------------------------------------------
def test_stop_removes_containers_and_networks(fake):
    box = make_box("cmsbx-xyz")
    box.stop()
    joined = [" ".join(c) for c in fake.calls]
    assert any("rm -f cmsbx-xyz-agent" in j for j in joined)
    assert any("rm -f cmsbx-xyz-egress" in j for j in joined)
    assert any("network rm cmsbx-xyz-inner" in j for j in joined)
    assert any("network rm cmsbx-xyz-edge" in j for j in joined)


def test_stop_is_idempotent(fake):
    # Simulate "no such container/network": every rm returns nonzero, and stop must not raise.
    fake._dispatch = lambda args, text: (1, "Error: No such container")
    box = make_box("cmsbx-gone")
    box.stop()
    box.stop()


def test_get_reconstructs_handle(fake):
    fake.egress_ip = "10.7.0.3"
    fake.env_json = {"cmsbx-a-egress": ["AOM_UPSTREAM_PROXY=socks5h://u:p@h:1080", "AOM_MODEL_URL=x"]}
    box = dock.DockerSandbox.get("cmsbx-a")
    assert box.id == "cmsbx-a"
    assert box.egress_ip == "10.7.0.3"
    assert box.proxy == "socks5h://u:p@h:1080"


def test_get_missing_raises(fake):
    fake.missing_containers = {"cmsbx-nope-agent"}
    with pytest.raises(SandboxError):
        dock.DockerSandbox.get("cmsbx-nope")


def test_list_dedupes_ids_and_reconstructs(fake):
    # ps prints the aom.id label once per matching container (agent + egress) -> duplicates.
    fake.ps_ids = ["cmsbx-a", "cmsbx-a", "cmsbx-b", "cmsbx-b"]
    boxes = dock.DockerSandbox.list()
    assert sorted(b.id for b in boxes) == ["cmsbx-a", "cmsbx-b"]
    ps = find(fake, lambda c: c[:1] == ["ps"])
    assert "label=aom.sandbox=1" in ps
