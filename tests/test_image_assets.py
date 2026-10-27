"""Tests for the static image assets: Dockerfile, init.sh, skills, README.

These assert structure, pinned versions, and LF/POSIX hygiene without ever
building the image (no docker, no network).
"""

from __future__ import annotations

from pathlib import Path

import pytest

IMAGE = Path(__file__).resolve().parent.parent / "agents_on_modal" / "sandbox" / "image"


def _read_bytes(name: str) -> bytes:
    return (IMAGE / name).read_bytes()


def _read_text(name: str) -> str:
    return (IMAGE / name).read_text(encoding="utf-8")


ASSETS = ["Dockerfile", "init.sh", "README.md",
          "skills/goal/SKILL.md", "skills/browser/SKILL.md"]


@pytest.mark.parametrize("name", ASSETS)
def test_asset_exists(name):
    assert (IMAGE / name).is_file(), f"missing {name}"


@pytest.mark.parametrize("name", ASSETS)
def test_asset_is_lf_only(name):
    data = _read_bytes(name)
    assert b"\r" not in data, f"{name} contains CR (must be LF-only)"


@pytest.mark.parametrize("name", ["Dockerfile", "init.sh"])
def test_no_backslash_paths(name):
    # Files that run in Linux containers must use POSIX paths.
    text = _read_text(name)
    assert "C:\\" not in text
    assert "\\work" not in text


def test_dockerfile_base_and_structure():
    df = _read_text("Dockerfile")
    assert "FROM node:24-trixie-slim" in df
    # execs its args via the init entrypoint
    assert '/opt/aom-init.sh' in df
    assert "ENTRYPOINT" in df
    assert "COPY init.sh /opt/aom-init.sh" in df
    # non-root agent user, uid 10001, passwordless sudo
    assert "--uid 10001" in df
    assert "useradd" in df
    assert "agent ALL=(ALL) NOPASSWD:ALL" in df
    # work + aom state dirs
    assert "/work" in df
    assert "/home/agent/.aom" in df
    # aom package added by the builder, not here
    assert "PYTHONPATH=/opt/aom" in df


def test_dockerfile_pins():
    df = _read_text("Dockerfile")
    for pin in (
        "CLAUDE_CODE_VERSION=2.1.310",
        "CODEX_VERSION=0.155.1",
        "PW_MCP_VERSION=0.0.83",
        "AGENT_BROWSER_VERSION=0.39.2",
        "PY_PLAYWRIGHT_VERSION=1.64.1",
        "DSH_SDK_VERSION=0.2.1rc2",
    ):
        assert pin in df, f"missing pin {pin}"
    # npm globals reference the pinned ARGs
    assert "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" in df
    assert "@openai/codex@${CODEX_VERSION}" in df
    assert "@playwright/mcp@${PW_MCP_VERSION}" in df
    assert "agent-browser@${AGENT_BROWSER_VERSION}" in df
    assert "playwright==${PY_PLAYWRIGHT_VERSION}" in df
    # the DeepSeek Harness SDK (the `dsh` harness) is pip-installed system-wide
    assert "deepseek-harness-sdk==${DSH_SDK_VERSION}" in df


def test_dockerfile_chromium_symlink_and_flag():
    df = _read_text("Dockerfile")
    assert "ln -sf /usr/bin/chromium /opt/google/chrome/chrome" in df
    assert "/etc/chromium.d/zz-container" in df
    assert "--no-sandbox" in df


def test_dockerfile_env():
    df = _read_text("Dockerfile")
    for env in (
        "PLAYWRIGHT_MCP_EXECUTABLE_PATH=/usr/bin/chromium",
        "PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium",
        "DISABLE_AUTOUPDATER=1",
        "ENABLE_TOOL_SEARCH=false",
    ):
        assert env in df, f"missing env {env}"
    # rustup, go tarball, gh apt repo present
    assert "rustup" in df or "sh.rustup.rs" in df
    assert "go.dev/dl/go" in df
    assert "cli.github.com/packages" in df


def test_dockerfile_firewall_wiring_is_not_here():
    # The firewall/egress RULES live in init.sh, not the Dockerfile (a doc
    # comment may still mention the trigger var).
    df = _read_text("Dockerfile")
    assert "iptables -A OUTPUT" not in df
    assert "-P OUTPUT DROP" not in df
    assert "apt.conf.d/01aom-proxy" not in df


def test_init_execs_args_and_is_passthrough_without_firewall():
    init = _read_text("init.sh")
    assert init.startswith("#!/bin/sh")
    # must exec "$@"
    assert 'exec "$@"' in init
    # passthrough when the firewall is not requested
    assert 'if [ "${AOM_INIT_FIREWALL:-}" != "1" ]; then' in init
    # firewall branch applies default-deny egress and drops to agent via gosu
    assert "iptables" in init
    assert "-P OUTPUT DROP" in init
    assert "gosu" in init
    assert "AOM_EGRESS_IP" in init


def test_init_sh_executable_shebang_only_sh():
    # POSIX sh, not bash (the passthrough must work on minimal shells).
    init = _read_text("init.sh")
    assert init.splitlines()[0] == "#!/bin/sh"


def test_skills_have_frontmatter():
    for rel in ("skills/goal/SKILL.md", "skills/browser/SKILL.md"):
        text = _read_text(rel)
        assert text.startswith("---\n")
        assert "\nname:" in text
        assert "\ndescription:" in text
