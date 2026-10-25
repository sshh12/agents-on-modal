"""Tests for agents_on_modal.sandbox.instructions (build_instructions + load_skills)."""

from __future__ import annotations

import pytest

from agents_on_modal.sandbox.instructions import build_instructions, load_skills


def _local() -> str:
    return build_instructions(
        "local", workdir="/home/me/proj", home="/home/me",
        shared_dir=None, network="host", proxy=False,
    )


def _docker() -> str:
    return build_instructions(
        "docker", workdir="/work", home="/home/agent",
        shared_dir=None, network="internet", proxy=False,
    )


def _modal() -> str:
    return build_instructions(
        "modal", workdir="/work", home="/home/agent",
        shared_dir=None, network="full", proxy=False,
    )


def test_header_names_each_place():
    assert "host machine (no container isolation)" in _local()
    assert "isolated Docker container" in _docker()
    assert "isolated Modal sandbox" in _modal()


def test_common_policy_lines_present():
    for text in (_local(), _docker(), _modal()):
        assert text.startswith("# Sandbox environment")
        assert "All approvals are disabled" in text
        assert "security boundary" in text
        assert "No cloud credentials are present." in text
        assert text.endswith("\n")


def test_network_rules_per_kind():
    local, docker, modal = _local(), _docker(), _modal()
    # docker: internet only, LAN/host blocked
    assert "unreachable by design" in docker
    assert "private/LAN addresses are refused" in docker
    # modal: full internet, no LAN-blocked language
    assert "Full outbound internet is available." in modal
    assert "unreachable by design" not in modal
    # local: shares host network / LAN reachable
    assert "share the host machine's network" in local
    assert "LAN" in local


def test_batteries_tools_and_browser_only_in_containers():
    local, docker, modal = _local(), _docker(), _modal()
    for text in (docker, modal):
        assert "Preinstalled tools" in text
        assert "## Browser (headless Chromium 156" in text
        # the browser recipe: snapshot-first Playwright MCP + agent-browser
        assert "browser_navigate" in text
        assert "browser_snapshot" in text
        assert "agent-browser open" in text
        assert "channel=\"chrome\"" in text
    # local is the host, not the batteries image
    assert "not the batteries image" in local
    assert "## Browser (headless Chromium 156" not in local


def test_workdir_and_home_rendered():
    docker = _docker()
    assert "`/work` (your cwd)" in docker
    assert "`$HOME` = `/home/agent` is ephemeral" in docker
    assert "/work/.goal/GOAL.md" in docker
    local = _local()
    assert "`$HOME` = `/home/me` is persisted" in local


def test_shared_dir_changes_workspace_line():
    text = build_instructions(
        "modal", workdir="/work", home="/home/agent",
        shared_dir="/shared", network="full", proxy=False,
    )
    assert "`/shared`" in text
    assert "sibling agents see it live" in text
    assert "Your cwd is `/work`" in text


def test_proxy_flag_adds_upstream_line():
    without = _docker()
    assert "upstream proxy" not in without
    with_proxy = build_instructions(
        "docker", workdir="/work", home="/home/agent",
        shared_dir=None, network="internet", proxy=True,
    )
    assert "upstream proxy" in with_proxy
    assert "Model inference does NOT use it." in with_proxy


def test_unknown_kind_and_network_raise():
    with pytest.raises(ValueError):
        build_instructions("nope", workdir="/w", home="/h",
                           shared_dir=None, network="full", proxy=False)
    with pytest.raises(ValueError):
        build_instructions("modal", workdir="/w", home="/h",
                           shared_dir=None, network="weird", proxy=False)


def test_load_skills_finds_goal_and_browser():
    skills = load_skills()
    assert {"goal", "browser"} <= set(skills)
    for name, text in skills.items():
        # every skill carries YAML frontmatter with name + description
        assert text.startswith("---")
        assert "name:" in text
        assert "description:" in text
    assert "GOAL.md" in skills["goal"]
    assert "never narrow the objective" in skills["goal"].lower()
    assert "browser_snapshot" in skills["browser"]
    assert "agent-browser" in skills["browser"]
