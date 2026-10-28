"""Root instructions text and portable skills for every sandbox kind.

``build_instructions`` renders the one markdown block a sandbox hands to every
harness (written where the harness reads user-level instructions). ``load_skills``
reads the ``SKILL.md`` files shipped under ``image/skills`` so they can be
installed into the harness's skill directory.

Both are pure/side-effect-free apart from reading packaged resources, so they are
cheap to call during harness setup and easy to unit-test offline.
"""

from __future__ import annotations

from importlib.resources import files

# Network policy descriptors accepted by ``build_instructions`` and the lines
# they render. Keyed by the ``network`` argument a sandbox passes.
#   host     -> local sandbox: shares the host's network, LAN reachable.
#   internet -> docker sandbox: outbound internet only, host/LAN blocked.
#   full     -> modal sandbox: full outbound internet.
_NETWORK = {
    "host": (
        "You share the host machine's network: the host, its LAN, and the "
        "internet are all reachable. This is NOT an isolated network, so be "
        "careful with scans and destructive calls."
    ),
    "internet": (
        "Full outbound internet is available. The host machine and its LAN are "
        "unreachable by design (private/LAN addresses are refused); do not try "
        "to reach them."
    ),
    "full": "Full outbound internet is available.",
}

# Per-kind phrasing for the opening line and the $HOME persistence note.
_PLACE = {
    "local": "the host machine (no container isolation)",
    "docker": "an isolated Docker container",
    "modal": "an isolated Modal sandbox",
}


def _tools_section(kind: str) -> str:
    """Tool inventory for the batteries image (docker/modal)."""

    if kind == "modal":
        # Modal sandboxes set no_new_privileges, which neuters setuid sudo -- so sudo/apt do
        # NOT work here. Python's install targets are agent-owned, so plain sudo-free installs land.
        header = (
            "## Preinstalled tools (prefer them). sudo is NOT available here (the sandbox sets "
            "no-new-privileges), so do not use `sudo`/`apt-get`. Add packages without root: "
            "`pip install <pkg>` (installs to a writable location -- no sudo, no venv needed), "
            "`npm i -g <pkg>`, `cargo install <pkg>`, `go install <pkg>`.\n"
        )
    else:
        header = (
            "## Preinstalled tools (prefer them; install more with "
            "`sudo apt-get install -y`, `uv pip install`, `npm i -g`, "
            "`cargo install`, `go install`)\n"
        )
    return (
        header
        + "- Languages: python3 3.13 + uv/pip/pipx (PEP 668 disabled), node 24 + "
        "npm/pnpm/tsx, go, rust (cargo), gcc/clang/cmake/ninja.\n"
        "- CLI: git, gh, rg, fd, fzf, jq, yq, bat, tree, sqlite3, psql, mysql, "
        "redis-cli, curl, wget, tmux, vim, shellcheck, ffmpeg, imagemagick, "
        "pdftotext.\n"
        "- Network diagnostics: dig, host, ping, traceroute, mtr, nc, socat, "
        "nmap, tcpdump, whois, ip."
    )


def _browser_section() -> str:
    """The zero-config browser recipe (docker/modal)."""

    return (
        "## Browser (headless Chromium 156 at /usr/bin/chromium; no display)\n"
        "- To READ a page, this is the reliable path -- use it, do not hand-write a browser "
        "script: `agent-browser open <url>` then `agent-browser read` (returns the full rendered "
        "page text, including headings). State persists across invocations, so open once then "
        "read/act in later calls. The ONLY subcommands are `open <url>`, `snapshot -i`, `read`, "
        "`click @eN`, `fill @eN \"text\"` -- do NOT guess others (`browse`, `eval`, `js`, `get` "
        "do not exist).\n"
        "- Or the `browser_*` MCP tools (Playwright): `browser_navigate`, then `browser_snapshot` "
        "to read as text; act with `browser_click`/`browser_type`/`browser_fill_form` using the "
        "`[ref=eN]` value as `target`. Prefer snapshots over screenshots.\n"
        "- Avoid writing raw Playwright/Puppeteer scripts for simple reads -- they are error-prone "
        "here (navigation/locator timing) and unnecessary. The libraries exist for genuine custom "
        "automation only; if you must, launch with `channel=\"chrome\"` (or "
        "`executable_path=\"/usr/bin/chromium\"`) and never run `playwright install`.\n"
        "- If a click reports `element is not visible`, the target is hidden or the locator matched "
        "several elements -- re-`snapshot` and click the right `[ref=eN]`, or use `fill` (it works on "
        "hidden inputs); do not blindly retry the same click or patch a script across turns.\n"
        "- See the `browser` skill for details. To test a local server, bind it "
        "to 127.0.0.1 and browse `http://127.0.0.1:<port>`."
    )


def build_instructions(
    kind: str,
    *,
    workdir: str,
    home: str,
    shared_dir: str | None,
    network: str,
    proxy: bool,
) -> str:
    """Render the root instructions markdown for a sandbox.

    ``kind`` is ``local`` | ``docker`` | ``modal``; ``network`` is one of
    :data:`_NETWORK` (``host`` | ``internet`` | ``full``). ``workdir`` is the
    agent cwd and ``home`` its ``$HOME``. When ``shared_dir`` is set the
    workspace is shared live (swarms); when ``proxy`` is true all agent TCP is
    routed through an upstream proxy. Facts the agent cannot cheaply discover,
    stated as imperative one-liners, most important first.
    """

    if kind not in _PLACE:
        raise ValueError(f"unknown sandbox kind {kind!r}")
    if network not in _NETWORK:
        raise ValueError(f"unknown network policy {network!r}")

    batteries = kind in ("docker", "modal")
    # $HOME persists only on the host; containers are torn down per session.
    home_persists = "persisted" if kind == "local" else "ephemeral"
    if kind == "local":
        user = "your own account"
    elif kind == "modal":
        user = "user `agent` (uid 10001); sudo is NOT available (the sandbox sets no-new-privileges)"
    else:
        user = "user `agent` with passwordless `sudo`"

    lines: list[str] = []
    lines.append("# Sandbox environment")
    lines.append("")
    lines.append(
        f"You are running inside {_PLACE[kind]} as {user}. All approvals are "
        "disabled; the sandbox is the security boundary. Work autonomously; do "
        "not wait for confirmation."
    )
    lines.append("")

    # --- Files
    lines.append("## Files")
    if shared_dir:
        lines.append(
            f"- Shared workspace (the user and sibling agents see it live): "
            f"`{shared_dir}`. Coordinate through it. Your cwd is `{workdir}`."
        )
    else:
        lines.append(
            f"- Workspace (the user sees it live): `{workdir}` (your cwd). Put "
            "deliverables here; keep scratch files in `/tmp`."
        )
    lines.append(
        f"- `$HOME` = `{home}` is {home_persists}. Nothing outside the "
        f"workspace is guaranteed to survive the session."
    )
    lines.append("")

    # --- Tools + browser (batteries image only)
    if batteries:
        lines.append(_tools_section(kind))
        lines.append("")
        lines.append(_browser_section())
        lines.append("")
    else:
        lines.append("## Tools")
        lines.append(
            "- Use whatever is already installed on the host; this is not the "
            "batteries image. A headless browser and MCP tools may be absent."
        )
        lines.append("")

    # --- Network
    lines.append("## Network")
    lines.append(f"- {_NETWORK[network]}")
    if proxy:
        lines.append(
            "- All outbound TCP is routed through an upstream proxy (sticky "
            "per-session). Model inference does NOT use it."
        )
    lines.append(
        "- Model endpoint: internal; do not call it directly. No cloud "
        "credentials are present."
    )
    lines.append("")

    # --- Finishing
    lines.append("## Finishing")
    lines.append(
        "- Verify before claiming success: run the tests/build/command and show "
        "the output."
    )
    lines.append(
        f"- If a goal file exists at `{workdir}/.goal/GOAL.md`, keep working "
        "until its checks pass (use the `goal` skill)."
    )

    return "\n".join(lines) + "\n"


def load_skills() -> dict[str, str]:
    """Portable skills shipped with the image: ``{skill name: SKILL.md text}``.

    Reads ``image/skills/<name>/SKILL.md`` via ``importlib.resources`` so it
    works from an installed wheel as well as a source checkout.
    """

    # Chain single-arg joinpath so this works on Python 3.10/3.11 too
    # (multi-arg Traversable.joinpath is 3.12+).
    skills_root = files("agents_on_modal.sandbox").joinpath("image").joinpath("skills")
    out: dict[str, str] = {}
    for entry in skills_root.iterdir():
        if not entry.is_dir():
            continue
        skill_md = entry.joinpath("SKILL.md")
        if skill_md.is_file():
            out[entry.name] = skill_md.read_text(encoding="utf-8")
    return out


__all__ = ["build_instructions", "load_skills"]
