# Sandbox image

The batteries-included image both container backends run agents in. One
`Dockerfile`, built the same way by each:

| backend | how it builds this image |
|---|---|
| `docker` | `docker build -t aom-sandbox:<tag>` of this directory as the build context |
| `modal`  | `modal.Image.from_dockerfile("Dockerfile", ...)`, published as a named Modal image |
| `local`  | no image; the agent runs on the host with whatever tools it has |

The build context is **this directory** (`agents_on_modal/sandbox/image/`), so
`init.sh` sits next to the `Dockerfile` for the `COPY init.sh /opt/aom-init.sh`
step. The `skills/` tree is read at runtime by
`agents_on_modal.sandbox.instructions.load_skills()` (via `importlib.resources`),
not baked into the image — a harness installs them into its own skill dir.

## What's inside

- Base `node:24-trixie-slim` (Node 24 — required by `agent-browser`; Python 3.13;
  Chromium 156 from Debian).
- A broad dev/cyber toolchain (build, languages, VCS, search/text, DB clients,
  network diagnostics, media, archive); see the `Dockerfile` package list.
- One system Chromium made to look like Google Chrome stable (the
  `/opt/google/chrome/chrome` symlink + `/etc/chromium.d/zz-container`
  `--no-sandbox` drop-in) so every browser tool works with zero flags.
- Browser agents: `@playwright/mcp` (the `browser_*` MCP server), `agent-browser`
  (shell CLI), plus Python/Node Playwright and `puppeteer-core`.
- The agent CLIs, pinned: Claude Code and Codex.
- A non-root `agent` user (uid 10001) with passwordless `sudo`; `/work` is the
  default workdir and `/home/agent/.aom` its state dir.

## Build args (pins)

Bump these deliberately, then rebuild and republish. Defaults are the pinned
versions; tests assert the Dockerfile still carries them.

| arg | default | what |
|---|---|---|
| `CLAUDE_CODE_VERSION` | `2.1.310` | `@anthropic-ai/claude-code` |
| `CODEX_VERSION` | `0.155.1` | `@openai/codex` |
| `PW_MCP_VERSION` | `0.0.83` | `@playwright/mcp` |
| `AGENT_BROWSER_VERSION` | `0.39.2` | `agent-browser` |
| `PY_PLAYWRIGHT_VERSION` | `1.64.1` | pip `playwright` |
| `UV_VERSION` | `0.13.5` | pip `uv` |
| `GO_VERSION` | `1.24.8` | upstream Go tarball |
| `TARGETARCH` | (from buildkit) | `amd64` / `arm64` for the Go tarball |

## Runtime entrypoint

`ENTRYPOINT` is `tini -- /opt/aom-init.sh`, and `init.sh` **execs its args**, so
`docker run <image> <cmd...>` / `Sandbox.create([...])` run `<cmd...>` as given.
`init.sh` only does extra work when `AOM_INIT_FIREWALL=1`: it applies a
default-deny iptables egress policy (allowing only loopback and the broker at
`AOM_EGRESS_IP:AOM_EGRESS_PROXY_PORT,AOM_EGRESS_MODEL_PORT`) and then drops to
the `agent` user with `gosu`. With the var unset it is a pure passthrough — the
case on Modal, where iptables is unavailable and the backend uses `gosu`
per-command. The egress/proxy server itself lives in `agents_on_modal.sandbox.net`,
not in this image.

The `agents_on_modal` package is **not** installed in the image. The builder
mounts/copies it at `/opt/aom` (the `Dockerfile` sets `PYTHONPATH=/opt/aom`) so
`python3 -m agents_on_modal.runner` resolves without rebuilding the image when
the package changes.

## Rebuilding

1. Edit the `Dockerfile` (or a build arg) here.
2. `docker build .` from this directory to smoke-test locally; drive a browser
   agent to confirm Chromium still launches headless (`--no-sandbox`).
3. The `modal` backend republishes its named image from the same `Dockerfile`.
4. Run `python -m pytest tests/test_image_*.py -q`; the pin assertions guard
   against accidental drift.
