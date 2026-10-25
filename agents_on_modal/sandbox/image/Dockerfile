# Batteries-included sandbox image for agents-on-modal.
#
# A broad dev/cyber box with headless Chromium, the Playwright MCP browser
# server, agent-browser, gh, and both agent CLIs (Claude Code + Codex) pinned.
# Agents run here with every approval disabled: the sandbox (container / Modal
# Sandbox / egress firewall) is the security boundary, never the harness.
#
# Shared by both backends that build from an image:
#   * docker  -> `docker build` of this file
#   * modal   -> `Image.from_dockerfile(this file)`
# The firewall/egress wiring is NOT baked in here; it is applied at runtime by
# init.sh (only when AOM_INIT_FIREWALL=1) so the same image works on Modal,
# where iptables is unavailable, as a plain full-internet sandbox.
#
# The `agents_on_modal` package itself is NOT installed here. The builder mounts
# or copies it and sets PYTHONPATH=/opt/aom so `python3 -m agents_on_modal.*`
# resolves; this image only provides the OS, tools, browser, and agent CLIs.

FROM node:24-trixie-slim

# Pins. Bump these deliberately; the README explains the rebuild flow. The
# version strings are also asserted by tests, so keep them on their own lines.
ARG CLAUDE_CODE_VERSION=2.1.310
ARG CODEX_VERSION=0.155.1
ARG PW_MCP_VERSION=0.0.83
ARG AGENT_BROWSER_VERSION=0.39.2
ARG PY_PLAYWRIGHT_VERSION=1.64.1
ARG UV_VERSION=0.13.5
ARG GO_VERSION=1.24.8
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive

# --- OS packages: build toolchains, language stacks, VCS, search/text, data
# clients, network diagnostics, process/debug, media, archive, init/users.
# Debian 13 "trixie" package names (verified against trixie-slim).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg lsb-release apt-transport-https \
        build-essential pkg-config make cmake ninja-build autoconf automake \
        clang lld llvm gdb binutils file patch \
        python3 python3-pip python3-venv python3-dev pipx python-is-python3 \
        git git-lfs openssh-client \
        ripgrep fd-find fzf jq yq bat tree less vim nano tmux shellcheck \
        sqlite3 postgresql-client default-mysql-client redis-tools \
        iproute2 iptables iputils-ping traceroute mtr-tiny bind9-dnsutils bind9-host \
        whois netcat-openbsd socat nmap tcpdump net-tools \
        procps psmisc lsof strace ltrace htop \
        imagemagick ffmpeg poppler-utils \
        zip unzip xz-utils zstd rsync \
        tini gosu sudo locales tzdata; \
    rm -rf /var/lib/apt/lists/*

# Locale so tools and TUIs render UTF-8 correctly.
RUN sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Convenience symlinks: Debian ships `fd` as `fdfind` and `bat` as `batcat`.
RUN set -eux; \
    ln -sf "$(command -v fdfind)" /usr/local/bin/fd; \
    ln -sf "$(command -v batcat)" /usr/local/bin/bat

# --- Chromium, one browser for every tool.
# Install Debian `chromium` (amd64 + arm64), then make it look like Google
# Chrome stable so Playwright/Puppeteer/agent-browser/CDP all work with zero
# browser flags: a `chrome` symlink plus a chromium.d drop-in that forces
# --no-sandbox (container seccomp blocks the user-namespace sandbox).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        chromium \
        fonts-liberation fonts-noto-color-emoji fonts-noto-cjk fonts-dejavu-core \
        xvfb xauth; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /opt/google/chrome; \
    ln -sf /usr/bin/chromium /opt/google/chrome/chrome; \
    echo 'export CHROMIUM_FLAGS="$CHROMIUM_FLAGS --no-sandbox"' > /etc/chromium.d/zz-container

# --- gh (GitHub CLI) from GitHub's own apt repo; Debian's package lags badly.
RUN set -eux; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg; \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends gh; \
    rm -rf /var/lib/apt/lists/*

# --- Rust via rustup into a shared prefix so the agent can use and extend it.
ENV RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/cargo
RUN set -eux; \
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal; \
    chmod -R a+rwX /opt/rust /opt/cargo; \
    /opt/cargo/bin/rustc --version

# --- Go from the upstream tarball (Debian's golang-go is too old).
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
        arm64) go_arch=arm64 ;; \
        *) go_arch=amd64 ;; \
    esac; \
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${go_arch}.tar.gz" -o /tmp/go.tgz; \
    tar -C /usr/local -xzf /tmp/go.tgz; \
    rm -f /tmp/go.tgz; \
    /usr/local/go/bin/go version

# --- Node global tooling and the agent CLIs, installed as root into /usr/local
# so the unprivileged agent can run but not clobber them. Pinned versions never
# fetched at runtime (no `npx @pkg@latest` from inside the sandbox).
RUN set -eux; \
    npm install -g --no-fund --no-audit \
        @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
        @openai/codex@${CODEX_VERSION} \
        @playwright/mcp@${PW_MCP_VERSION} \
        agent-browser@${AGENT_BROWSER_VERSION} \
        pnpm typescript tsx; \
    npm cache clean --force; \
    claude --version || true; \
    codex --version || true

# --- Python browser lib + uv, system-wide (PEP 668 override: this is a devbox).
RUN set -eux; \
    pip3 install --no-cache-dir --break-system-packages \
        playwright==${PY_PLAYWRIGHT_VERSION} "uv==${UV_VERSION}"

# This is a throwaway devbox: let the agent `pip install` without a venv or the
# per-command --break-system-packages dance.
RUN rm -f /usr/lib/python3*/EXTERNALLY-MANAGED || true

# --- Environment.
# Browser: point every runtime at the one system Chromium and never download.
# Updaters/telemetry off. Open-weight models need all MCP tools sent upfront
# (ENABLE_TOOL_SEARCH=false) since they do not understand Claude tool search.
ENV CHROME_BIN=/usr/bin/chromium \
    CHROME_PATH=/usr/bin/chromium \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    PLAYWRIGHT_MCP_BROWSER=chromium \
    PLAYWRIGHT_MCP_EXECUTABLE_PATH=/usr/bin/chromium \
    PLAYWRIGHT_MCP_HEADLESS=1 \
    PLAYWRIGHT_MCP_ISOLATED=1 \
    PLAYWRIGHT_MCP_OUTPUT_DIR=/tmp/playwright-mcp \
    PUPPETEER_SKIP_DOWNLOAD=1 \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium \
    AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium \
    DISABLE_AUTOUPDATER=1 \
    DISABLE_UPDATES=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    ENABLE_TOOL_SEARCH=false \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1

# The aom package is mounted/copied by the builder at /opt/aom (not here); make
# `python3 -m agents_on_modal.*` resolve either way.
ENV PYTHONPATH=/opt/aom

# `aom-swarm` shim: the dynamic-swarm orchestrator drives subagents with this. It only touches the
# shared control volume (no Modal creds), so it is safe in every sandbox.
RUN printf '%s\n' '#!/bin/sh' 'exec python3 -m agents_on_modal.swarmctl "$@"' \
        > /usr/local/bin/aom-swarm; \
    chmod 0755 /usr/local/bin/aom-swarm

# --- Non-root agent user (uid 10001, passwordless sudo).
# uid/gid 10001 avoids colliding with the base image's `node` user. The agent
# may install arbitrary system packages; the container is the boundary, so
# passwordless root inside it is intended.
RUN set -eux; \
    groupadd --gid 10001 agent; \
    useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash agent; \
    mkdir -p /work /home/agent/.aom /home/agent/.npm-global /home/agent/go; \
    chown -R agent:agent /work /home/agent; \
    printf '%s\n' \
        'agent ALL=(ALL) NOPASSWD:ALL' \
        'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy all_proxy"' \
        'Defaults env_keep += "HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY ALL_PROXY"' \
        > /etc/sudoers.d/agent; \
    chmod 0440 /etc/sudoers.d/agent

ENV HOME=/home/agent \
    AOM_HOME=/home/agent/.aom \
    GOPATH=/home/agent/go \
    GOFLAGS=-buildvcs=false \
    NPM_CONFIG_PREFIX=/home/agent/.npm-global \
    PATH=/home/agent/.local/bin:/home/agent/.npm-global/bin:/home/agent/go/bin:/opt/cargo/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Codex runs commands through `bash -lc`; a login shell resets PATH from
# /etc/profile and would drop the additions above. Re-add them for login shells.
RUN printf '%s\n' \
    'export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/go/bin:/opt/cargo/bin:/usr/local/go/bin:$PATH"' \
    > /etc/profile.d/10-aom-path.sh; \
    chmod 0644 /etc/profile.d/10-aom-path.sh

# The firewall needs root (CAP_NET_ADMIN), so the container starts as root and
# init.sh drops to the agent user with gosu after applying rules. On Modal the
# backend runs as root and uses gosu per-command instead; init.sh is a no-op
# passthrough there (AOM_INIT_FIREWALL unset).
COPY init.sh /opt/aom-init.sh
RUN chmod 0755 /opt/aom-init.sh

WORKDIR /work

# tini reaps zombies (headless Chromium spawns many); init.sh execs the real
# command so `Sandbox.create([...])` / `docker run image <cmd>` args run as-is.
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/aom-init.sh"]
CMD ["sleep", "infinity"]
