# agents-on-modal

Run agent coding CLIs — **Claude Code**, **Codex**, and more to come — inside a sandbox of your
choice, against **open-weight models you serve on [Modal](https://modal.com)** GPUs. Three
independent axes you mix freely:

- **model** — any entry in the catalog (Qwen3, GPT-OSS, DeepSeek-V4, GLM-5.3, …), served by SGLang or
  llama.cpp on a Modal GPU that scales to zero when idle.
- **harness** — the agent CLI: `claude` (Claude Code) or `codex` (Codex CLI).
- **sandbox** — where the harness runs: `local` (directly on your machine, no container), `docker`
  (a throwaway local container with full internet but no access to your LAN or host), or `modal`
  (a Modal Sandbox, batteries-included, massively parallel).

Every harness reaches every model through one in-container **gateway** that translates the harness's
native protocol (Anthropic Messages for Claude Code, OpenAI Responses for Codex) to the engine's
chat-completions API — so any harness works with any model, with reasoning effort and context limits
mapped per model.

```python
import agents_on_modal as aom

model = aom.models.get("qwen3.8-27b")
endpoint = aom.models.deploy(model)                 # idempotent; weights downloaded on CPU first

box = aom.sandbox.create("docker")                  # batteries-included container
run = box.start("claude", endpoint, "add a /health route and test it", goal=True, timeout=1800)
for event in run.events(follow=True):
    print(event.kind, event.data)
```

## Install

```sh
pip install git+https://github.com/sshh12/agents-on-modal
aom setup            # logs into Modal and stores a workspace proxy token
```

Requires Python 3.10+ and a Modal account. For the `docker` sandbox you also need Docker Desktop; the
`local` sandbox additionally needs the harness CLI (`claude` / `codex`) installed on your PATH.

## Choose and deploy a model

```sh
aom models list
aom models deploy qwen3.8-27b
aom models scale qwen3.8-27b --min 2 --max 32     # live, no redeploy — for a fan-out
```

Adding a model is one entry in `agents_on_modal/catalog.json` (repo, engine, GPU, parsers, context,
and the per-model reasoning-effort map), not new code.

## Run an agent

```sh
# drop into an interactive session on THIS repo, no container, pick harness + model by flag
aom shell --harness claude --model qwen3.8-27b

# a batteries-included local Docker container (full internet, no LAN), interactive
aom shell --sandbox docker --harness codex --model qwen3.8-27b

# a headless goal run in a Modal sandbox; streams the live trace
aom run --sandbox modal --harness claude --model qwen3.8-27b --goal --timeout 3600 \
  "make the test suite pass"
```

The example scripts in [`examples/`](examples/) show the same three flows as ~50-line programs.

## Route sandbox traffic through a residential proxy

For the `docker` and `modal` sandboxes you can send the agent's code/browser traffic through a
SOCKS5 or HTTP residential proxy. **Model inference always stays direct**, so you never pay to tunnel
gigabytes of token traffic.

```sh
aom run --sandbox docker --proxy "socks5h://user-{session}:pass@gate.example.com:7000" \
  --harness claude --model qwen3.8-27b "scrape and summarize example.com"
```

`{session}` is replaced per sandbox, so each agent gets its own sticky IP. The credential lives in the
egress layer, never in the agent container.

## Swarms and the live dashboard

Spawn N Modal sandboxes running the same harness/model/prompt, sharing a `/shared` directory they
collaborate in, and watch every agent's trace stream live — the dashboard is a single static HTML
page that re-discovers in-flight agents after a restart, so you can close the tab and come back.

```sh
aom swarm --harness claude --model qwen3.8-27b --count 50 --goal --timeout 7200 \
  "design a REST API for a library; write the spec and a reference impl in /shared"
python examples/swarm/server.py            # open the dashboard
```

An impossible `/goal` keeps running (auto-compacting) until the timeout, then you read whatever the
agents left in `/shared`.

## Lifecycle and cost

Models deploy as persistent Modal apps with a persistent weights volume and scale to zero after an
idle window; you pay for GPU only while a request is in flight (plus the one-time CPU weight
download). Sandboxes have an idle timeout so a wedged run can't bill forever. `aom models stop
<name>` tears a model's app down.

## Development

```sh
pip install -e ".[dev]"
python -m pytest            # offline unit tests; no Modal, Docker, or CLIs required
AOM_E2E=1 python -m pytest tests/e2e    # opt-in; spends Modal credits
```

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture.
