# agents-on-modal

Run agent coding CLIs — **Claude Code**, **Codex**, and more to come — inside a sandbox of your
choice, against **open-weight models you serve on [Modal](https://modal.com)** GPUs. Three
independent axes you mix freely:

- **model** — any entry in the **[model catalog](agents_on_modal/catalog.json)** (Qwen3, GPT-OSS,
  DeepSeek-V4, GLM-5.3, …), served by SGLang or llama.cpp on a Modal GPU that scales to zero when idle.
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
endpoint = aom.models.ensure_deployed(model, max_containers=1)  # deploys if needed; weights on CPU first

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

Every servable model is one entry in **[`agents_on_modal/catalog.json`](agents_on_modal/catalog.json)**
— browse it for the names, GPUs, context windows and engine settings; each entry's `description` is
its sizing logbook (why that GPU, what failed). `aom models list` prints the same catalog.

```sh
aom models list
aom models deploy qwen3.8-27b --max 1             # --max: GPU container cap (a redeploy restarts it)
aom models scale qwen3.8-27b --min 2 --max 32     # live, no redeploy — for a fan-out
```

Adding a model is one entry in the [catalog](agents_on_modal/catalog.json) (repo, engine, GPU,
parsers, context, and the per-model reasoning-effort map), not new code. Follow
**[`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md)**: the sizing rules, the bring-up and
pressure-test checklist, and the recurring gotchas every new model has hit so far.

The container cap isn't in the [catalog](agents_on_modal/catalog.json). The apps using an endpoint
choose it: every command and example asks for 1 by default (`--max-containers N` to ask for more).
The first app to need the model
deploys it at that cap. Later ones attach to it as-is and only ever raise the cap, so they never
restart it under a session or swarm already using it. Only `aom models scale --max` lowers it. The
cap is a ceiling, not a reservation: Modal adds a container only when one is saturated past the
model's `target_concurrency`, and idle ones still scale to zero.

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

Each of these deploys the model first if it isn't deployed yet (a live endpoint is attached to as-is,
never restarted), then waits out any cold start. A deploy needs `HF_TOKEN` in the environment.

The example scripts in [`examples/`](examples/) show the same flows as short programs. Like the
dashboard below, they set themselves up (`examples/bootstrap.py`: the repo `.env`, the `.venv`, the
Modal login and proxy token), so plain `python examples/local_session.py --model <name>` works;
anything they don't recognise, such as an initial prompt, goes to the harness.

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
python examples/swarm/server.py            # open the dashboard (add --swarm <id> to jump to one)
```

The dashboard sets itself up: it loads the repo `.env` (`HF_TOKEN`, `AOM_SWARM_PRESET_KEY`, …),
re-launches on the repo `.venv` (creating it and `pip install -e .`-ing on first run), and runs the
Modal login / `aom setup` if they haven't been done, so any `python` on your PATH works. A spawn
asks for a cap of 1 like every other app. To give a second, parallel swarm its own GPU once the first
one saturates the endpoint, raise the model's max instances live in the dashboard's model endpoints
panel; that's also where you lower it again. To roll out a [catalog](agents_on_modal/catalog.json)
change to a deployed model, run `aom models deploy <name> --max N`, which does replace its running
containers.

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
