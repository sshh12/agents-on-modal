# Adding a model to the catalog

A model is one `agents_on_modal/catalog.json` entry (`Model` in `models/__init__.py`). Adding a model
is a catalog edit; everything below is about getting that entry **right** and proving the whole stack
(serving × every harness feature) works on it before you trust it.

> **Keep this file current.** This is a living logbook, not a frozen spec. Whenever you discover a new
> per-model behavior that breaks something — a quant/arch that needs a specific engine image or
> unmerged commit, a GPU/`mem-fraction`/`target_concurrency` combination that OOMs or won't boot, a
> context/KV limit that the harness overflows, a parser/chat-template quirk, a sandbox or harness
> friction that only shows up on certain models — **add it here** (and, when it's sizing, into that
> model's catalog `description`). The next model is only fast to bring up because the last surprise got
> written down.

Two phases, cheap-before-expensive:

- **Author + deploy** — editing the catalog and `aom models deploy` are **CPU-only** (image build +
  weight download on a `prepare` function). No GPU is billed. Do all the static validation here.
- **Bring up + pressure-test** — the GPU cold-starts on the **first real request**; that is where
  billing begins. Keep `max_containers=1` throughout, watch the run so a broken boot doesn't idle a
  GPU, and scale to zero the moment you're done.

Rule of thumb for this whole doc: **the GPU endpoint is the cost, not the agents.** One warm,
capped-at-1 endpoint can serve every pressure-test concurrently (they batch/queue through the single
container — which is itself a real throttle test). Never fire an agent until a cheap smoke completion
proves the endpoint generates.

---

## 0. Read the model card first

- [ ] **Architecture** — is it a *new* arch? New arches don't load on stock engine releases. The V4.1
      FP8 entry needs the `lmsysorg/sglang:dev-dsv41` preview image; the V4.1 GGUF needs an *unmerged*
      llama.cpp commit. Stock SGLang / llama.cpp / Ollama will refuse them.
- [ ] **Quant & size on disk** — FP8/NVFP4/GGUF bpw and the **GiB on disk** (≈ the weight VRAM you must
      fit). This number drives GPU sizing (§2).
- [ ] **Recommended GPUs/recipe** — but verify it fits *this* artifact. Model-card recipes are often for
      a smaller base build (the FP8 card's `0.80 / 4×H200` recipe was for a smaller artifact than the
      475 GiB uncensored build, and did not fit).
- [ ] **Context length** — native window, and any YaRN/rope scaling. We cap every entry at **256k**
      (`262144`) even when the checkpoint supports 1M.
- [ ] **Vision?** — safetensors multimodal needs no extra flag (`vision: true`); a GGUF with **no
      mmproj projector is text-only** here regardless of the base model.
- [ ] **Parsers / chat template** — reasoning parser, tool-call parser, and the thinking kwarg
      (`thinking` vs `enable_thinking`).
- [ ] **Gated/license** — if the repo is gated, set `hf_token_env: "HF_TOKEN"` and make sure that token
      is in `.env`.

Pick the engine: **safetensors → `sglang`**, **GGUF → `llamacpp`** (SGLang cannot load GGUF; the
llamacpp runtime builds llama-server from a pinned source commit on its own CUDA base).

---

## 1. Author the catalog entry

Fields (see `Model` / `Scaling` / `Reasoning` in `agents_on_modal/models/__init__.py`):

- [ ] **`name`** — lowercase `[a-z0-9.-]`, ≤40 chars. It is the catalog key, the served model id, and
      the app suffix (`aom-<name>`, dots → dashes).
- [ ] **`repo`** / optional **`revision`** — HuggingFace repo (pin a revision for reproducibility).
- [ ] **`engine`** — `sglang` | `llamacpp`.
- [ ] **`gpu`** — `H200` or `B300:4`. GPU **count must be a power of two** (tensor parallelism); the
      validator rejects otherwise.
- [ ] **`context_window`** / **`max_output_tokens`** — need `context_window >= 8192` and
      `max_output_tokens < context_window`. For **llama.cpp pin `context_window` to the served `-c`**
      (e.g. 32768) so the harness's auto-compaction math never overflows the real KV. (32k on the GGUF
      is the *server's* KV limit, not a harness cap — the SGLang entries genuinely serve the full 256k.)
- [ ] **`image`** — only when the arch needs a non-default engine image (e.g. `lmsysorg/sglang:dev-dsv41`).
      Omit for llamacpp (it builds its own CUDA base).
- [ ] **`engine_args`** — every entry must be `--flag` or `--flag=value` (**use the `=` form for
      llama.cpp**, which rejects space-separated values). **Do not** repeat reserved flags the wrapper
      injects: `--host --port --model-path --served-model-name --tp --context-length --model -m --alias
      -a --n-gpu-layers -ngl --ctx-size -c`. (`tests/test_catalog.py::test_no_reserved_engine_args_in_catalog`.)
- [ ] **`download_subdir`** — **required for llamacpp** (the GGUF quant folder, e.g. `Mix-256G`); fetches
      only that quant, not the whole repo.
- [ ] **`scaling`** — `{"max_containers": 1, ...}` to start (see §2 for `target_concurrency`).
- [ ] **`reasoning`** — `default` effort + the full `efforts` map onto `(none minimal low medium high
      xhigh max)` + `kwarg` (thinking switch). `none` → `false` (thinking off).
- [ ] **`startup_timeout`** — bump to `7200` for big models (download + compile + cuda-graph capture can
      take a long time on the first cold start).
- [ ] **`base_repo`/`base_revision`/`base_volume`/`base_volume_path`** — optional weight-dedup against an
      already-downloaded base model's Volume; **all four go together**, and the base Volume must already
      exist at the version requested (a stale v1-vs-v2 mismatch fails the deploy — we removed exactly
      this from `deepseek-v4-flash-abliterated`). Leave unset unless you have a reason.
- [ ] **Write the `description` as the sizing logbook.** The best entries (see the V4.1 FP8/GGUF ones)
      record *why* this GPU/mem-fraction/parser was chosen and what failed — that is what makes the next
      model fast to size.

**Validate offline (no network, no GPU):**

```bash
python -m pytest tests/test_catalog.py -q
python -c "from agents_on_modal import models; m=models.get('<name>'); print(m.app_name, m.gpu_count, m.context_window)"
```

`Model.from_json` runs `_validate` (name/engine/gpu/region/context/efforts/engine-arg shape), so a
malformed entry fails here, instantly, for free.

---

## 2. Right-sizing the GPU (what we've seen)

The goal is the **smallest stable allocation**: the fewest/cheapest GPUs on which the engine boots and
serves without OOM. GPU memory has to hold three things, not just the weights:

```
per-GPU usable  ≈  weights/GPU  +  KV-cache pool  +  cuda-graph capture & activations
                   └── static ───┘   (the "non-static" ~10–15% headroom) ─────────────┘
   mem-fraction-static = (weights + static KV) / total ;  the rest = 1 − mem_fraction
```

Observed **usable** VRAM per GPU (after driver/runtime overhead): **H200 ≈ 139 GiB**, **B300 ≈ 267.7
GiB**. `weights/GPU = weights_on_disk / tp`.

**The sizing rule that has held:**

- Aim for **weights/GPU ≲ 55–60% of usable**. That leaves `mem-fraction-static ≈ 0.85` with a fat KV
  pool *and* a ~15% non-static pool for cuda-graph capture. This is the comfortable zone.
- **Danger zone: weights/GPU ≳ 80% of usable.** SGLang then hard-fails at boot: *"Loaded weights leave
  no GPU memory for the KV cache under `--mem-fraction-static=0.8` … minimum viable 0.870."* Raising
  `mem-fraction-static` above that minimum to make KV fit then **starves the non-static pool and
  cuda-graph capture OOMs (exit −9)**. Both failures are the same squeeze from opposite ends.

**Worked example — DeepSeek V4.1 FP8, 475.3 GiB weights:**

| Config | weights/GPU | % of usable | Result |
|---|---|---|---|
| H200:4 tp4 | 118.8 GiB | ~85% of 139 | ❌ no KV room at 0.80; pushing mem-fraction → capture OOM |
| B300:2 tp2 | 237.6 GiB | ~89% of 267.7 | ❌ worse |
| **B300:4 tp4** | **118.8 GiB** | **~44% of 267.7** | ✅ 0.85 → ~100+ GiB/GPU KV, ~40 GiB/GPU for capture |

The GLM NVFP4 entry hit the **identical** capture OOM when `mem-fraction-static` was pushed to 0.92
(non-static fell to ~8%). Keep non-static ≳ 10–15%.

**Stable minimums we actually landed on** (start here for a similar size/quant, then verify):

| Model (quant) | weights | Stable GPU | context |
|---|---|---|---|
| deepseek-v4-flash-abliterated | — | **H200:2** | 256k |
| glm-5.3-flash-uncensored (FP8) | — | **B300:2** | 256k |
| glm-5.3-abliterated (NVFP4) | — | **B300:2** | 256k |
| deepseek-v4.1-flash-uncensored (FP8) | 475 GiB | **B300:4** | 256k |
| deepseek-v4.1-flash-abliterated (GGUF) | 235.8 GiB in-mem | **H200:2** | 32k (server KV) |

**Other sizing knobs:**

- [ ] **`target_concurrency`** — requests one container batches before Modal adds another. Default 32
      (small models). **Lower it for big/slow models** so you don't over-batch a saturated GPU: FP8 used
      `4.0`, GGUF `1.0`. Pair with SGLang `--max-running-requests` / llama.cpp `--parallel`.
- [ ] **`max_containers: 1`** — hard ceiling = graceful throttle under load (never a GPU-per-agent
      fan-out). This is the default policy and the cost guardrail; raise it deliberately, live
      (`aom models scale … --max-containers N`), only when a fan-out genuinely needs it.
- [ ] **`min_containers: 0`** — scale to zero when idle (the gateway holds requests through the next
      cold start, so this costs latency, not 503s). `scaledown_window` (default 900s) keeps a container
      warm so back-to-back use isn't punished by repeated cold starts.
- [ ] **`mem-fraction-static`** — start at **0.85** and move in small steps. If boot says KV won't fit,
      you need *more* GPU, not a higher fraction (see the danger zone).
- [ ] Do **not** pin SGLang attention/moe/gemm backends unless you must — manual overrides can fall back
      to slow Triton. Confirm the **resolved backends in the startup log** instead.

---

## 3. Deploy (CPU only — cheap)

```bash
aom models deploy <name>                      # idempotent: image + CPU `prepare` downloads/verifies weights
# or: python scratchpad/deploy_model.py <name>
```

- [ ] `prepare` downloaded all shards and wrote its manifest (no GPU allocated yet).
- [ ] If you changed the **sandbox Dockerfile**, rebuild+publish the image first:
      `python scratchpad/build_image.py` (CPU build; tag is a content hash, so only changed layers
      rebuild).

---

## 4. Bring up + smoke (first GPU dollars)

```bash
python -c "from agents_on_modal import models as M; e=M.endpoint(M.get('<name>')); M.wait_ready(e); print('ready', e.url)"
```

- [ ] **`wait_ready` returns** — `/health` is 200 only when the engine answers a *real* generation, so
      this is a true readiness check through the cold start.
- [ ] **Startup log is clean** — resolved parsers/backends as expected; no mem-fraction warning; no
      cuda-graph capture OOM (exit −9); no repeated container restarts.
- [ ] **`/v1/models` reports `max_model_len == context_window`** — proves the window reached the engine.
- [ ] **One smoke completion generates** before you spawn any agent.
- [ ] **Watch for deploy churn:** two near-simultaneous `modal deploy` calls SIGTERM a cold-starting
      container → `"no upstreams available"` 503. Don't race deploys. **Recover without re-deploying**:
      autoscaler `max→0` then `→1` + `wait_ready` (`python scratchpad/reset_fp8.py <name>`).

---

## 5. Harness-feature pressure-tests

Endpoint warm, capped at 1. Each check has a one-command helper (currently in `scratchpad/`; promote the
ones you want to keep into `tests/e2e/`). Run against the new model name.

- [ ] **Full context window, both wire protocols** — needle-in-haystack at ~`context_window` depth on
      **`/v1/chat/completions`** (codex/chat) *and* **`/v1/messages`** (claude), plus `/v1/models`
      `max_model_len`. Proves the window isn't silently clamped.
      `python scratchpad/context_check.py <name> [target_tokens]`
- [ ] **Both harnesses e2e in a Modal sandbox** — `claude` (Anthropic protocol) and `codex` (Responses
      protocol), each exercising shell + file + **pip dev-install** + **browser** + a verifiable report.
      The current local harness code is synced in, so you can bug-fix and re-run without rebuilding.
      `python scratchpad/harness_matrix.py <name> [claude codex …]`
- [ ] **Auto-compaction is smooth, per harness** — override the window small (e.g. 40k) so the harness
      compacts at a reachable point while the gateway still serves the real 256k; verify the run stays
      correct across the compaction (recovers an early *and* a late fact). Claude emits
      `compact_boundary`; codex bounds via truncation.
      `python scratchpad/autocompact_check.py <name> [window] [harness …]`
- [ ] **Tool features actually execute in-sandbox (Claude)** — cron (`CronList`) / `Monitor` / subagents
      (`Task`) are *advertised*; verify they *run* and whether each `tool_result` came back ok. (cron may
      legitimately fail with no claude.ai creds in the isolated sandbox — that itself is a finding.)
      `python scratchpad/caps_check.py <name>`
- [ ] **Sandbox friction is zero** — run a small swarm, then aggregate friction from the persisted
      traces: no `pip`/`sudo`/`PYTHONPATH` errors (the Modal sandbox has no sudo; deps install with plain
      `pip`, `PYTHONPATH=/opt/aom`, caches→/tmp), no raw-Playwright flailing (agent should use the
      `agent-browser` subcommands). `python scratchpad/friction_review.py <swarm-id>`
- [ ] **Swarm fan-out + dynamic swarm** — N sandboxes sharing `/shared`, stateless reconnect, and the
      dynamic orchestrator + `aom-swarm` control plane. Launch from the dashboard
      (`python scratchpad/run_dashboard.py`) or `python scratchpad/spawn_swarm.py`.
- [ ] **Proxy modes propagate** (if the model will do web/browser work) — `off` / `always` (per-agent
      sticky residential IPs, distinct per agent) / `optin` (direct unless the agent opts in), passed
      through to fan-out subagents; **model inference stays direct in every mode** (`NO_PROXY` carries
      the model host — no model bytes metered over the residential link).
      `python scratchpad/proxy_e2e.py`
- [ ] **Reasoning efforts map** — spot-check that `none` turns thinking off and that `low/high/max`
      reach the engine (reflected in the trace / token behavior).
- [ ] **Vision** (only if `vision: true`) — one image prompt round-trips.

---

## 6. Cost control + teardown

- [ ] Endpoint ran capped at `max_containers=1` the whole time; no runaway fan-out.
- [ ] **Stop everything when done — don't rely on idle scaledown alone:**
      `python scratchpad/stop_all.py <name>`  (or `aom models stop <name>`).
- [ ] **Verify it's actually off:** `aom models status <name>` shows the app stopped / autoscaler cleared
      (not just "min=0"). A force-killed sandbox can leave a stale record — confirm the GPU is gone.
- [ ] Secrets stayed out of git: `.env`, `proxies.csv` are gitignored; nothing residential-proxy was
      written to `/shared`; subagents never used real keys or touched the user's real `~/.claude`/`~/.codex`.

---

## Quick command reference

```bash
aom models list | deploy <name> | scale <name> [--min N] [--max N] | stop <name> | status <name>
aom run   --model <name> --harness claude|codex  "<prompt>"     # one headless run
aom shell --model <name> --harness claude|codex                 # interactive TUI
aom swarm --model <name> -n <N> …                               # fan out + dashboard

# session validation helpers (scratchpad/ — promote keepers into tests/e2e/):
deploy_model.py  context_check.py  harness_matrix.py  autocompact_check.py
caps_check.py    friction_review.py  proxy_e2e.py  reset_fp8.py  stop_all.py  build_image.py
```
