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
  billing begins. Keep the endpoint at 1 container throughout (`aom models deploy <name> --max 1`),
  watch the run so a broken boot doesn't idle a GPU, and scale to zero the moment you're done.

Rule of thumb for this whole doc: **the GPU endpoint is the cost, not the agents.** One warm,
capped-at-1 endpoint can serve every pressure-test concurrently (they batch/queue through the single
container — which is itself a real throttle test). Never fire an agent until a cheap smoke completion
proves the endpoint generates.

---

## Recurring themes (read these first — every bring-up so far hit at least one)

The per-model logbook at the bottom is the evidence; these are the patterns it keeps repeating.

1. **The architecture decides the image before anything else.** A new arch (`deepseek_v41`,
   `glm_moe_dsa`) loads on no stock engine release — it needs a preview image or an unmerged commit,
   and a quant-specific build can need a different one again. Check `config.json`'s
   `architectures`/`model_type` against the engine's release notes before sizing anything.
2. **Fit is weights + KV + cuda-graph capture, never weights alone.** Model-card recipes are often for
   a different (smaller) artifact. Size from GiB on disk with the §2 rule; when boot says "no memory
   for the KV cache", add GPU, don't raise `mem-fraction-static`.
3. **The attention type sets how many agents one box can serve.** MLA / DSA (sparse) archs keep a tiny
   KV per token — DeepSeek-V4.1 on B300:4 holds ~68M tokens, GLM-5.3 ~1M+ — so memory never limits
   the running batch and a low `--max-running-requests` only leaves requests queued. Dense-attention
   models are the opposite: KV is the limit and the batch cap protects you. Read the KV capacity off
   the engine (`/get_server_info` → `max_total_num_tokens`) before choosing batch flags.
4. **The engine is the queue; nothing throttles above it.** The gateway forwards every request (no
   concurrency limit, 503 only while the engine is down), the engine queues what doesn't fit in its
   running batch, and the harnesses wait patiently (`API_TIMEOUT_MS` = 2h). Batch flags therefore
   trade per-request latency against throughput — they never turn load into 429s. Prove it on a
   pressure test: zero 429/503/overloaded lines and zero harness API retries in the raw traces.
5. **Every catalog byte is baked into the deployed image.** Any edit — engine args, scaling, even the
   `description` — changes the image, so the next `aom models deploy` *restarts* the endpoint and cuts
   every in-flight request. Apps never redeploy (`ensure_deployed` attaches to a live endpoint), so a
   catalog change only goes live when you deploy it, deliberately, while the endpoint is idle.
6. **Harnesses differ in strictness — test both.** Codex rejects malformed tool-call JSON that Claude
   Code shrugs off, and an engine/parser bug that only one harness trips over looks like a model bug.
7. **Respect the harness's base cost.** Claude Code's system prompt + tool/skill definitions are ~30k
   tokens before any work; a `context_window` near that compacts immediately and derails. Floor it at
   ~128k unless the engine truly can't.
8. **Measure inside the box, not from outside.** `nvidia-smi` and the engine's own load endpoints
   (§4) answer "is it saturated / queued / idle" in seconds; guessing from agent latency does not.
9. **Keep the `description` true to the live config.** It is the sizing logbook — and a description
   that disagrees with `scaling`/`engine_args` (we found one claiming `target_concurrency=20` on an
   endpoint running 4) sends the next person down the wrong path.

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
- [ ] **`scaling`** — `{"target_concurrency": ...}` (see §2). **No `max_containers`**: the catalog
      rejects it. The container cap belongs to the apps using the endpoint (`--max` on deploy).
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
| glm-5.3-uncensored (FP8, full 753B) | 703.7 GiB | **B300:4** | 256k |
| deepseek-v4.1-flash-uncensored (FP8) | 475 GiB | **B300:4** | 256k |
| deepseek-v4.1-flash-abliterated (GGUF, 3.36 bpw) | 235.8 GiB in-mem | **H200:2** | 131k |
| deepseek-v4.1-flash-huihui (GGUF, 1-bit IQ1) | 233.2 GiB in-mem | **H200:2** | 131k (quality-limited) |

> Measured on bring-up: the full **GLM-5.3 FP8 (753B `glm_moe_dsa`)** loads **175.9 GiB/GPU on B300:4**
> (≈66% of usable), leaving **~89.6 GiB/GPU** for KV + capture; SGLang auto-resolves
> `moe_runner_backend=flashinfer_trtllm` (the right Blackwell FP8 MoE kernel — do **not** pin it), cuda
> graphs capture fine at `mem-fraction-static=0.85`, and because the arch is DSA (sparse attention) the
> KV pool holds **~1M tokens**, so 256k is trivially in budget.

**Other sizing knobs:**

- [ ] **`target_concurrency`** — in-flight requests per container before Modal adds another (only
      matters once the endpoint's container cap is above 1). Default 32. Set it to what one container
      can actually serve, not a guess: its `--cuda-graph-max-bs-decode` when the entry sets one (the
      H200:2 DeepSeek-V4 entries: 16; GLM-flash: 32), a pressure-tested number once you have one, else
      the 32 default. GGUF uses `1.0` (llama.cpp `--parallel 1`). Lower values don't protect a GPU —
      the engine queues regardless (theme 4) — they only add a second GPU early.
- [ ] **Metrics on** — every entry carries `--enable-metrics` (SGLang) / `--metrics` (llama.cpp), so
      the engine's own Prometheus `/metrics` is there when a run needs inspecting (§4).
- [ ] **Engine batch cap** (SGLang `--max-running-requests`, llama.cpp `--parallel`) — see theme 3.
      For MLA/DSA archs leave SGLang's cap unset (it sizes the batch from KV capacity): a 100-agent
      test of DeepSeek-V4.1 at `--max-running-requests=20` kept 80 requests queued behind 20 (logbook).
      Set it only for dense-attention models where KV is the limit. Add `--enable-metrics` so the
      engine exports `sglang:*` Prometheus metrics.
- [ ] **Container cap 1** — hard ceiling = graceful throttle under load (never a GPU-per-agent
      fan-out). Not a catalog field: deploy with `--max 1`; apps ask for 1 by default and only ever
      raise it. Raise it deliberately, live (`aom models scale … --max N`), only when a fan-out
      genuinely needs it.
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
aom models deploy <name> --max 1              # image + CPU `prepare` downloads/verifies weights
# or: python scratchpad/deploy_model.py <name>
```

- [ ] `prepare` downloaded all shards and wrote its manifest (no GPU allocated yet).
- [ ] **A redeploy of a live endpoint restarts it** (theme 5) — check nothing is using it first.
- [ ] The **sandbox image** rebuilds itself: its tag hashes the Dockerfile, `init.sh` *and* the
      `agents_on_modal` source baked into it, so the first sandbox created after any code change
      builds it (seconds — the Dockerfile layers are cached). `python scratchpad/build_image.py` still
      forces one.

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
- [ ] **Look inside the live container** (theme 8) — the engine listens on `127.0.0.1:8001` in it:

      ```bash
      PYTHONUTF8=1 modal container list                                   # the ta-… id
      PYTHONUTF8=1 modal container exec <id> -- nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
      PYTHONUTF8=1 modal container exec <id> -- curl -s localhost:8001/get_load        # running vs waiting reqs
      PYTHONUTF8=1 modal container exec <id> -- curl -s localhost:8001/get_server_info # KV capacity, batch cap, gen throughput
      PYTHONUTF8=1 modal container exec <id> -- curl -s localhost:8001/metrics         # with --enable-metrics
      ```

      On Windows the exec's output stream can die mid-print ("detached sandbox"); redirect it to a file
      and read that instead.

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
- [ ] **Swarm pressure test: one wave of ~100 agents against the single container** — a dynamic swarm
      whose orchestrator submits ONE stage with `max_concurrent` = the agent count (all at once, not a
      pool), a mixed harness split (`"harness": "codex"` on every third agent), browser + shell + file
      + a script per agent, a ~15-min deadline, and a per-agent feedback note. Record: dispatch spread
      (first→last agent start), time-to-first-token, done-vs-timeout per harness, engine `get_load`
      and `nvidia-smi` mid-run, sandbox deaths, and API errors/retries in the raw traces (should be
      zero — theme 4). Then have reviewers read individual traces and the feedback notes for friction.
      This is what found the batch-cap queueing, the shared-file clobbering, and the idle-timeout kills
      in the logbook below.
- [ ] **Reasoning efforts map** — spot-check that `none` turns thinking off and that `low/high/max`
      reach the engine (reflected in the trace / token behavior).
- [ ] **Vision** (only if `vision: true`) — one image prompt round-trips.

---

## 6. Cost control + teardown

- [ ] Endpoint ran capped at 1 container the whole time; no runaway fan-out.
- [ ] **Stop everything when done — don't rely on idle scaledown alone:**
      `python scratchpad/stop_all.py <name>`  (or `aom models stop <name>`).
- [ ] **Verify it's actually off:** `aom models status <name>` shows the app stopped / autoscaler cleared
      (not just "min=0"). A force-killed sandbox can leave a stale record — confirm the GPU is gone.
- [ ] Secrets stayed out of git: `.env`, `proxies.csv` are gitignored; nothing residential-proxy was
      written to `/shared`; subagents never used real keys or touched the user's real `~/.claude`/`~/.codex`.

---

## Logbook: per-model breakages discovered

Newest first. Add to this when a bring-up teaches you something (see the note at the top).

**DeepSeek-V4.1 FP8 (B300:4) — 100-agent swarm pressure test — 2026-11:**

- **`--max-running-requests=20` was the bottleneck, not memory.** 100 agents (66 claude, 34 codex)
  dispatched in 64 s; the engine ran 20 and queued 80 (~2M prompt tokens pending), GPUs at 98–99%
  util / ~900 W each / 270 of 275 GB, decode 384–743 tok/s. TTFT p50 18 s / p90 35 s; only 21/100
  finished a 15-min task (claude 12/66, codex 9/34), the rest hit their deadline mid-work. KV pool
  ~68M tokens, so the cap protected nothing. Removed it (and added `--enable-metrics`); see theme 3.
  **Re-run uncapped** (SGLang auto-picked 256): 0 queued, ~1.5k decode tok/s, 99% prefix-cache hit,
  ~10% of KV in use, and **100/100 agents finished** (median 342 s, TTFT p50 9.5 s) — with the
  platform fixes below landed too, so not all of that gain is the cap.
- **The gateway was its own hidden throttle:** aiohttp's default connector capped it at 100 concurrent
  engine connections (lifted), and a stale pooled connection surfaced as a 503 that Codex reports as
  "Selected model is at capacity" and fails the turn on (now retried before any 503).
- **Zero throttling end to end.** No 429/503/overloaded, no harness API retries across all 100 raw
  traces — requests waited in the engine queue (theme 4).
- **Platform lessons the same run surfaced** (fixed in code; listed so a future bring-up doesn't
  misread them as model problems): Modal's sandbox `idle_timeout` reaps a sandbox whose only work is a
  *detached* process, so agents waiting on a busy model were killed silently — runs now start as an
  attached exec. Concurrent appends to one file on the shared Volume clobber each other (100 appends →
  1 line) — agents are told to keep to their own files. Claude Code reset `cd /shared` back to the
  workdir — `/shared` and `/tmp` are now working directories. The sandbox image used to bake in a stale
  `agents_on_modal` — its tag now hashes the source.

**GGUF / llama.cpp path (both `deepseek-v4.1-flash-*-gguf` entries) — 2026-10:**

- **`--jinja` is mandatory for tool calling, and is now auto-injected in `llamacpp.py:launch_argv`.**
  llama-server only renders the request's `tools` into the prompt (via the GGUF chat template) and
  parses tool-calls back into structured `tool_calls` when `--jinja` is set. Without it, the model
  free-forms its native tool-call markup (DeepSeek-V4.1's `<｜DSML｜…｜>` tokens) as **plain assistant
  text**, the harness sees no tool call, and the agent stalls on turn 1. Reasoning (`<think>`) is
  extracted regardless, so the **smoke completion passes but every agent run fails** — a confusing
  split. If you add a new GGUF family and tool calls come back as text, this is why.
- **Pin GGUF `context_window` well above the harness's base cost, not to a token-minimal server `-c`.**
  The first GGUF entries used `-c 32768`; Claude Code's system prompt + tool/skill defs nearly fill
  32k on their own, so the harness hit **premature auto-compaction**, and the model then misread the
  injected compaction summary as a "write a conversation summary" task and derailed. Raising both GGUF
  entries to **131072** fixed it (orcarouter 3.36 bpw then PASSed the full claude e2e). It fits because
  DeepSeek-V4.1-Flash is **MLA** (a tiny compressed KV): ~9 GiB at 131072 vs H200:2's ~46 GiB headroom.
- **codex is not usable on the `deepseek41` GGUF yet; claude is.** The unmerged llama.cpp PR #30817
  emits **invalid tool-call-argument JSON** for this arch (duplicate `workdir` fields, runaway
  single-line output) that codex's strict parser rejects → reconnect loop → context exhaustion. claude
  tolerates/recovers from the occasional malformed arg; codex does not. Treat `deepseek41` GGUF as
  **claude-first** until the PR merges with a proper args grammar.
- **1-bit (IQ1) dynamic quants are below the agentic quality floor.** `huihui UD-IQ1_M` serves,
  generates, and (post-`--jinja`) parses tool calls correctly, but it **narrates steps without calling
  the tool** or **repetition-collapses** mid-task. Fine for a smoke/generation check; not for agent
  work. Prefer a ≥3–4 bpw quant (e.g. orcarouter `Mix-256G`, 3.36 bpw) when a GGUF must drive a harness.

**SGLang path:**

- **GLM-5.3 FP8 (full 753B `glm_moe_dsa`) on B300:4 is comfortable and self-tunes.** See the sizing
  callout in §2: 175.9 GiB/GPU weights, flashinfer_trtllm auto-selected, ~1M-token KV pool. codex
  PASSes clean; claude completes all deliverables (occasional single truncated-tool-arg friction, but
  recovers — a model hiccup, not a pipeline bug).

**Windows tooling (bring-up/monitoring):**

- `modal app logs <app>` **crashes** on the Windows console codec (`'charmap' codec can't encode …`)
  because engine logs contain non-ASCII. Prefix every invocation with `PYTHONUTF8=1
  PYTHONIOENCODING=utf-8`. (It is also a snapshot, not a reliable follow — use the harness's own
  `wait_ready`/`READY` line as the readiness signal and `modal app logs` for a point-in-time look.)

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
