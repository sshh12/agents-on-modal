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
   Three that the gateway now absorbs, so a new model doesn't rediscover them:
   - Codex replays a malformed call in every later request, and SGLang rejects the whole history, so
     one bad call ended the session. The gateway wraps such arguments as `{"_invalid_arguments": …}`.
   - Codex's freeform `apply_patch` reaches a chat model only as a Lark grammar in a description, and
     GLM copied the rule names (`begin_patch`, `add_file: x`) instead of the markers. The gateway adds
     a labels-vs-literals note plus a worked example, and repairs the patch envelope (never the hunks).
   - A sampled model can collapse into a short loop ("TheTheThe…") and run to the output cap. The
     gateway ends a reasoning/text stream that is exactly periodic over 1,000 characters as `length`.
7. **Respect the harness's base cost.** Claude Code's system prompt + tool/skill definitions are ~30k
   tokens before any work; a `context_window` near that compacts immediately and derails. Floor it at
   ~128k unless the engine truly can't.
8. **Measure inside the box, not from outside.** `nvidia-smi` and the engine's own load endpoints
   (§4) answer "is it saturated / queued / idle" in seconds; guessing from agent latency does not.
9. **Keep the `description` true to the live config.** It is the sizing logbook — and a description
   that disagrees with `scaling`/`engine_args` (we found one claiming `target_concurrency=20` on an
   endpoint running 4) sends the next person down the wrong path.
10. **Read the checkpoint's `generation_config.json`: the engine applies it to every request.** SGLang's
    default `--sampling-defaults model` fills `temperature/top_p/top_k/min_p/repetition_penalty` from it
    for every field a request leaves out, and the harnesses leave most of them out. The boot log says
    so (`Using default chat sampling params from model generation config: {...}`). Fine-tuned and
    abliterated re-uploads edit it: all three dealignai GLM-5.3 builds add `repetition_penalty: 1.1`,
    which the zai-org original doesn't have. Over a 30k-token agent prompt that penalises nearly every
    token the model has to copy. In a 100-agent run it caused degenerate reasoning
    ("assumedTheTheThe…"), drifting whitespace in Edit `old_string`s, and invented keys
    (`old_string_string`). Pin `sampling` in the catalog (the gateway sends it on every request) rather
    than inheriting it.
11. **At ~100 agents, the engine's HTTP process is the first bottleneck, before the GPU.** One
    single-threaded process renders and tokenizes each turn's whole conversation, runs the streaming
    tool-call parser, and serialises every streamed chunk for every agent. When it saturates, every
    agent's TTFT climbs while the GPUs look busy. Check its `%CPU` in `top` and profile it (§4) before
    touching GPU flags. The levers are an engine release with the model's fast paths (theme 12),
    `--stream-interval=8` (agents don't need per-token streaming), and no per-chunk usage stats (the
    gateway asks only for the final usage).
12. **For a new architecture, the engine release matters more than its flags.** Read the engine's
    commit log for the model's files (`gh api "repos/sgl-project/sglang/commits?path=…"`) before tuning,
    and re-check flag spellings against the new release. SGLang v0.5.19 → v0.5.23 brought GLM-5.3:
    - a Python chat template with a per-segment token-id cache (re-tokenizing the history had been 98%
      of the HTTP process's time);
    - a fix for the glm47 tool parser rescanning the whole tool call on every streamed chunk (O(n²):
      5.9 s of CPU for one 32k-character `write_file` argument);
    - a non-streaming parser fix (it had turned a string argument `"true"` into `"True"`);
    - DSA/MTP fixes, and the rename of DP attention to `--attn-dp-size`.
13. **Test tool-call fidelity, not just tool calling.** Run awkward argument values (JSON-looking
    strings, digit strings, quotes and backslashes, typed ints/bools/objects, parallel calls) through
    every wire path and compare them byte for byte. Replay the harness's *real* request (Codex's
    freeform `apply_patch` grammar, Claude Code's tool set) rather than a toy tool, then read the
    harness's own rejections in a pressure run: Codex logs every refused patch and unparseable
    argument as `ERROR codex_core::tools::router`.
14. **Spec decoding is usually the throughput lever for big MoE decode, and its defaults are tuned
    for latency.** At 50–100 concurrent agents, decode on a 4-GPU MoE box is bound by reading nearly
    every expert's weights each step, so verifying several MTP draft tokens per step costs little.
    But SGLang silently caps `--max-running-requests` at 48 when speculative decoding is on (the boot
    log says "Max running requests is reset to 48"), which queues half of a 100-agent swarm. Set it
    explicitly.

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
      and read that instead. Past a few KB it is cut off reliably (the full `/metrics` or
      `/get_server_info`), so filter inside the container (`grep`/`awk` the metric names, or print
      the few `get_server_info` keys you need) and keep the output small. SGLang v0.5.23 dropped
      `/get_load`; its Prometheus gauges (`sglang:num_running_reqs`, `num_queue_reqs`,
      `gen_throughput`, `token_usage`) say the same.
- [ ] **Profile the engine's HTTP process under load** (theme 11): it is the `python` process whose
      children are the `sglang::scheduler`s (`top -b -n 2 -o %CPU` in the container). Near 100% CPU
      means it's the bottleneck. `pip install py-spy` in the container, then `py-spy record -p
      <pid> -d 60 --nonblocking -f raw -o /tmp/spy.raw` and sum the leaf frames: on GLM-5.3 the leaves
      were tokenizer `encode` (98%) on v0.5.19, then `jsonschema` tool-schema validation, which SGLang
      re-runs uncached for every tool of every request, on v0.5.23. Take ≥60 s: gVisor sampling is
      slow, and a 30 s sample (29 stacks) pointed at the wrong function.
- [ ] **Check the boot log for silent overrides**: `Using default chat sampling params from model
      generation config` (theme 10), `Max running requests is reset to 48 for speculative decoding`
      (theme 14), `Using Python chat template 'glm-5.3' in place of the Jinja chat template` (a newer
      engine replacing the checkpoint's template, possibly with different defaults: here
      `clear_thinking`, below).
- [ ] **Cold start is weights + JIT + graphs; time each.** The GLM-5.3 FP8 cold start broke down as:
      - ~1.5 min container start;
      - weights (704 GiB): 536 s single-pass, 384 s with `--weight-loader-prefetch-checkpoints`, which
        has each TP rank prefetch a slice into the page cache so ranks don't each read every shard
        from the Volume;
      - DeepGEMM JIT warmup: 16,384 M values per GEMM shape by default, about 2.5k with
        `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` in `engine_env`;
      - CUDA graph capture: decode, plus prefill shapes on v0.5.23.

      A shared-Volume JIT cache was tried and reverted (`a59c06d`: slower than recompiling).

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
- [ ] **Serving-config rounds: same wave, one change set per round.** To tune engine flags, skip the
      orchestrator: create the ~100 sandboxes directly, two thirds claude and one third codex, all
      on one short tool-heavy coding task. The GLM-5.3 task: write a module + pytest tests, run them,
      rename a function with the *edit tool*, write a JSON result to `/shared`, write a friction note.
      Give a ~10 min deadline. Keep the task and image identical across rounds so only the serving
      config differs. Sample the engine's Prometheus counters every 30 s. Compare on:
      - agents done by 4 / 6 / 8 / 10 min;
      - decode tok/s while ≥50 agents are still working (a whole-round average falls when agents
        finish early);
      - total tokens generated (fewer for the same work = less waste);
      - TTFT and the HTTP process's CPU;
      - tool anomalies in the traces: Claude Edit `old_string` misses, Codex `apply_patch`
        rejections and unparseable arguments, degenerate run-on text, harness API retries.
      Each config change is a redeploy and a full cold start (~13–16 min on B300:4), so batch related
      changes into one round and chain the deploy to the end of the previous round's trace collection.
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

**GLM-5.3 FP8 uncensored (B300:4) — serving-config rounds, 100 agents each — 2026-11:**

Every round used the same 100 Modal agents (66 claude, 34 codex) on the same 10-minute coding task.
Each round was one redeploy plus a full cold start (§5).

| round | change | agents done (by 4/6/8/10 min) | decode tok/s, ≥50 agents active | per-stream tok/s | TTFT |
|---|---|---|---|---|---|
| R0 | baseline: v0.5.19, inherited sampling | 2 / 50 / 71 / 82 | 1,460 | 25.6 | 1.9 s |
| R1 | v0.5.23, `repetition_penalty` 1.0, `--stream-interval=8`, no per-chunk usage | 25 / 95 / 98 / 98 | 1,731 | 23.8 | 0.45 s |
| R2 | + MTP (EAGLE 3/1/4) with `--max-running-requests=128` | 41 / 93 / 97 / 99 | 2,103 | 34.5 | 0.84 s |
| R3 | + gateway `apply_patch` fix + loop guard, pytest in the image | 92 (4 min) / 100 (6 min) | 2,293 | 30.3 | 0.84 s |

The "agents done" column counts from the last agent's launch. R3's launch took 216 s, since the
first sandbox rebuilt the image layer with pytest, so its early agents had a head start. Against R0,
R3's median time-to-done is 151 s vs 261 s (claude) and 194 s vs 383 s (codex); request e2e is
10.2 s vs 16.5 s; and it generated 27% fewer tokens for more finished work. Codex `apply_patch`
rejections went 58 → 2; Claude Edit misses went 10 → 4.

- **R0's bottleneck was the engine's HTTP process, not the GPUs.** It sat at 96% CPU, and 98% of its
  samples were tokenizer `encode` of the whole conversation every turn. The GPUs were at 100% "util"
  the whole time and KV peaked at 72% (1.12M tokens), with nothing queued. On v0.5.23 the same
  process ran at 39%.
- **Most of R1's jump was the model behaving better, not the engine running faster.** Decode speed
  per stream was unchanged, yet agents finished 20–28% sooner while generating 19% fewer tokens.
  With the checkpoint's `repetition_penalty: 1.1` removed, degenerate run-on text fell from 22
  events to 5, and Codex `apply_patch` rejections fell from 58 to 12.
- **MTP's gain is real once the 48-request cap is lifted:** about 3.0 of 4 draft tokens accepted,
  +21% aggregate and +45% per-stream decode, and request e2e 13.9 → 11.5 s.
- **Codex `apply_patch` was failing on the grammar, not the model's patch.** All 40 rejected patches
  captured from R2's Codex rollouts began with the literal line `begin_patch`, the Lark rule name,
  with a correct patch after it. The gateway now explains rule names vs literals, includes an example,
  and repairs the envelope. Replayed offline, the repair fixed all 40 and left all 159 valid patches
  unchanged.
- **Runaway loops are rare but expensive.** Two R1 agents spent their whole 10 minutes in one
  generation (a 100k-character "TheTheThe…", a "FailFailFail…"). Each ran to the 32k output cap
  while holding a decode slot. The gateway now cuts them at 1,000 characters of exact repetition.
- **Sandbox friction worth fixing:** 95–99 of 100 agents hit `No module named pytest` and spent turns
  installing it. pytest is now in the image.
- The cold start was ~15 min. Weights took 536 s → 384 s with checkpoint prefetch; DeepGEMM warmup
  and CUDA-graph capture (decode, prefill on v0.5.23, MTP verify) took the rest.

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

- **GLM-5.3 FP8 (full 753B `glm_moe_dsa`) on B300:4 is comfortable and self-tunes** on memory. See
  the sizing callout in §2: 175.9 GiB/GPU weights, flashinfer_trtllm auto-selected, ~1M-token KV
  pool. codex PASSes clean; claude completes all deliverables. The occasional truncated or garbled
  tool argument seen at bring-up was mostly the checkpoint's hidden `repetition_penalty: 1.1`
  (theme 10), not a model hiccup. The serving rounds at the top of this logbook also found the
  engine, sampling and gateway settings it needs at 100 agents.

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
