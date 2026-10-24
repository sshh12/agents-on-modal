# codex-via-modal

Run Codex against a Hugging Face model on Modal. The wrapper supports Modal's managed
Endpoints and a generic self-managed SGLang app for models outside the catalog. Codex
is launched with an isolated, Modal-only configuration; wrapper-owned resources are
cleaned up on exit.

<img width="897" height="313" alt="Code_LtxLIER9F5" src="https://github.com/user-attachments/assets/65793242-3629-4959-9f5e-9b2a2f4af000" />

## Quick start

Requires Python 3.10+, `codex-cli 0.146.1`, the `codex` executable on `PATH`, and a
Modal account. The launcher creates `.venv` and installs the pinned dependencies.

```sh
# macOS/Linux
./codex-modal.sh setup
./codex-modal.sh --modal-dry-run
./codex-modal.sh
```

```powershell
# Windows PowerShell
.\codex-modal.ps1 setup
.\codex-modal.ps1 --modal-dry-run
.\codex-modal.ps1
```

The remaining examples use the POSIX launcher. On Windows, substitute
`.\codex-modal.ps1` for `./codex-modal.sh`; every option is the same.

`setup` logs into Modal if needed and creates a proxy token. Credentials come from
`MODAL_PROXY_TOKEN` (or separate ID/secret variables), the OS keyring, or an
owner-only, git-ignored local fallback. For an RBAC environment, use
`setup --modal-env prod`.

## Choose a model

The default preset is `deepseek-ai/DeepSeek-V4-Flash-0731`.

```sh
./codex-modal.sh --modal-list
./codex-modal.sh --modal-pick
./codex-modal.sh --modal-model Qwen/Qwen3.6-27B
```

For a catalog-compatible fine-tune, give Modal the supported base architecture and
the custom checkpoint:

```sh
./codex-modal.sh \
  --modal-model Qwen/Qwen3.6-27B \
  --modal-custom-hf-repo your-org/your-finetune \
  --modal-custom-hf-revision <exact-commit>
```

For a model outside Modal's catalog, deploy the generic SGLang app. GPU topology is
required because it cannot be inferred safely for arbitrary weights. Start with the
model card's known-good SGLang command and repeat `--modal-sglang-arg` for its engine
flags.

```sh
./codex-modal.sh \
  --modal-self-managed \
  --modal-model your-org/your-model \
  --modal-model-revision <exact-commit> \
  --modal-gpu B200:2 \
  --modal-sglang-arg '--trust-remote-code' \
  --modal-startup-timeout 7200
```

Private repositories can use `--modal-hf-token-env HF_PRIVATE_TOKEN`. Large
fine-tunes can reuse unchanged shards from an existing Modal Volume with
`--modal-base-volume` and `--modal-base-volume-path`; run `--modal-help` for the full
surface. SGLang architecture support, parsers, quantization, tensor parallelism, and
GPU memory must still match the checkpoint.

### Catalog deploys

Repeatable self-managed deploys live in `catalog.json` under its `self_managed`
section (a `models` array of full deploy specs; the `modal_managed` section
holds the managed-endpoint presets used by `--modal-list`). Bring one up by
index or name; the served endpoint URL is printed and the app is left warm
(subject to scale-to-zero):

```sh
./modal-deploy.sh glm-5.3-flash-uncensored
./modal-deploy.sh 0 --gpu H200:4 --dry-run
```

Adding a model is one catalog entry, not new code. Afterwards, attach by the
entry's `endpoint_name` with `--modal-use-app <name>`, or from another client such
as blue-green-red via `--model <role>=modal:<endpoint_name>`.

## Lifecycle and cost controls

Self-managed apps default to zero warm containers, at most one serving container, and
GPU scale-to-zero after 300 idle seconds. Set 15 minutes with:

```sh
./codex-modal.sh --modal-self-managed <other-options> --modal-scaledown-window 900
```

Normally, the wrapper stops only the exact endpoint/app it created and deletes only
its owned temporary Volume. A detached watchdog and `cleanup` recover resources after
a local crash. `--modal-keep-endpoint` keeps a deployment intentionally; idle
scale-to-zero still applies.

Attach without taking ownership or changing the deployed scale-down policy:

```sh
./codex-modal.sh --modal-use-endpoint <endpoint-name-or-host>
./codex-modal.sh --modal-use-app <app-name> --modal-model your-org/your-model
```

Useful operator hooks:

```text
setup / cleanup                     credentials / stale-resource recovery
--modal-dry-run                     validate without login or provisioning
--modal-startup-timeout SECONDS     provisioning/readiness timeout
--modal-routing-region REGION       request-proxy placement
--modal-compute-region REGION       repeatable compute placement
--modal-no-history                  disable isolated Codex transcripts
--modal-help                        complete wrapper CLI
```

## Codex isolation

Each run uses a git-ignored `CODEX_HOME` containing exactly one model/provider. The
wrapper pins the main, review, and subagent models to Modal; removes OpenAI credentials
and base-URL overrides; disables analytics, feedback, OpenTelemetry exporters, prompt
logging, web search, plugins/apps, memories, and remote compaction; and rejects Codex
arguments that could override the provider. Project instructions and ordinary Codex
commands still work.

## Local Docker sandbox

`--docker` runs Codex in a throwaway local container with approvals and Codex's
own sandbox bypassed: the agent is unrestricted inside the box, and the box is the
boundary. The image is a general dev/cyber box (compilers, Python/Node/Go/Rust,
network tools, and a headless Chromium the agent can drive). It has internet but
no view of the host — it sits alone on a Docker `--internal` network whose only
peer is an egress broker allowing public addresses on ports 80/443 only (host,
LAN, loopback, link-local, metadata, CGNAT refused; second-layer in-container
iptables default-deny). No host path is mounted (per-run volumes, read-only
rootfs, caps dropped, non-root, no Docker socket), and the broker attaches the
Modal token per request so the agent can use the model but never read it.

```sh
./codex-modal.sh --docker                    # Modal model, Codex in a container
./codex-modal.sh --docker --docker-shell     # a shell in the same sandbox
./codex-modal.sh --docker-prune              # remove leftovers after a crash
```

Each run exports raw logs (session rollout, `agent-console.log`, `egress.jsonl`)
to `.codex-modal/docker-runs/<id>/`. Run `--modal-help` for the full option list;
`--docker-upstream` targets any OpenAI-compatible server with no Modal at all.

## Development

The test suite and dry run do not create paid Modal resources:

```sh
python -m unittest discover -s tests -v
./codex-modal.sh --modal-dry-run
```

See [Modal Endpoints](https://modal.com/docs/guide/endpoints),
[Modal Servers](https://modal.com/docs/guide/servers), and
[Codex configuration](https://developers.openai.com/codex/config-basic).
