"""Wrapper option parsing while preserving Codex's own argument surface."""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

from .paths import CATALOG_PATH


VALID_REASONING = {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
ROUTING_REGIONS = {"us-west", "us-east", "ca-central", "eu-west", "ap-south"}
ENDPOINT_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
ENDPOINT_HOST = re.compile(
    r"^([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)\.([a-z0-9-]+)\.modal\.direct$"
)


@dataclass
class WrapperOptions:
    action: str = "run"
    preset: str | None = None
    model: str | None = None
    model_revision: str | None = None
    custom_hf_repo: str | None = None
    custom_hf_revision: str | None = None
    hf_token_env: str | None = None
    self_managed: bool = False
    base_volume: str | None = None
    base_volume_path: str | None = None
    gpu: str | None = None
    runtime: str | None = None
    download_subdir: str | None = None
    sglang_image: str | None = None
    sglang_args: list[str] = field(default_factory=list)
    serving_pip: list[str] = field(default_factory=list)
    cpu: int | None = None
    memory: int | None = None
    scaledown_window: int | None = None
    target_inputs: int | None = None
    endpoint_name: str | None = None
    use_endpoint: str | None = None
    use_app: str | None = None
    environment_name: str | None = None
    routing_region: str = "us-west"
    compute_regions: list[str] = field(default_factory=list)
    colocate_compute: bool = False
    context_window: int | None = None
    reasoning_effort: str | None = None
    reasoning_levels: tuple[str, ...] | None = None
    startup_timeout: int = 2700
    wait_for_endpoint: bool = True
    keep_endpoint: bool = False
    persist_history: bool = True
    dry_run: bool = False
    pick: bool = False
    force_token: bool = False
    docker: bool = False
    docker_build: bool = False
    docker_upstream: str | None = None
    docker_upstream_auth_env: str | None = None
    docker_model_slug: str | None = None
    docker_image: str | None = None
    docker_codex_version: str | None = None
    docker_packages: str | None = None
    docker_allow_ports: list[int] = field(default_factory=list)
    docker_allow_hosts: list[str] = field(default_factory=list)
    docker_firewall: str = "enforce"
    docker_memory: str | None = None
    docker_cpus: str | None = None
    docker_pids: int | None = None
    docker_copy_in: str | None = None
    docker_export: str | None = None
    docker_export_work: bool = False
    docker_keep: bool = False
    docker_shell: bool = False
    docker_rust_log: str | None = None
    docker_env_note: str | None = None
    docker_no_env_doc: bool = False
    codex_arguments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedModel:
    preset_name: str
    base_model: str
    display_model: str
    custom_hf_repo: str | None
    context_window: int
    reasoning_effort: str
    reasoning_levels: tuple[str, ...]


VALUE_OPTIONS = {
    "--modal-preset": "preset",
    "--modal-model": "model",
    "--modal-model-revision": "model_revision",
    "--modal-custom-hf-repo": "custom_hf_repo",
    "--modal-custom-hf-revision": "custom_hf_revision",
    "--modal-hf-token-env": "hf_token_env",
    "--modal-base-volume": "base_volume",
    "--modal-base-volume-path": "base_volume_path",
    "--modal-gpu": "gpu",
    "--modal-runtime": "runtime",
    "--modal-download-subdir": "download_subdir",
    "--modal-sglang-image": "sglang_image",
    "--modal-cpu": "cpu",
    "--modal-memory": "memory",
    "--modal-scaledown-window": "scaledown_window",
    "--modal-target-inputs": "target_inputs",
    "--modal-endpoint-name": "endpoint_name",
    "--modal-use-endpoint": "use_endpoint",
    "--modal-use-app": "use_app",
    "--modal-env": "environment_name",
    "--modal-routing-region": "routing_region",
    "--modal-context-window": "context_window",
    "--modal-reasoning-effort": "reasoning_effort",
    "--modal-reasoning-levels": "reasoning_levels",
    "--modal-startup-timeout": "startup_timeout",
    "--docker-upstream": "docker_upstream",
    "--docker-upstream-auth-env": "docker_upstream_auth_env",
    "--docker-model-slug": "docker_model_slug",
    "--docker-image": "docker_image",
    "--docker-codex-version": "docker_codex_version",
    "--docker-packages": "docker_packages",
    "--docker-firewall": "docker_firewall",
    "--docker-memory": "docker_memory",
    "--docker-cpus": "docker_cpus",
    "--docker-pids": "docker_pids",
    "--docker-copy-in": "docker_copy_in",
    "--docker-export": "docker_export",
    "--docker-rust-log": "docker_rust_log",
    "--docker-env-note": "docker_env_note",
}

REPEATED_OPTIONS = {
    "--modal-compute-region": "compute_regions",
    "--modal-sglang-arg": "sglang_args",
    "--modal-serving-pip": "serving_pip",
    "--docker-allow-port": "docker_allow_ports",
    "--docker-allow-host": "docker_allow_hosts",
}

DOCKER_FIREWALL_MODES = {"enforce", "warn", "off"}


def _set_action(options: WrapperOptions, action: str) -> None:
    if options.action != "run" and options.action != action:
        raise ValueError(f"Wrapper actions {options.action!r} and {action!r} cannot be combined.")
    options.action = action


def parse_arguments(arguments: list[str]) -> WrapperOptions:
    options = WrapperOptions()
    index = 0
    if arguments and arguments[0] in {"setup", "cleanup"}:
        options.action = arguments[0]
        index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            options.codex_arguments.append("--")
            options.codex_arguments.extend(arguments[index + 1 :])
            break
        if argument in VALUE_OPTIONS or argument in REPEATED_OPTIONS:
            if index + 1 >= len(arguments):
                raise ValueError(f"{argument} requires a value.")
            value = arguments[index + 1]
            if argument in REPEATED_OPTIONS:
                target = getattr(options, REPEATED_OPTIONS[argument])
                if argument == "--docker-allow-port":
                    try:
                        target.append(int(value))
                    except ValueError as error:
                        raise ValueError("--docker-allow-port requires an integer.") from error
                else:
                    target.append(value)
            else:
                attribute = VALUE_OPTIONS[argument]
                if attribute in {
                    "context_window",
                    "startup_timeout",
                    "cpu",
                    "memory",
                    "scaledown_window",
                    "target_inputs",
                    "docker_pids",
                }:
                    try:
                        value = int(value)
                    except ValueError as error:
                        raise ValueError(f"{argument} requires an integer.") from error
                elif attribute == "reasoning_levels":
                    value = tuple(part.strip() for part in value.split(",") if part.strip())
                setattr(options, attribute, value)
            index += 2
            continue
        flags = {
            "--modal-help": "help",
            "--modal-setup": "setup",
            "--modal-cleanup": "cleanup",
            "--modal-list": "list",
        }
        if argument in flags:
            _set_action(options, flags[argument])
        elif argument == "--modal-pick":
            options.pick = True
        elif argument == "--modal-self-managed":
            options.self_managed = True
        elif argument == "--modal-colocate-compute":
            options.colocate_compute = True
        elif argument == "--modal-no-wait":
            options.wait_for_endpoint = False
        elif argument == "--modal-keep-endpoint":
            options.keep_endpoint = True
        elif argument == "--modal-no-history":
            options.persist_history = False
        elif argument == "--modal-dry-run":
            options.dry_run = True
        elif argument == "--modal-force-token":
            options.force_token = True
        elif argument == "--docker":
            options.docker = True
        elif argument == "--docker-build":
            options.docker_build = True
        elif argument == "--docker-export-work":
            options.docker_export_work = True
        elif argument == "--docker-keep":
            options.docker_keep = True
        elif argument == "--docker-shell":
            options.docker_shell = True
        elif argument == "--docker-no-env-doc":
            options.docker_no_env_doc = True
        elif argument == "--docker-prune":
            _set_action(options, "docker-prune")
        elif argument.startswith("--modal-") or argument.startswith("--docker-"):
            raise ValueError(f"Unknown wrapper option {argument!r}. Run codex-modal --modal-help.")
        else:
            options.codex_arguments.append(argument)
        index += 1
    validate_wrapper_options(options)
    return options


SANDBOX_POLICY_ARGUMENTS = (
    "-s",
    "--sandbox",
    "-a",
    "--ask-for-approval",
    "--approve-for-me",
    "--dangerously-bypass-approvals-and-sandbox",
)


def codex_sets_own_policy(arguments: list[str]) -> bool:
    """True when the caller already chose an approval/sandbox policy for Codex."""

    for argument in _before_double_dash(arguments):
        if argument in SANDBOX_POLICY_ARGUMENTS:
            return True
        if any(
            argument.startswith(f"{option}=")
            for option in SANDBOX_POLICY_ARGUMENTS
            if option.startswith("--")
        ):
            return True
        if re.fullmatch(r"-[sa].+", argument):
            return True
    return False


def _validate_docker_options(options: WrapperOptions) -> None:
    supplied = [
        name
        for name, value in (
            ("--docker-build", options.docker_build),
            ("--docker-upstream", options.docker_upstream),
            ("--docker-upstream-auth-env", options.docker_upstream_auth_env),
            ("--docker-model-slug", options.docker_model_slug),
            ("--docker-image", options.docker_image),
            ("--docker-codex-version", options.docker_codex_version),
            ("--docker-packages", options.docker_packages),
            ("--docker-allow-port", options.docker_allow_ports),
            ("--docker-allow-host", options.docker_allow_hosts),
            ("--docker-memory", options.docker_memory),
            ("--docker-cpus", options.docker_cpus),
            ("--docker-pids", options.docker_pids),
            ("--docker-copy-in", options.docker_copy_in),
            ("--docker-export", options.docker_export),
            ("--docker-export-work", options.docker_export_work),
            ("--docker-keep", options.docker_keep),
            ("--docker-shell", options.docker_shell),
            ("--docker-env-note", options.docker_env_note),
            ("--docker-no-env-doc", options.docker_no_env_doc),
            ("--docker-rust-log", options.docker_rust_log),
        )
        if value
    ]
    if options.docker_firewall != "enforce":
        supplied.append("--docker-firewall")
    if not options.docker and options.action == "run" and supplied:
        raise ValueError(f"{supplied[0]} requires --docker.")
    if not options.docker:
        return
    if options.docker_firewall not in DOCKER_FIREWALL_MODES:
        raise ValueError(
            "--docker-firewall must be one of: "
            + ", ".join(sorted(DOCKER_FIREWALL_MODES))
            + "."
        )
    if options.docker_upstream is not None:
        if not re.match(r"^https?://[^\s/]+", options.docker_upstream):
            raise ValueError("--docker-upstream must be an http:// or https:// base URL.")
    elif options.docker_upstream_auth_env:
        raise ValueError("--docker-upstream-auth-env requires --docker-upstream.")
    if options.docker_upstream and (
        options.self_managed or options.use_endpoint or options.use_app
    ):
        raise ValueError(
            "--docker-upstream replaces the Modal provider entirely; it cannot be "
            "combined with Modal deployment or attach options."
        )
    for port in options.docker_allow_ports:
        if not 1 <= port <= 65535:
            raise ValueError(f"--docker-allow-port {port} is out of range.")
    if options.docker_pids is not None and options.docker_pids < 32:
        raise ValueError("--docker-pids must be at least 32.")


def validate_wrapper_options(options: WrapperOptions) -> None:
    _validate_docker_options(options)
    if options.context_window is not None and options.context_window < 8192:
        raise ValueError("--modal-context-window must be at least 8192.")
    if options.startup_timeout < 30:
        raise ValueError("--modal-startup-timeout must be at least 30 seconds.")
    if options.startup_timeout > 86_400:
        raise ValueError("--modal-startup-timeout cannot exceed 86400 seconds.")
    if options.routing_region not in ROUTING_REGIONS:
        raise ValueError(f"Unsupported Modal routing region {options.routing_region!r}.")
    for region in options.compute_regions:
        if not re.fullmatch(r"[a-z0-9-]+", region):
            raise ValueError(f"Invalid Modal compute region {region!r}.")
    if options.colocate_compute and options.compute_regions:
        raise ValueError(
            "--modal-colocate-compute and --modal-compute-region are mutually exclusive."
        )
    if options.custom_hf_revision and not options.custom_hf_repo:
        raise ValueError("--modal-custom-hf-revision requires --modal-custom-hf-repo.")
    if options.hf_token_env and not options.custom_hf_repo:
        if not options.self_managed:
            raise ValueError("--modal-hf-token-env requires --modal-custom-hf-repo.")
    attachment_options = [
        value
        for value in (options.use_endpoint, options.use_app, options.endpoint_name)
        if value
    ]
    if len(attachment_options) > 1:
        raise ValueError(
            "--modal-use-endpoint, --modal-use-app, and --modal-endpoint-name "
            "are mutually exclusive."
        )
    if options.use_app:
        assert_endpoint_name(options.use_app)
    if options.endpoint_name:
        assert_endpoint_name(options.endpoint_name)
    custom_app_values = {
        "--modal-model-revision": options.model_revision,
        "--modal-base-volume": options.base_volume,
        "--modal-base-volume-path": options.base_volume_path,
        "--modal-gpu": options.gpu,
        "--modal-runtime": options.runtime,
        "--modal-download-subdir": options.download_subdir,
        "--modal-sglang-image": options.sglang_image,
        "--modal-sglang-arg": options.sglang_args,
        "--modal-serving-pip": options.serving_pip,
        "--modal-cpu": options.cpu,
        "--modal-memory": options.memory,
        "--modal-scaledown-window": options.scaledown_window,
        "--modal-target-inputs": options.target_inputs,
    }
    if not options.self_managed:
        selected = next((name for name, value in custom_app_values.items() if value), None)
        if selected:
            raise ValueError(f"{selected} requires --modal-self-managed.")
        return

    if options.use_endpoint or options.use_app:
        raise ValueError(
            "--modal-self-managed cannot be combined with an attach option; omit "
            "--modal-self-managed when attaching to an existing resource."
        )
    if not options.gpu:
        raise ValueError("--modal-self-managed requires an explicit --modal-gpu.")
    if options.runtime is not None:
        from .runtimes import KNOWN_RUNTIMES

        if options.runtime not in KNOWN_RUNTIMES:
            raise ValueError(
                f"--modal-runtime must be one of: {', '.join(KNOWN_RUNTIMES)}."
            )
    if bool(options.base_volume) != bool(options.base_volume_path):
        raise ValueError(
            "--modal-base-volume and --modal-base-volume-path must be provided together."
        )
    if options.base_volume and not options.model_revision:
        raise ValueError(
            "--modal-base-volume requires --modal-model-revision so shard reuse is "
            "validated against an exact base revision."
        )
    if options.model_revision and options.custom_hf_repo and not options.base_volume:
        raise ValueError(
            "--modal-model-revision is only used for the served repo or for validating "
            "--modal-base-volume reuse."
        )
    if options.cpu is not None and options.cpu < 1:
        raise ValueError("--modal-cpu must be at least 1.")
    if options.memory is not None and options.memory < 1024:
        raise ValueError("--modal-memory must be at least 1024 MiB.")
    if options.scaledown_window is not None and options.scaledown_window < 0:
        raise ValueError("--modal-scaledown-window cannot be negative.")
    if options.target_inputs is not None and options.target_inputs < 1:
        raise ValueError("--modal-target-inputs must be at least 1.")


def load_presets() -> dict[str, Any]:
    """Return the Modal managed-endpoint presets (the ``modal_managed`` section
    of the unified catalog) in the ``{"default", "presets"}`` shape."""
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not load the catalog from {CATALOG_PATH}: {error}") from error
    document = catalog.get("modal_managed") if isinstance(catalog, dict) else None
    if not isinstance(document, dict) or not isinstance(document.get("presets"), dict):
        raise RuntimeError(
            f"Invalid catalog: {CATALOG_PATH} has no 'modal_managed' section with presets."
        )
    return document


def choose_interactively(options: WrapperOptions, document: dict[str, Any]) -> None:
    presets = list(document["presets"].items())
    print("Available model presets:")
    for index, (name, preset) in enumerate(presets, start=1):
        print(f"  {index}) {name} - {preset.get('description', '')}")
    print("  A) arbitrary Hugging Face model or fine-tune")
    choice = input("Selection: ").strip()
    if choice.lower() == "a":
        options.model = input("Modal catalog/base model repo ID: ").strip()
        custom = input("Custom Hugging Face fine-tune repo ID (blank for base model): ").strip()
        if custom:
            options.custom_hf_repo = custom
        return
    try:
        selected = int(choice)
    except ValueError as error:
        raise ValueError("Invalid model selection.") from error
    if selected < 1 or selected > len(presets):
        raise ValueError("Invalid model selection.")
    options.preset = presets[selected - 1][0]


def resolve_model(options: WrapperOptions, document: dict[str, Any]) -> ResolvedModel:
    if options.pick:
        choose_interactively(options, document)
    default_name = str(document.get("default", ""))
    preset_name = options.preset or default_name
    preset = document["presets"].get(preset_name)
    if not isinstance(preset, dict):
        raise ValueError(f"Unknown preset {preset_name!r}. Run codex-modal --modal-list.")
    base_model = options.model or str(preset.get("base_model", ""))
    if not base_model or any(character.isspace() for character in base_model):
        raise ValueError("The Modal catalog/base model repo ID is empty or invalid.")
    display_model = options.custom_hf_repo or base_model
    arbitrary = options.model is not None and options.model != preset.get("base_model")
    context_window = options.context_window or (
        131072 if arbitrary else int(preset["context_window"])
    )
    reasoning_effort = options.reasoning_effort or (
        "high" if arbitrary else str(preset["reasoning_effort"])
    )
    reasoning_levels = options.reasoning_levels or tuple(
        ("low", "medium", "high") if arbitrary else preset["reasoning_levels"]
    )
    if reasoning_effort not in VALID_REASONING:
        raise ValueError(f"Unsupported Codex reasoning effort {reasoning_effort!r}.")
    if any(level not in VALID_REASONING for level in reasoning_levels):
        raise ValueError("One or more --modal-reasoning-levels values are unsupported.")
    if reasoning_effort not in reasoning_levels:
        reasoning_levels = (*reasoning_levels, reasoning_effort)
    return ResolvedModel(
        preset_name=preset_name,
        base_model=base_model,
        display_model=display_model,
        custom_hf_repo=options.custom_hf_repo,
        context_window=context_window,
        reasoning_effort=reasoning_effort,
        reasoning_levels=tuple(reasoning_levels),
    )


def assert_endpoint_name(name: str) -> None:
    if not ENDPOINT_NAME.fullmatch(name):
        raise ValueError(
            "Modal endpoint names must be lowercase DNS labels of at most 63 characters."
        )


def generated_endpoint_name(display_model: str) -> str:
    leaf = display_model.rsplit("/", 1)[-1].lower()
    leaf = re.sub(r"[^a-z0-9]+", "-", leaf).strip("-") or "model"
    suffix = secrets.token_hex(4)
    name = f"codex-{leaf[:43]}-{suffix}".strip("-")
    assert_endpoint_name(name)
    return name


def endpoint_target(options: WrapperOptions, display_model: str) -> tuple[str, str, str]:
    """Return endpoint name, hostname/model slug, and shared Responses base URL."""

    region = options.routing_region
    if options.use_app:
        name = options.use_app
    elif options.use_endpoint:
        host_match = ENDPOINT_HOST.fullmatch(options.use_endpoint)
        if host_match:
            name, region = host_match.groups()
            options.routing_region = region
        else:
            name = options.use_endpoint
            assert_endpoint_name(name)
    else:
        name = options.endpoint_name or generated_endpoint_name(display_model)
    host = f"{name}.{region}.modal.direct"
    return name, host, f"https://inference.{region}.modal.direct/v1"


BLOCKED_CODEX_COMMANDS = {
    "login",
    "cloud",
    "app",
    "app-server",
    "mcp-server",
    "exec-server",
    "remote-control",
    "update",
}
NO_INFERENCE_COMMANDS = {
    "help",
    "completion",
    "features",
    "doctor",
    "debug",
    "mcp",
    "plugin",
    "sandbox",
    "apply",
    "archive",
    "delete",
    "unarchive",
    "logout",
}


def _before_double_dash(arguments: list[str]) -> list[str]:
    try:
        return arguments[: arguments.index("--")]
    except ValueError:
        return arguments


def assert_isolated_codex_arguments(arguments: list[str]) -> None:
    visible = _before_double_dash(arguments)
    forbidden_long = (
        "--model",
        "--config",
        "--profile",
        "--oss",
        "--local-provider",
        "--search",
        "--enable",
        "--remote",
        "--remote-auth-token-env",
    )
    for argument in visible:
        if argument in forbidden_long or any(
            argument.startswith(f"{option}=") for option in forbidden_long
        ):
            raise ValueError(
                f"Codex option {argument!r} is blocked because it can escape the Modal-only "
                "provider/configuration. Use the corresponding --modal-* option instead."
            )
        if argument in {"-m", "-c", "-p"} or re.fullmatch(r"-[mcp].+", argument):
            raise ValueError(
                f"Codex option {argument!r} is blocked because it can override the isolated model."
            )
    command_name = first_codex_word(visible)
    if command_name in BLOCKED_CODEX_COMMANDS:
        raise ValueError(
            f"Codex command {command_name!r} is blocked because it leaves the local "
            "Modal-only inference path. Run the regular codex executable explicitly if intended."
        )


def first_codex_word(arguments: list[str]) -> str | None:
    options_with_values = {
        "-C",
        "--cd",
        "--add-dir",
        "-a",
        "--ask-for-approval",
        "-s",
        "--sandbox",
        "-i",
        "--image",
        "--disable",
    }
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            return arguments[index + 1] if index + 1 < len(arguments) else None
        if argument in options_with_values:
            index += 2
            continue
        if any(argument.startswith(f"{option}=") for option in options_with_values if option.startswith("--")):
            index += 1
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument
    return None


def codex_needs_inference(arguments: list[str]) -> bool:
    visible = _before_double_dash(arguments)
    if any(argument in {"-h", "--help", "-V", "--version"} for argument in visible):
        return False
    command_name = first_codex_word(arguments)
    return command_name not in NO_INFERENCE_COMMANDS
