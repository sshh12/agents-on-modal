"""The ``aom`` command line.

This is the only module that prints and returns exit codes; everything it calls raises
:class:`~agents_on_modal.config.AomError` subclasses on failure. ``main`` wraps every command in a
single ``try/except AomError`` that writes to stderr and returns ``1``.

Commands::

    aom setup                                  authenticate to Modal and store a proxy token
    aom models list|deploy|scale|stop|status   manage model deployments
    aom run    --harness --model ... PROMPT     one headless run, streamed to stdout
    aom shell  --harness --model ...            an interactive harness TUI on this terminal
    aom swarm  --harness --model --count N ...  fan out N Modal agents; open the dashboard

``modal`` is imported lazily inside ``setup`` only, so the CLI imports off-host.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import agents_on_modal.models.lifecycle as model_ops
from . import models, sandbox, swarm
from .config import AomError

# ``model_ops`` is the lifecycle submodule: deploy/endpoint/scale/wait_ready/stop/status. It is
# aliased (rather than calling ``models.deploy`` etc.) so these stay easy to monkeypatch in tests.
# ``models.get``/``models.list_models`` are called through ``models`` directly.

SANDBOX_KINDS = ("local", "docker", "modal")
HARNESSES = ("claude", "codex")


# --------------------------------------------------------------------------------------------- #
# event rendering                                                                               #
# --------------------------------------------------------------------------------------------- #
def _short(value: Any, limit: int = 240) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _print_event(event: Any) -> None:
    """Render one normalized :class:`~agents_on_modal.traces.Event` for a human (no emoji)."""

    kind = event.kind
    data = event.data or {}
    tag = f"{event.agent}: " if event.agent else ""
    if kind == "session":
        print(f"{tag}-- session {data.get('session_id', '')} model={data.get('model', '')}")
    elif kind == "text":
        text = str(data.get("text", "")).rstrip()
        if text:
            print(f"{tag}{text}")
    elif kind == "thinking":
        text = str(data.get("text", "")).strip()
        if text:
            print(f"{tag}(thinking) {text}")
    elif kind == "tool_call":
        print(f"{tag}> {data.get('name', '?')} {_short(data.get('input', {}))}")
    elif kind == "tool_result":
        flag = "error " if data.get("is_error") else ""
        print(f"{tag}< {flag}{_short(data.get('output', ''))}")
    elif kind == "usage":
        print(f"{tag}[usage in={data.get('input_tokens')} out={data.get('output_tokens')}"
              f" ctx={data.get('context_tokens')}]")
    elif kind == "compaction":
        print(f"{tag}[compaction {data.get('pre_tokens')}->{data.get('post_tokens')}]")
    elif kind == "subagent":
        print(f"{tag}[subagent {data.get('id', '')} {data.get('status', '')}]")
    elif kind == "goal":
        print(f"{tag}[goal {data.get('status', '')}] {data.get('reason', '')}".rstrip())
    elif kind == "turn_end":
        result = data.get("result")
        if result:
            print(f"{tag}{_short(result)}")
    elif kind == "error":
        print(f"{tag}[error] {data.get('message', '')}", file=sys.stderr)
    elif kind == "status":
        extra = " ".join(f"{k}={v}" for k, v in data.items() if k != "event")
        print(f"{tag}[{data.get('event', 'status')}] {extra}".rstrip())
    elif kind == "log":
        text = str(data.get("text", "")).rstrip()
        if text:
            print(f"{tag}{text}")


def _stream_run(run: Any) -> int:
    """Stream a run to stdout until it is terminal; exit 0 on ``done``, else 1."""

    for event in run.events(follow=True):
        _print_event(event)
    state = str(run.status().get("state", "unknown"))
    print(f"\n[run {state}]")
    return 0 if state == "done" else 1


# --------------------------------------------------------------------------------------------- #
# shared helpers                                                                                #
# --------------------------------------------------------------------------------------------- #
def _resolve_endpoint(args: argparse.Namespace) -> Any:
    """Attach to the model (default) or deploy it first when ``--deploy`` is given."""

    model = models.get(args.model)
    if getattr(args, "deploy", False):
        endpoint = model_ops.deploy(model)
        model_ops.wait_ready(endpoint)
        return endpoint
    return model_ops.endpoint(model)


def _create_sandbox(args: argparse.Namespace, endpoint: Any) -> Any:
    options: dict[str, Any] = {"endpoint": endpoint}
    if getattr(args, "proxy", None):
        options["proxy"] = args.proxy
    if getattr(args, "workdir", None):
        options["workdir"] = args.workdir
    return sandbox.create(args.sandbox, **options)


# --------------------------------------------------------------------------------------------- #
# commands                                                                                      #
# --------------------------------------------------------------------------------------------- #
def _cmd_setup(args: argparse.Namespace) -> int:
    from .config import ProxyToken, save_token

    try:
        import modal
    except ImportError as error:  # pragma: no cover - modal is a hard dep
        raise AomError("The `modal` package is required for `aom setup`.") from error

    try:
        workspace = modal.Workspace.from_context()
    except Exception as error:  # noqa: BLE001 - not authenticated
        raise AomError(
            "Not authenticated to Modal. Run `modal setup` (or `python -m modal setup`) first, "
            f"then re-run `aom setup`. ({error})"
        ) from error

    try:
        minted = workspace.proxy_tokens.create()
    except Exception as error:  # noqa: BLE001
        raise AomError(f"Could not mint a Modal proxy token: {error}") from error

    token = ProxyToken(minted.token_id, minted.token_secret)
    path = save_token(token)
    print(f"Saved Modal proxy token {token.token_id} to {path}")
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    for model in models.list_models():
        print(f"{model.name:28}  {model.gpu:10}  {model.description}")
    return 0


def _cmd_models_deploy(args: argparse.Namespace) -> int:
    endpoint = model_ops.deploy(models.get(args.name))
    print(f"deployed {endpoint.model.name} -> {endpoint.url}")
    return 0


def _cmd_models_scale(args: argparse.Namespace) -> int:
    settings: dict[str, int] = {}
    if args.min is not None:
        settings["min_containers"] = args.min
    if args.max is not None:
        settings["max_containers"] = args.max
    if args.buffer is not None:
        settings["buffer_containers"] = args.buffer
    if not settings:
        raise AomError("scale needs at least one of --min / --max / --buffer.")
    model_ops.scale(models.get(args.name), **settings)
    print(f"scaled {args.name}: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
    return 0


def _cmd_models_stop(args: argparse.Namespace) -> int:
    model_ops.stop(models.get(args.name))
    print(f"stopped {args.name}")
    return 0


def _cmd_models_status(args: argparse.Namespace) -> int:
    print(json.dumps(model_ops.status(models.get(args.name)), indent=2, default=str))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    endpoint = _resolve_endpoint(args)
    box = _create_sandbox(args, endpoint)
    run = box.start(
        args.harness, endpoint, args.prompt,
        goal=args.goal, timeout=args.timeout, effort=args.effort,
    )
    print(f"run {run.run_id} in {args.sandbox}:{box.id}")
    return _stream_run(run)


def _cmd_shell(args: argparse.Namespace) -> int:
    endpoint = _resolve_endpoint(args)
    box = _create_sandbox(args, endpoint)
    return box.interactive(args.harness, endpoint, effort=args.effort)


def _cmd_swarm(args: argparse.Namespace) -> int:
    result = swarm.Swarm.launch(
        args.harness, args.model, args.prompt, count=args.count,
        goal=args.goal, timeout=args.timeout, proxy=args.proxy, effort=args.effort,
    )
    print(f"swarm {result.id}: launched {len(result.agents())} agents on model {result.model}")
    print("Open the dashboard:")
    print(f"  AOM_SWARM={result.id} python examples/swarm/server.py")
    print("  then browse http://127.0.0.1:8787")
    return 0


# --------------------------------------------------------------------------------------------- #
# argument parsing                                                                              #
# --------------------------------------------------------------------------------------------- #
def _add_model_run_args(parser: argparse.ArgumentParser, *, workdir: bool = True) -> None:
    parser.add_argument("--sandbox", choices=SANDBOX_KINDS, default="local",
                        help="where the harness runs (default: local)")
    parser.add_argument("--harness", choices=HARNESSES, required=True)
    parser.add_argument("--model", required=True, help="catalog model name")
    parser.add_argument("--proxy", help="upstream proxy URL (socks5h/http, {session} supported)")
    parser.add_argument("--effort", help="reasoning effort (none..max)")
    parser.add_argument("--deploy", action="store_true",
                        help="deploy + prepare the model first (default: attach to an existing one)")
    if workdir:
        parser.add_argument("--workdir", help="working directory for the agent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aom", description="Run agent CLIs in sandboxes against "
                                     "open-weight models served on Modal.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="authenticate to Modal and store a proxy token")
    setup.add_argument("--name", default="aom", help="name for the minted proxy token")
    setup.set_defaults(func=_cmd_setup)

    mp = sub.add_parser("models", help="manage model deployments")
    msub = mp.add_subparsers(dest="models_command", required=True)
    msub.add_parser("list", help="list catalog models").set_defaults(func=_cmd_models_list)
    deploy = msub.add_parser("deploy", help="deploy a model (idempotent)")
    deploy.add_argument("name")
    deploy.set_defaults(func=_cmd_models_deploy)
    scale = msub.add_parser("scale", help="change a model's autoscaler live")
    scale.add_argument("name")
    scale.add_argument("--min", type=int, help="min_containers")
    scale.add_argument("--max", type=int, help="max_containers")
    scale.add_argument("--buffer", type=int, help="buffer_containers")
    scale.set_defaults(func=_cmd_models_scale)
    stop = msub.add_parser("stop", help="stop a model's app")
    stop.add_argument("name")
    stop.set_defaults(func=_cmd_models_stop)
    status = msub.add_parser("status", help="show a model's deployment status")
    status.add_argument("name")
    status.set_defaults(func=_cmd_models_status)

    run = sub.add_parser("run", help="one headless run, streamed to stdout")
    _add_model_run_args(run)
    run.add_argument("--goal", action="store_true", help="treat the prompt as a long-horizon goal")
    run.add_argument("--timeout", type=float, help="deadline in seconds")
    run.add_argument("prompt")
    run.set_defaults(func=_cmd_run)

    shell = sub.add_parser("shell", help="an interactive harness TUI on this terminal")
    _add_model_run_args(shell)
    shell.set_defaults(func=_cmd_shell)

    swarm_cmd = sub.add_parser("swarm", help="fan out N Modal agents and open a dashboard")
    swarm_cmd.add_argument("--harness", choices=HARNESSES, required=True)
    swarm_cmd.add_argument("--model", required=True)
    swarm_cmd.add_argument("--count", type=int, required=True)
    swarm_cmd.add_argument("--goal", action="store_true")
    swarm_cmd.add_argument("--timeout", type=float)
    swarm_cmd.add_argument("--proxy")
    swarm_cmd.add_argument("--effort")
    swarm_cmd.add_argument("prompt")
    swarm_cmd.set_defaults(func=_cmd_swarm)

    return parser


def _configure_stdio() -> None:
    """On Windows the console is often cp1252; subprocess output (modal's ✓/progress UI, agent
    traces) contains UTF-8. Reconfigure so streaming it never raises UnicodeEncodeError."""

    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AomError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130
