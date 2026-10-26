"""In-sandbox bridge from the DeepSeek Harness Python SDK to the runner's newline-JSON contract.

The runner (:mod:`agents_on_modal.runner`) launches a harness as a subprocess and reads one JSON
object per stdout line, which :meth:`agents_on_modal.harness.dsh.DeepSeek.parse` turns into events.
The DeepSeek Harness ships as a Python SDK that drives the ``dsh`` process over JSON-RPC and exposes
results through Python objects and an ``on_notification`` callback -- not a stdout JSON stream. This
module is the only place that imports ``deepseek_harness``: it constructs the SDK, re-emits each
notification as a line in the stable schema ``dsh.py`` owns, and prints a final ``turn_end`` from the
``RunResult``. Keeping all SDK contact here lets the harness class stay pure and offline-testable,
and lets this adapter be defensive about an SDK whose exact notification shape is not fully
documented.

Invocation (from ``DeepSeek.headless_argv``)::

    python3 -m agents_on_modal.harness._dsh_adapter --message <text> [--resume <sid>] [--goal]

Everything else is read from the environment ``DeepSeek.prepare`` set (``AOM_DSH_*``, ``DSH_HOME``).
The adapter owns prompt composition: on the first turn (no ``--resume``) it prepends the sandbox
instructions file and, in ``--goal`` mode, appends the goal preamble; a resume carries the restored
session and so sends the continuation message verbatim.

Resilience choices (the SDK's notification schema is under-documented, and this runs on an expensive
warm model where one failed launch is costly):

* Kwargs are filtered to those the installed ``DeepSeekHarness.__init__`` actually accepts, so an SDK
  revision that renames or drops a parameter does not abort the launch.
* ``reasoning_effort`` is NOT sent by default -- the gateway already applies the catalog's default
  effort, and sending a route-specific effort string risks a validation error. Set
  ``AOM_DSH_SEND_EFFORT=1`` to forward it.
* Every notification is normalized best-effort; with ``AOM_DSH_DEBUG=1`` the raw (plain-serialized)
  notification is also emitted as a ``raw`` line (ignored by ``parse``) so a live run can teach us
  the exact wire shape without a second deploy.
* An unrecognized notification still emits a compact ``log`` line, so the live trace is never blank.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import traceback
import uuid
from typing import Any

_PRINT_LOCK = threading.Lock()
_DEBUG = os.environ.get("AOM_DSH_DEBUG") == "1"


# --------------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------------
def emit(obj: dict[str, Any]) -> None:
    """Write one normalized line to stdout (atomic under the GIL + an explicit lock)."""

    try:
        line = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - never let serialization kill the run
        line = json.dumps({"type": "log", "text": f"<unserializable {obj.get('type')!r}>"})
    with _PRINT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(text: str) -> None:
    emit({"type": "log", "text": text})


# --------------------------------------------------------------------------------------------------
# robust serialization of arbitrary SDK objects
# --------------------------------------------------------------------------------------------------
def to_plain(obj: Any, depth: int = 0) -> Any:
    """Best-effort conversion of an SDK notification/event into JSON-safe primitives."""

    if depth > 8:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_plain(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_plain(v, depth + 1) for k, v in obj.items()}
    for attr in ("model_dump", "to_dict", "dict", "_asdict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return to_plain(fn(), depth + 1)
            except Exception:  # noqa: BLE001
                pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return to_plain(dataclasses.asdict(obj), depth + 1)
        except Exception:  # noqa: BLE001
            pass
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and d:
        return {str(k): to_plain(v, depth + 1) for k, v in d.items() if not str(k).startswith("_")}
    return str(obj)


def _first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


# --------------------------------------------------------------------------------------------------
# notification -> normalized lines
# --------------------------------------------------------------------------------------------------
#: ``payload.event.type`` values that are routing noise, not model actions (verified against a live
#: dsh session). The committed assistant message, tool calls and tool results carry the content.
_NOISE_EVENTS = (
    "turn/start", "turn/end", "step/start", "step/end", "request/", "response/", "session/title",
    "session/status", "agent/inbox", "user/message", "system/message", "assistant/attempt",
    "llm/retry", "llm/request", "llm/response", "context/", "compaction/start",
)


def normalize(notif: Any) -> list[dict[str, Any]]:
    """Map one dsh notification onto zero or more normalized lines (dsh.py's schema).

    Verified wire shape (deepseek-harness-sdk 0.2.1rc2)::

        {"method": "session.event",
         "payload": {"sessionId": "...", "event": {"type": "<etype>", "seq": N, "data": {...}}}}
        {"method": "session.status", "payload": {"sessionId": "...", "status": "running"}}

    The real action is nested at ``payload.event``; :func:`_normalize_event` maps its ``type`` +
    ``data``. Field names inside ``data`` are looked up under several spellings (camelCase and
    snake_case) and anything unrecognized becomes a compact ``log`` line, so the trace is never
    blank. ``AOM_DSH_DEBUG=1`` additionally dumps the raw object, which is how the exact event-type
    strings were pinned down.
    """

    d = to_plain(notif)
    if not isinstance(d, dict):
        return [{"type": "log", "text": f"[dsh] {str(d)[:200]}"}]

    method = str(_first(d, "method") or "").lower()
    payload = _first(d, "payload", "params", "data") or {}
    if not isinstance(payload, dict):
        payload = {}

    event = payload.get("event")
    if isinstance(event, dict) and ("event" in method or method.endswith(".event") or "type" in event):
        return _normalize_event(event, payload)
    if "status" in method:
        status = payload.get("status")
        return [{"type": "log", "text": f"[dsh] status:{status}"}] if status else []
    return [{"type": "log", "text": f"[dsh] {method or 'notification'}"}]


def _normalize_event(event: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    etype = str(event.get("type") or "").lower()
    data = event.get("data")
    if not isinstance(data, dict):
        data = {"value": data}
    agent = _first(data, "agentId", "agent_id", "agent") or _first(payload, "agentId", "agent_id")

    def tag(obj: dict[str, Any]) -> dict[str, Any]:
        if agent:
            obj.setdefault("agent", str(agent))
        return obj

    # committed assistant message (final text + any tool-call blocks it carries)
    if "assistant" in etype and "message" in etype:
        return [tag(e) for e in _blocks_to_events(_message_content(data))]
    # explicit tool-call / tool-result events
    if "tool" in etype and any(k in etype for k in ("call", "use", "invoc", "exec")):
        if any(k in etype for k in ("result", "output", "return", "end", "complete")):
            return [tag(_tool_result(data))]
        return [tag(_tool_call(data))]
    if "tool" in etype and any(k in etype for k in ("result", "output", "return")):
        return [tag(_tool_result(data))]
    if "reason" in etype or "think" in etype:
        txt = _as_text(_first(data, "text", "content", "delta"))
        return [tag({"type": "thinking", "text": txt})] if txt else []
    if "usage" in etype:
        u = data.get("usage") if isinstance(data.get("usage"), dict) else data
        return [tag({"type": "usage",
                     "input_tokens": int(_first(u, "input_tokens", "inputTokens", "prompt_tokens") or 0),
                     "output_tokens": int(_first(u, "output_tokens", "outputTokens", "completion_tokens") or 0)})]
    if ("agent" in etype or "subagent" in etype) and any(k in etype for k in
                                                         ("spawn", "child", "start", "end", "status")):
        return [tag({"type": "subagent",
                     "id": _first(data, "agentId", "agent_id", "id"),
                     "status": _subagent_status(_first(data, "status", "state")),
                     "description": _first(data, "prompt", "goal", "description")})]
    if any(n in etype for n in _NOISE_EVENTS):  # turn/end is authoritative from RunResult
        return []
    return [{"type": "log", "text": f"[dsh] {etype}"}]


def _tool_call(data: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call",
            "id": _first(data, "id", "callId", "toolCallId", "call_id", "tool_id"),
            "name": _first(data, "name", "tool", "toolName", "tool_name"),
            "input": _first(data, "input", "arguments", "args", "parameters") or {}}


def _tool_result(data: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_result",
            "id": _first(data, "id", "callId", "toolCallId", "call_id", "tool_id"),
            "output": _first(data, "output", "result", "content", "stdout", "text"),
            "is_error": _is_error(data)}


def _is_error(data: dict[str, Any]) -> bool:
    if _first(data, "is_error", "isError", "error", "failed"):
        return True
    return str(_first(data, "status") or "").lower() in ("error", "failed")


def _message_content(data: dict[str, Any]) -> Any:
    """The assistant message's content, whether under ``data.message.content`` or ``data.content``."""

    msg = data.get("message")
    if isinstance(msg, dict):
        return msg.get("content")
    return _first(data, "content", "text")


def _blocks_to_events(content: Any) -> list[dict[str, Any]]:
    """Flatten a message's content (string or a list of typed blocks) into normalized lines."""

    out: list[dict[str, Any]] = []
    if content is None:
        return out
    if isinstance(content, str):
        if content.strip():
            out.append({"type": "text", "text": content})
        return out
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                text = str(block)
                if text.strip():
                    out.append({"type": "text", "text": text})
                continue
            bt = str(block.get("type") or "").lower()
            if "tool" in bt and any(k in bt for k in ("call", "use", "invoc")):
                out.append(_tool_call(block))
            elif "think" in bt or "reason" in bt:
                txt = _as_text(_first(block, "text", "thinking", "content"))
                if txt:
                    out.append({"type": "thinking", "text": txt})
            else:  # text / content / untyped
                txt = _as_text(_first(block, "text", "content"))
                if txt:
                    out.append({"type": "text", "text": txt})
    return out


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, dict):
                parts.append(str(_first(block, "text", "content", "value") or ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    if isinstance(value, dict):
        return str(_first(value, "text", "content", "value") or value)
    return str(value)


def _subagent_status(status: Any) -> str:
    text = str(status or "").lower()
    if text in ("completed", "complete", "done", "success", "succeeded"):
        return "completed"
    if text in ("failed", "error", "cancelled", "canceled"):
        return "failed"
    return "started"


# --------------------------------------------------------------------------------------------------
# prompt composition
# --------------------------------------------------------------------------------------------------
def compose_prompt(message: str, *, resume: str | None, goal: bool) -> str:
    """First turn: prepend the sandbox instructions file and, in goal mode, append the sentinel
    preamble. A resume carries the restored session, so its continuation message is sent as-is."""

    if resume:
        return message
    parts: list[str] = []
    path = os.environ.get("AOM_DSH_INSTRUCTIONS_FILE")
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                parts.append(fh.read().strip())
        except OSError:
            pass
    parts.append(message)
    if goal:
        # Imported lazily so the adapter has no hard dependency on the harness module at import.
        from .dsh import _GOAL_PREAMBLE

        parts.append(_GOAL_PREAMBLE)
    return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
def _build_kwargs() -> dict[str, Any]:
    """Candidate constructor kwargs from the environment; filtered to supported names in ``main``."""

    kwargs: dict[str, Any] = {
        "provider": os.environ.get("AOM_DSH_PROVIDER") or "deepseek-official",
        "model": os.environ.get("AOM_DSH_MODEL"),
        "dsh_home": os.environ.get("DSH_HOME"),
        "cwd": os.environ.get("AOM_DSH_WORKDIR") or os.getcwd(),
        "profile": os.environ.get("AOM_DSH_PROFILE") or "sdk-minimal",
        "base_url": os.environ.get("AOM_DSH_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL"),
        "api_key": os.environ.get("AOM_DSH_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
    }
    max_tokens = os.environ.get("AOM_DSH_MAX_TOKENS")
    if max_tokens and max_tokens.isdigit() and int(max_tokens) > 0:
        kwargs["max_tokens"] = int(max_tokens)
    request_timeout = os.environ.get("AOM_DSH_REQUEST_TIMEOUT")
    if request_timeout:
        try:
            kwargs["request_timeout_seconds"] = float(request_timeout)
        except ValueError:
            pass
    effort = os.environ.get("AOM_DSH_EFFORT")
    if effort and os.environ.get("AOM_DSH_SEND_EFFORT") == "1":
        kwargs["reasoning_effort"] = effort
    return {k: v for k, v in kwargs.items() if v is not None}


def _supported(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the target accepts (pass everything through if it declares ``**kwargs``)."""

    import inspect

    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


#: Documented ``DeepSeekHarnessConfig`` fields, used only if the class cannot be introspected.
_CONFIG_FIELDS_FALLBACK = frozenset({
    "provider", "model", "reasoning_effort", "max_tokens", "cwd", "runtime_cwd", "dsh_bin",
    "profile", "patches", "dsh_home", "env", "initialize_timeout_seconds",
    "request_timeout_seconds", "shutdown_timeout_seconds", "base_url", "api_key",
})


def _config_field_names(deepseek_harness: Any) -> set[str]:
    """The field names ``DeepSeekHarnessConfig`` accepts (dataclass fields, else its signature)."""

    cfg = getattr(deepseek_harness, "DeepSeekHarnessConfig", None)
    fields = getattr(cfg, "__dataclass_fields__", None)
    if isinstance(fields, dict):
        return set(fields)
    if cfg is not None:
        import inspect

        try:
            return set(inspect.signature(cfg.__init__).parameters) - {"self"}
        except (TypeError, ValueError):
            pass
    return set(_CONFIG_FIELDS_FALLBACK)


def _filter_ctor_kwargs(deepseek_harness: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the config dataclass accepts.

    ``DeepSeekHarness.__init__`` declares ``**kwargs`` but forwards them to a *strict*
    ``DeepSeekHarnessConfig`` dataclass, so an unknown key (a callback the config does not model,
    say) raises ``TypeError`` and aborts the launch. ``on_notification`` and ``session_id`` are
    run() parameters, never config fields, so they are dropped here by construction.
    """

    names = _config_field_names(deepseek_harness)
    return {k: v for k, v in kwargs.items() if k in names}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek Harness SDK adapter (emits normalized JSON).")
    parser.add_argument("--message", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--goal", action="store_true")
    args = parser.parse_args(argv)

    session_id = args.resume or uuid.uuid4().hex
    emit({"type": "session", "session_id": session_id})
    prompt = compose_prompt(args.message, resume=args.resume, goal=args.goal)

    try:
        import deepseek_harness
    except Exception as exc:  # noqa: BLE001
        emit({"type": "turn_end", "is_error": True, "finish_reason": "error",
              "result": f"deepseek_harness import failed: {exc!r}"})
        return 1

    harness_cls = getattr(deepseek_harness, "DeepSeekHarness", None)
    if harness_cls is None:  # fall back to the first Harness-looking class in the module
        harness_cls = next((obj for name, obj in vars(deepseek_harness).items()
                            if isinstance(obj, type) and name.endswith("Harness")), None)
    if harness_cls is None:
        emit({"type": "turn_end", "is_error": True, "finish_reason": "error",
              "result": "no DeepSeekHarness class in deepseek_harness"})
        return 1

    def on_notification(notif: Any) -> None:
        try:
            if _DEBUG:
                emit({"type": "raw", "src": "notif", "n": to_plain(notif)})
            for line in normalize(notif):
                emit(line)
        except Exception as exc:  # noqa: BLE001 - a bad notification must not kill the run
            log(f"notif-handler-error: {exc!r}")

    # Auth bridge: dsh's provider sends a credential the Modal endpoint's edge rejects (401). Route
    # it through a loopback proxy that injects `Authorization: Bearer <token>`, which the edge
    # accepts. Disable with AOM_DSH_AUTH_SHIM=0 (e.g. a provider that already sends Bearer).
    proxy_server = None
    upstream = _gateway_root(os.environ.get("AOM_DSH_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL"))
    token = os.environ.get("AOM_DSH_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if os.environ.get("AOM_DSH_AUTH_SHIM", "1") != "0" and upstream and upstream.startswith("http") and token:
        try:
            from . import _dsh_auth_proxy

            proxy_base, proxy_server = _dsh_auth_proxy.start(upstream, token, verbose=_DEBUG)
            shim_v1 = proxy_base + "/v1"
            os.environ["AOM_DSH_BASE_URL"] = shim_v1
            os.environ["DEEPSEEK_BASE_URL"] = shim_v1
            log(f"auth-proxy: {shim_v1} -> {upstream}")
        except Exception as exc:  # noqa: BLE001 - fall back to a direct (likely 401) base_url
            log(f"auth-proxy failed to start: {exc!r}")

    ctor_kwargs = _filter_ctor_kwargs(deepseek_harness, _build_kwargs())

    finish_reason: str | None = None
    final_response: str | None = None
    is_error = False
    try:
        with harness_cls(**ctor_kwargs) as harness:
            # session_id and on_notification are run() parameters (verified against the SDK), never
            # constructor/config fields.
            run_kwargs = _supported(harness.run, {"session_id": session_id,
                                                  "on_notification": on_notification})
            result = harness.run(prompt, **run_kwargs)

        final_response = getattr(result, "final_response", None)
        finish_reason = getattr(result, "finish_reason", None)
        result_sid = getattr(result, "session_id", None)
        if result_sid and str(result_sid) != session_id:
            emit({"type": "session", "session_id": str(result_sid)})

        usage = _extract_usage(result)
        if usage:
            emit({"type": "usage", **usage})

        is_error = str(finish_reason or "").lower() == "error"
        if _DEBUG:
            for key in ("notifications", "events"):
                for item in (getattr(result, key, None) or []):
                    emit({"type": "raw", "src": f"result.{key}", "n": to_plain(item)})
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - report any SDK/runtime failure as a clean turn_end
        emit({"type": "turn_end", "is_error": True, "finish_reason": "error",
              "result": f"dsh run failed: {exc!r}"})
        if _DEBUG:
            log("traceback: " + traceback.format_exc())
        return 1
    finally:
        if proxy_server is not None:
            try:
                proxy_server.shutdown()
            except Exception:  # noqa: BLE001
                pass

    emit({"type": "turn_end", "is_error": is_error,
          "finish_reason": finish_reason or "completed", "result": final_response})
    return 1 if is_error else 0


def _gateway_root(base_url: str | None) -> str | None:
    """The gateway root for the auth proxy: ``AOM_DSH_BASE_URL`` with any trailing ``/v1`` removed."""

    if not base_url:
        return None
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    return root


def _extract_usage(result: Any) -> dict[str, int] | None:
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    u = to_plain(usage)
    if not isinstance(u, dict):
        return None
    inp = _first(u, "input_tokens", "prompt_tokens", "total_input_tokens")
    out = _first(u, "output_tokens", "completion_tokens", "total_output_tokens")
    if inp is None and out is None:
        return None
    return {"input_tokens": int(inp or 0), "output_tokens": int(out or 0)}


if __name__ == "__main__":
    raise SystemExit(main())
