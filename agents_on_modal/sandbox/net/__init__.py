"""Sandbox networking shared by the Docker egress broker and the Modal loopback relay.

Public API other sandbox backends import::

    from agents_on_modal.sandbox.net import parse_proxy, UpstreamProxy, relay

    proxy = parse_proxy(url, session=sandbox_id)       # UpstreamProxy | None
    env = relay.env(proxy, endpoint_host)              # *_PROXY + NO_PROXY for the sandbox
    # python -m agents_on_modal.sandbox.net.relay       (Modal: loopback relay, residential egress)
    # python -m agents_on_modal.sandbox.net.egress      (Docker: egress broker + model proxy)

Model inference always stays direct; only code+browser traffic may be routed through the upstream
residential proxy.
"""

from __future__ import annotations

from .policy import Policy, PolicyError, address_is_public
from .proxy import ProxyError, UpstreamProxy, no_proxy_value, parse_proxy

__all__ = [
    "Policy",
    "PolicyError",
    "ProxyError",
    "UpstreamProxy",
    "address_is_public",
    "no_proxy_value",
    "parse_proxy",
]
