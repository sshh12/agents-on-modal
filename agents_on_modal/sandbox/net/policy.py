"""Destination policy for the Docker egress broker.

The agent container has no route to the internet except through the egress broker. This policy
decides what the broker is allowed to reach: only globally-routable public addresses, on an
allow-list of ports. Private, loopback, link-local, multicast, reserved, carrier-grade-NAT
(100.64.0.0/10) and cloud-metadata addresses are refused, so the agent can reach the public
internet but never the host, the LAN, the Docker networks, or the instance metadata service.

Hostnames are resolved here and the connection is made to the validated address, so a DNS
rebinding answer cannot smuggle a private IP past the check.

This file ships into Linux containers; keep it LF + POSIX.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket

from ...config import AomError

#: Default egress ports (HTTP and HTTPS).
DEFAULT_PORTS = frozenset({80, 443})

#: Cloud-metadata endpoints that must never be reachable even though they look "public".
_METADATA_ADDRESSES = frozenset({"169.254.169.254", "fd00:ec2::254"})


class PolicyError(AomError):
    """A destination was refused by the egress policy (or the policy was misconfigured)."""


def parse_ports(raw: str) -> frozenset[int]:
    ports: set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            port = int(piece)
        except ValueError as error:
            raise PolicyError(f"Invalid egress port {piece!r}.") from error
        if not 1 <= port <= 65535:
            raise PolicyError(f"Invalid egress port {piece!r}.")
        ports.add(port)
    return frozenset(ports)


def parse_hosts(raw: str) -> tuple[str, ...]:
    return tuple(piece.strip().lower() for piece in raw.split(",") if piece.strip())


def address_is_public(address: str) -> bool:
    """True only for a globally-routable public unicast address."""

    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if str(ip) in _METADATA_ADDRESSES:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    # IPv6 transition addresses carry an embedded IPv4 address; reject if that is private.
    for tunnelled in (getattr(ip, "sixtofour", None), getattr(ip, "teredo", None)):
        if tunnelled is not None:
            candidates = tunnelled if isinstance(tunnelled, tuple) else (tunnelled,)
            if any(not address_is_public(str(part)) for part in candidates):
                return False
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    if getattr(ip, "is_site_local", False):
        return False
    if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
        return False
    return True


class Policy:
    """Destination policy: allowed ports, an optional host allow-list, and public-IP enforcement."""

    def __init__(
        self,
        allowed_ports: frozenset[int] = DEFAULT_PORTS,
        allowed_hosts: tuple[str, ...] = (),
    ) -> None:
        self.allowed_ports = allowed_ports
        self.allowed_hosts = allowed_hosts

    def host_allowed(self, host: str) -> bool:
        if not self.allowed_hosts:
            return True
        host = host.lower().rstrip(".")
        for pattern in self.allowed_hosts:
            if pattern.startswith("."):
                if host == pattern[1:] or host.endswith(pattern):
                    return True
            elif host == pattern:
                return True
        return False

    async def resolve(self, host: str, port: int) -> tuple[str, int]:
        """Resolve ``host`` to one validated public address, or raise :class:`PolicyError`.

        The caller must connect to the returned address (not re-resolve ``host``) so a rebinding
        answer cannot swap in a private IP after the check.
        """

        if port not in self.allowed_ports:
            raise PolicyError(f"port {port} is not in the sandbox egress allow-list")
        if not self.host_allowed(host):
            raise PolicyError(f"host {host!r} is not in the sandbox egress allow-list")
        literal = host.strip("[]")
        try:
            ipaddress.ip_address(literal)
        except ValueError:
            pass
        else:
            if not address_is_public(literal):
                raise PolicyError(f"address {literal} is not a public internet address")
            return literal, port
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as error:
            raise PolicyError(f"could not resolve {host!r}: {error}") from error
        for info in infos:
            address = str(info[4][0])
            if address_is_public(address):
                return address, port
        raise PolicyError(f"{host!r} only resolves to non-public addresses")
