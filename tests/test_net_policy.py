"""Destination policy: public-IP enforcement with monkeypatched DNS (no real network)."""

from __future__ import annotations

import socket

import pytest

from agents_on_modal.sandbox.net.policy import (
    DEFAULT_PORTS,
    Policy,
    PolicyError,
    address_is_public,
    parse_hosts,
    parse_ports,
)

PRIVATE = [
    "10.0.0.5",
    "172.16.0.1",
    "192.168.1.1",
    "127.0.0.1",
    "169.254.169.254",  # metadata / link-local
    "169.254.1.1",
    "100.64.0.1",  # CGNAT
    "100.127.255.255",
    "0.0.0.0",
    "224.0.0.1",  # multicast
    "::1",
    "fe80::1",  # link-local v6
    "fc00::1",  # unique-local v6
    "fd00:ec2::254",  # v6 metadata
]
PUBLIC = ["8.8.8.8", "1.1.1.1", "140.82.112.3", "2606:4700:4700::1111"]


@pytest.mark.parametrize("addr", PRIVATE)
def test_private_addresses_refused(addr):
    assert address_is_public(addr) is False


@pytest.mark.parametrize("addr", PUBLIC)
def test_public_addresses_allowed(addr):
    assert address_is_public(addr) is True


def test_ipv4_mapped_private_refused():
    assert address_is_public("::ffff:10.0.0.1") is False


def test_parse_ports_and_hosts():
    assert parse_ports("80, 443, 8443") == frozenset({80, 443, 8443})
    assert parse_hosts("A.com, .B.com ,") == ("a.com", ".b.com")
    with pytest.raises(PolicyError):
        parse_ports("0")
    with pytest.raises(PolicyError):
        parse_ports("nope")


def _patch_dns(monkeypatch, address):
    def fake_getaddrinfo(host, port, *a, **k):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, port or 0))]

    # Policy.resolve calls loop.getaddrinfo; patching the socket function underneath it keeps the
    # default executor path but never touches the real resolver.
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


@pytest.mark.asyncio
async def test_resolve_public_hostname(monkeypatch):
    _patch_dns(monkeypatch, "8.8.8.8")
    policy = Policy(DEFAULT_PORTS)
    assert await policy.resolve("dns.google", 443) == ("8.8.8.8", 443)


@pytest.mark.asyncio
async def test_resolve_private_hostname_refused(monkeypatch):
    _patch_dns(monkeypatch, "10.1.2.3")
    policy = Policy(DEFAULT_PORTS)
    with pytest.raises(PolicyError):
        await policy.resolve("sneaky.internal", 443)


@pytest.mark.asyncio
async def test_resolve_rejects_disallowed_port(monkeypatch):
    _patch_dns(monkeypatch, "8.8.8.8")
    policy = Policy(DEFAULT_PORTS)
    with pytest.raises(PolicyError):
        await policy.resolve("dns.google", 22)


@pytest.mark.asyncio
async def test_resolve_ip_literal_public_bypasses_dns():
    policy = Policy(DEFAULT_PORTS)
    assert await policy.resolve("1.1.1.1", 80) == ("1.1.1.1", 80)


@pytest.mark.asyncio
async def test_resolve_ip_literal_private_refused():
    policy = Policy(DEFAULT_PORTS)
    with pytest.raises(PolicyError):
        await policy.resolve("192.168.0.1", 443)


@pytest.mark.asyncio
async def test_host_allowlist(monkeypatch):
    _patch_dns(monkeypatch, "8.8.8.8")
    policy = Policy(DEFAULT_PORTS, parse_hosts(".example.com"))
    assert await policy.resolve("api.example.com", 443) == ("8.8.8.8", 443)
    with pytest.raises(PolicyError):
        await policy.resolve("other.net", 443)
