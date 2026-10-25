"""parse_proxy parsing/session-substitution/bad-input, and the relay env helper."""

from __future__ import annotations

import pytest

from agents_on_modal.sandbox.net import parse_proxy, relay
from agents_on_modal.sandbox.net.proxy import ProxyError, UpstreamProxy, no_proxy_value


def test_none_and_empty_return_none():
    assert parse_proxy(None) is None
    assert parse_proxy("") is None
    assert parse_proxy("   ") is None


@pytest.mark.parametrize("scheme", ["socks5", "socks5h", "http"])
def test_schemes_with_auth(scheme):
    p = parse_proxy(f"{scheme}://user:pass@proxy.example.com:1234")
    assert p == UpstreamProxy(scheme, "proxy.example.com", 1234, "user", "pass")


def test_no_auth():
    p = parse_proxy("http://proxy.example.com:8080")
    assert p.username is None and p.password is None
    assert (p.host, p.port) == ("proxy.example.com", 8080)


def test_default_ports():
    assert parse_proxy("socks5://h").port == 1080
    assert parse_proxy("socks5h://h").port == 1080
    assert parse_proxy("http://h").port == 8080


def test_https_normalises_to_http():
    assert parse_proxy("https://h:443").scheme == "http"


def test_session_substitution_in_username():
    p = parse_proxy("socks5h://cust-{session}:secret@gw:1080", session="abc123")
    assert p.username == "cust-abc123"
    assert p.password == "secret"


def test_session_substitution_in_password_too():
    p = parse_proxy("http://user:tok-{session}@gw:8080", session="xyz")
    assert p.password == "tok-xyz"


def test_no_session_leaves_placeholder():
    p = parse_proxy("socks5h://cust-{session}@gw:1080")
    assert p.username == "cust-{session}"


def test_percent_decoding():
    p = parse_proxy("http://user:p%40ss@gw:8080")
    assert p.password == "p@ss"


@pytest.mark.parametrize("bad", ["ftp://h:21", "not-a-url", "socks4://h:1080", "http://", "socks5://h:notaport"])
def test_bad_inputs_raise(bad):
    with pytest.raises(ProxyError):
        parse_proxy(bad)


def test_url_round_trip():
    assert parse_proxy("socks5h://u:p@gw:1080").url == "socks5h://u:p@gw:1080"


def test_no_proxy_value_includes_model_host_first():
    value = no_proxy_value("ws--aom-x-server.us-east.modal.direct")
    parts = value.split(",")
    assert parts[0] == "ws--aom-x-server.us-east.modal.direct"
    assert "localhost" in parts and "127.0.0.1" in parts and "::1" in parts


def test_no_proxy_value_without_model_host():
    assert no_proxy_value(None) == "localhost,127.0.0.1,::1"


def test_upstream_env_points_at_relay_and_keeps_model_direct():
    p = parse_proxy("socks5h://u:p@gw:1080", session="s")
    env = p.env("model.example.com")
    assert env["HTTP_PROXY"] == "http://127.0.0.1:3128"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:3128"
    assert env["ALL_PROXY"] == "socks5://127.0.0.1:1080"
    assert env["http_proxy"] == env["HTTP_PROXY"]
    assert "model.example.com" in env["NO_PROXY"].split(",")
    assert "model.example.com" in env["no_proxy"].split(",")
    assert "NODE_EXTRA_CA_CERTS" not in env


def test_upstream_env_with_ca_cert():
    p = parse_proxy("http://gw:8080")
    env = p.env("m", ca_cert="/etc/ca.pem")
    assert env["NODE_EXTRA_CA_CERTS"] == "/etc/ca.pem"
    assert env["REQUESTS_CA_BUNDLE"] == "/etc/ca.pem"


def test_relay_env_none_is_empty():
    assert relay.env(None, "model.example.com") == {}


def test_relay_env_with_proxy_matches_upstream_env():
    p = parse_proxy("http://gw:8080")
    assert relay.env(p, "model.example.com") == p.env("model.example.com")
    assert "model.example.com" in relay.env(p, "model.example.com")["NO_PROXY"]
