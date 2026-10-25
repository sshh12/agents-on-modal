#!/bin/sh
# Container init for the agents-on-modal sandbox image.
#
# This is a thin, no-op-friendly entrypoint: it execs whatever command the
# backend asked for ("$@"). Its one optional job is the egress firewall.
#
#   AOM_INIT_FIREWALL=1  -> apply a default-deny iptables policy that permits
#                           only loopback and the egress broker, then drop to
#                           the unprivileged agent user and exec "$@".
#   (unset / anything else) -> just exec "$@" as the current user.
#
# The firewall path is used by the `docker` backend (which also puts the agent
# on a Docker --internal network; the firewall is a second, independent layer).
# On Modal, iptables is unavailable and the backend manages the agent user with
# gosu per command, so AOM_INIT_FIREWALL is left unset and this is a passthrough.
#
# Keep this POSIX sh. The egress server itself lives in the aom package, not
# here; init.sh only installs packet-filter rules from environment the backend
# supplies.
set -eu

RUN_USER="${AOM_RUN_USER:-agent}"

# --- no firewall requested: plain passthrough (Modal / local docker runs).
if [ "${AOM_INIT_FIREWALL:-}" != "1" ]; then
    exec "$@"
fi

# --- firewall requested. These describe the only reachable destination: the
# egress broker. DNS is resolved broker-side, so nothing else is allowed out.
EGRESS_IP="${AOM_EGRESS_IP:?AOM_EGRESS_IP is required when AOM_INIT_FIREWALL=1}"
PROXY_PORT="${AOM_EGRESS_PROXY_PORT:-3128}"
MODEL_PORT="${AOM_EGRESS_MODEL_PORT:-8081}"

# Pick an iptables backend that actually works in this network namespace.
pick_iptables() {
    for candidate in iptables-legacy iptables iptables-nft; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -L -n >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

apply_firewall() {
    ipt="$1"
    # Allow loopback (Docker's embedded resolver lives on 127.0.0.11) and the
    # broker; reject everything else. REJECT (not DROP) keeps blocked calls fast
    # instead of hanging.
    "$ipt" -P FORWARD DROP
    "$ipt" -F OUTPUT
    "$ipt" -F INPUT
    "$ipt" -A OUTPUT -o lo -j ACCEPT
    "$ipt" -A INPUT -i lo -j ACCEPT
    "$ipt" -A OUTPUT -d "$EGRESS_IP" -p tcp --dport "$PROXY_PORT" -j ACCEPT
    "$ipt" -A OUTPUT -d "$EGRESS_IP" -p tcp --dport "$MODEL_PORT" -j ACCEPT
    "$ipt" -A INPUT -s "$EGRESS_IP" -p tcp -j ACCEPT
    "$ipt" -A OUTPUT -p tcp -j REJECT --reject-with tcp-reset
    "$ipt" -A OUTPUT -j REJECT
    "$ipt" -A INPUT -j DROP
    "$ipt" -P OUTPUT DROP
    "$ipt" -P INPUT DROP
}

apply_firewall6() {
    ipt6="$1"
    "$ipt6" -F OUTPUT 2>/dev/null || true
    "$ipt6" -F INPUT 2>/dev/null || true
    "$ipt6" -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
    "$ipt6" -A INPUT -i lo -j ACCEPT 2>/dev/null || true
    "$ipt6" -P OUTPUT DROP 2>/dev/null || true
    "$ipt6" -P INPUT DROP 2>/dev/null || true
    "$ipt6" -P FORWARD DROP 2>/dev/null || true
}

ipt="$(pick_iptables)" || {
    echo "aom-init: no usable iptables backend (needs --cap-add NET_ADMIN)." >&2
    exit 70
}
apply_firewall "$ipt"
for candidate in ip6tables-legacy ip6tables ip6tables-nft; do
    if command -v "$candidate" >/dev/null 2>&1; then
        apply_firewall6 "$candidate"
        break
    fi
done
echo "aom-init: egress locked to ${EGRESS_IP}:${PROXY_PORT},${MODEL_PORT} via ${ipt}." >&2

# Route apt through the broker; sudo resets the environment, so bake it in.
cat > /etc/apt/apt.conf.d/01aom-proxy <<EOF
Acquire::http::Proxy "http://${EGRESS_IP}:${PROXY_PORT}";
Acquire::https::Proxy "http://${EGRESS_IP}:${PROXY_PORT}";
EOF

# Drop privileges and run the real command as the agent user.
if command -v gosu >/dev/null 2>&1; then
    exec gosu "$RUN_USER" "$@"
fi
exec "$@"
