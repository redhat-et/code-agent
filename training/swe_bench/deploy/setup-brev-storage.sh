#!/bin/bash
#
# Relocate Docker and microk8s containerd data to /ephemeral.
#
# Brev GPU instances have a small root disk (~100GB) and a large
# ephemeral disk (~750GB) mounted at /ephemeral.  Container images
# (40-50GB each) must live on the ephemeral disk.
#
# This script moves the containerd data directories to /ephemeral
# and creates symlinks so Docker and microk8s find them transparently.
#
# Run ONCE on a fresh instance, BEFORE building container images.
#
# Usage (must run as root):
#   sudo bash training/swe_bench/deploy/setup-brev-storage.sh

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must run as root (sudo)" >&2
    exit 1
fi

EPHEMERAL="/ephemeral"

if [ ! -d "$EPHEMERAL" ]; then
    echo "ERROR: $EPHEMERAL not found — is this a Brev GPU instance?" >&2
    exit 1
fi

relocate() {
    local src="$1"
    local dst="$2"
    local label="$3"

    if [ -L "$src" ]; then
        echo "$label: $src is already a symlink, skipping."
        return
    fi

    echo "$label: moving $src → $dst ..."
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    ln -s "$dst" "$src"
    echo "$label: done."
}

echo "=== Stopping Docker ==="
systemctl stop docker docker.socket containerd 2>/dev/null || true

echo "=== Stopping microk8s ==="
microk8s stop 2>/dev/null || snap stop microk8s 2>/dev/null || true

# Docker's containerd (stores images built with `docker build`)
relocate /var/lib/containerd "$EPHEMERAL/containerd" "Docker containerd"

# Docker's own metadata
relocate /var/lib/docker "$EPHEMERAL/docker" "Docker metadata"

# microk8s containerd (stores images pulled by K8s pods)
relocate /var/snap/microk8s/common/var/lib/containerd \
         "$EPHEMERAL/microk8s-containerd" \
         "microk8s containerd"

echo "=== Starting Docker ==="
systemctl start containerd docker

echo "=== Starting microk8s ==="
microk8s start 2>/dev/null || snap start microk8s 2>/dev/null || true

echo "=== Waiting for microk8s to be ready ==="
microk8s status --wait-ready --timeout 120 2>/dev/null || \
    kubectl wait --for=condition=Ready node --all --timeout=120s 2>/dev/null || true

echo ""
echo "Storage relocated. Disk usage:"
df -h / "$EPHEMERAL"
