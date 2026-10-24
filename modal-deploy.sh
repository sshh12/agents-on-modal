#!/usr/bin/env sh
# Deploy a self-managed model from the JSON catalog:
#   ./modal-deploy.sh <index|name> [--gpu H100:8] [--dry-run]
#   ./modal-deploy.sh [catalog.json] <index|name> ...   # explicit catalog file
# Defaults the catalog to ./catalog.json when only a selector is given.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
venv_dir="$script_dir/.venv"

if [ -x "$venv_dir/bin/python" ]; then
    venv_python="$venv_dir/bin/python"
elif [ -f "$venv_dir/Scripts/python.exe" ]; then
    venv_python="$venv_dir/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    venv_python=python3
else
    venv_python=python
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
cd "$script_dir"
exec "$venv_python" -m codex_modal.catalog_deploy "$@"
