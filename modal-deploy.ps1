#!/usr/bin/env pwsh
# Deploy a self-managed model from the JSON catalog:
#   ./modal-deploy.ps1 <index|name> [--gpu H100:8] [--dry-run]
#   ./modal-deploy.ps1 [catalog.json] <index|name> ...   # explicit catalog file
# Defaults the catalog to ./catalog.json when only a selector is given.
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir ".venv/Scripts/python.exe"
if (-not (Test-Path $venvPython)) { $venvPython = "python" }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
Set-Location $scriptDir
& $venvPython -m codex_modal.catalog_deploy @args
exit $LASTEXITCODE
