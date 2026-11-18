# Isolated CPU install. Creates .venv with Python 3.11 via uv.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
uv venv .venv --python 3.11
uv pip install --python (Join-Path $root ".venv\Scripts\python.exe") -r requirements.txt
