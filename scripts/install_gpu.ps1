# Optional CUDA install (onnxruntime-gpu) in .venv-gpu.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
uv venv .venv-gpu --python 3.11
$py = Join-Path $root ".venv-gpu\Scripts\python.exe"
uv pip install --python $py -r requirements.txt
uv pip install --python $py "onnxruntime-gpu[cuda,cudnn]"
