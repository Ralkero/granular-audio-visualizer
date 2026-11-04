$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv or create a Python 3.11+ venv and run: python -m pip install -e ".[dev]"'
}
uv sync --locked --extra dev --python 3.14
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
