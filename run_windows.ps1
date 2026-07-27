$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $venvStreamlit)) {
    Write-Host "Ambiente virtual nao encontrado. Rodando setup_windows.ps1..."
    & (Join-Path $PSScriptRoot "setup_windows.ps1")
}

if (-not (Test-Path $venvStreamlit)) {
    Write-Host "Streamlit nao foi encontrado na .venv apos o setup." -ForegroundColor Red
    exit 1
}

Write-Host "Iniciando DataVeritas..."
& $venvStreamlit run app.py
