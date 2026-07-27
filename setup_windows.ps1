param(
    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-ProjectPython {
    $candidates = @(
        @{ Command = "py"; Args = @("-3.12") },
        @{ Command = "py"; Args = @("-3.11") },
        @{ Command = "py"; Args = @("-3") },
        @{ Command = "python"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.Command -ErrorAction SilentlyContinue
        if (-not $command) {
            continue
        }

        try {
            $versionCheck = @"
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
print(sys.executable)
"@
            $pythonPath = & $candidate.Command @($candidate.Args + @("-c", $versionCheck)) 2>$null
            if ($LASTEXITCODE -eq 0 -and $pythonPath) {
                return @{ Command = $candidate.Command; Args = $candidate.Args }
            }
        }
        catch {
            continue
        }
    }

    return $null
}

$python = Get-ProjectPython
if (-not $python) {
    Write-Host "Python 3.11+ nao encontrado." -ForegroundColor Red
    Write-Host "Instale Python 3.11 ou 3.12 e rode este script novamente."
    Write-Host "Se tiver winget: winget install Python.Python.3.12"
    exit 1
}

if ($Reinstall -and (Test-Path ".venv")) {
    Write-Host "Remova a pasta .venv manualmente antes de reinstalar do zero." -ForegroundColor Yellow
    Write-Host "Isso evita apagar um ambiente virtual por engano."
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "Criando ambiente virtual em .venv..."
    & $python.Command @($python.Args + @("-m", "venv", ".venv"))
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Nao foi possivel encontrar .venv\Scripts\python.exe." -ForegroundColor Red
    exit 1
}

Write-Host "Preparando pip..."
& $venvPython -m ensurepip --upgrade
& $venvPython -m pip install --upgrade pip

Write-Host "Instalando dependencias do requirements.txt..."
& $venvPython -m pip install -r requirements.txt

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Arquivo .env criado a partir de .env.example."
    Write-Host "Edite .env para colocar chaves como OPENAI_API_KEY ou GEMINI_API_KEY."
}

Write-Host ""
Write-Host "Setup concluido." -ForegroundColor Green
Write-Host "Para rodar: .\run_windows.ps1"
