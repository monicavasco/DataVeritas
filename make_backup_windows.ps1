param(
    [switch]$NoVectorStore
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupRoot = Join-Path $PSScriptRoot "backups"
$stage = Join-Path $backupRoot "DataVeritas-portable-$stamp"
$zipPath = "$stage.zip"

New-Item -ItemType Directory -Path $stage -Force | Out-Null

$excludedRootNames = @(
    ".venv",
    ".git",
    ".cache",
    "__pycache__",
    "backups",
    "crewai-storage"
)
$excludedRootFiles = @(
    ".env"
)

foreach ($item in Get-ChildItem -Force $PSScriptRoot) {
    if ($excludedRootNames -contains $item.Name) {
        continue
    }
    if ($excludedRootFiles -contains $item.Name) {
        continue
    }
    Copy-Item -LiteralPath $item.FullName -Destination $stage -Recurse -Force
}

if ($NoVectorStore) {
    $vectorPath = Join-Path $stage "data\chroma"
    if (Test-Path $vectorPath) {
        $resolvedStage = (Resolve-Path $stage).Path
        $resolvedVector = (Resolve-Path $vectorPath).Path
        if (-not $resolvedVector.StartsWith($resolvedStage, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Caminho inseguro para remover: $resolvedVector"
        }
        Remove-Item -LiteralPath $resolvedVector -Recurse -Force
    }
}

$resolvedStageForCleanup = (Resolve-Path $stage).Path
Get-ChildItem -Path $stage -Recurse -Force -Directory -Filter "__pycache__" | ForEach-Object {
    if ($_.FullName.StartsWith($resolvedStageForCleanup, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }
}

Get-ChildItem -Path $stage -Recurse -Force -File -Include "*.pyc", ".env" | ForEach-Object {
    if ($_.FullName.StartsWith($resolvedStageForCleanup, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $_.FullName -Force
    }
}

$itemsToZip = Get-ChildItem -Force $stage | ForEach-Object { $_.FullName }
Compress-Archive -LiteralPath $itemsToZip -DestinationPath $zipPath -Force

Write-Host "Backup criado:" -ForegroundColor Green
Write-Host $zipPath
