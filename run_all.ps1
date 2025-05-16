<#
.SYNOPSIS
  Lanza en 3 ventanas de PowerShell las tres APIs (Modelo 1, Modelo 2 y Router) minimizadas.

.PARAMETER VenvFolder
  Carpeta de tu virtualenv (por defecto "venv").
#>

param(
    [string]$VenvFolder = "venv"
)

# Raíz del proyecto (carpeta donde está este script)
$root = $PSScriptRoot
# Ruta al Activate.ps1 de la venv
$activate = Join-Path $root "$VenvFolder\Scripts\Activate.ps1"

function Start-Api {
    param(
        [string]$ModulePath,  # ejemplo: api.model1_api:app
        [int]$Port,
        [string]$Title       # descripción para el log
    )
    # Script que se ejecutará en la nueva ventana
    $cmd = @"
Set-Location -LiteralPath '$root'
if (Test-Path -LiteralPath '$activate') { & '$activate' }
& uvicorn $ModulePath --host 0.0.0.0 --port $Port --reload
"@

    # Lanzar PowerShell minimizada
    Start-Process powershell `
        -WindowStyle Minimized `
        -ArgumentList "-NoExit", "-NoProfile", "-Command", $cmd

    Write-Host "▶️ $Title arrancada (minimizada) en http://localhost:$Port"
}

# Arrancamos los tres servicios minimizados
Start-Api -ModulePath "api.model1_api:app" -Port 8000 -Title "Modelo 1 API"
Start-Api -ModulePath "api.model2_api:app" -Port 8001 -Title "Modelo 2 API"
Start-Api -ModulePath "api.router_api:app" -Port 8002 -Title "Router API"
Start-Process uvicorn -ArgumentList "api.langchain_agent:app --port 8003 --reload" -NoNewWindow
