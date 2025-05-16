<#
.SYNOPSIS
  Lanza en 5 ventanas de PowerShell las cinco tareas:
   1. Modelo 1 API
   2. Modelo 2 API
   3. Router API
   4. LangChain Agent API
   5. Servidor estático para el frontend

.PARAMETER VenvFolder
  Carpeta de tu virtualenv (por defecto "venv").
#>

param(
    [string]$VenvFolder = "venv"
)

# Carpeta raíz del proyecto (donde está este script)
$root      = $PSScriptRoot
# Ruta al Activate.ps1 de la venv
$activate  = Join-Path $root "$VenvFolder\Scripts\Activate.ps1"

function Start-Api {
    param(
        [string]$ModulePath,  # ej. api.model1_api:app
        [int]   $Port,
        [string]$Title       # descripción para el log
    )

    # Comando que ejecutaremos en la nueva ventana
    $cmd = @"
Set-Location -LiteralPath '$root'
if (Test-Path -LiteralPath '$activate') { & '$activate' }
# Usamos python -m uvicorn para asegurar que usa el intérprete de la venv
& python -m uvicorn $ModulePath --host 0.0.0.0 --port $Port --reload
"@

    Start-Process powershell `
        -WindowStyle Minimized `
        -ArgumentList "-NoExit", "-NoProfile", "-Command", $cmd

    Write-Host "▶️ $Title arrancada (minimizada) en http://localhost:$Port"
}

# ---------------------------------------------------
# 1–4: APIs
# ---------------------------------------------------
Start-Api -ModulePath "api.model1_api:app"       -Port 8000 -Title "Modelo 1 API"
Start-Api -ModulePath "api.model2_api:app"       -Port 8001 -Title "Modelo 2 API"
Start-Api -ModulePath "api.router_api:app"       -Port 8002 -Title "Router API"
Start-Api -ModulePath "api.langchain_agent:app"  -Port 8003 -Title "LangChain Agent API"

# ---------------------------------------------------
# 5: Servidor estático para el frontend
# ---------------------------------------------------
$cmdStatic = @"
Set-Location -LiteralPath '$root\frontend'
if (Test-Path -LiteralPath '$activate') { & '$activate' }
# Sirve frontend/index.html en http://localhost:8080
& python -m http.server 8080 --directory .
"@

Start-Process powershell `
    -WindowStyle Minimized `
    -ArgumentList "-NoExit", "-NoProfile", "-Command", $cmdStatic

Write-Host "▶️ Frontend arrancado (minimizado) en http://localhost:8080"
