#!/usr/bin/env pwsh
# Optimum (NativeTune) installer.
#
#   irm https://optimumtune.github.io/install.ps1 | iex
#
# The canonical copy of this script is served from the optimumtune.github.io
# repo; this file is the source of truth it is copied from. Keep the two in
# sync when editing.
#
# What this does, in order:
#   1. Checks for Python 3.10+ (does not install it for you).
#   2. Creates a private venv at %LOCALAPPDATA%\nativetune-app and installs
#      the `optimum` package into it from GitHub.
#   3. Adds that venv's Scripts folder to your user PATH, so `optimum` works
#      in any new terminal.
#   4. Points the app at a per-user data folder (%LOCALAPPDATA%\nativetune)
#      via a persisted NATIVETUNE_HOME environment variable, and creates the
#      llama / models / cache / data subfolders it expects.
#
# This installs the CLI only. llama.cpp's binaries and your own .gguf model
# files are NOT downloaded by this script (see the README for why) — you
# still place those yourself under the data folder this script prints at
# the end.

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "!! $msg" -ForegroundColor Yellow }

# --- 1. find a usable Python -------------------------------------------------
$python = $null
foreach ($candidate in @("py", "python")) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    try {
        $verOut = & $candidate --version 2>&1
        if ($verOut -match "Python (\d+)\.(\d+)") {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            if ($maj -gt 3 -or ($maj -eq 3 -and $min -ge 10)) {
                $python = $candidate
                break
            }
        }
    } catch {}
}
if (-not $python) {
    Write-Warn "Python 3.10+ not found on PATH."
    Write-Host "Install it first, e.g.:  winget install Python.Python.3.12"
    Write-Host "Then re-run this installer."
    exit 1
}
Write-Step "Using '$python' ($(& $python --version))"

# --- 2. venv + pip install ---------------------------------------------------
$installDir = Join-Path $env:LOCALAPPDATA "nativetune-app"
$venvDir    = Join-Path $installDir "venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$venvScripts = Join-Path $venvDir "Scripts"

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating venv at $venvDir"
    & $python -m venv $venvDir
} else {
    Write-Step "Reusing existing venv at $venvDir"
}

Write-Step "Installing optimum from GitHub (this can take a minute)..."
& $venvPython -m pip install --quiet --upgrade pip
& $venvPython -m pip install --quiet --upgrade "git+https://github.com/RushiAgrawal2004/optimum.git"
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pip install failed - see output above."
    exit 1
}

# --- 3. put it on PATH --------------------------------------------------------
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$venvScripts*") {
    Write-Step "Adding $venvScripts to your user PATH"
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$venvScripts", "User")
    $env:PATH = "$env:PATH;$venvScripts"   # so it also works in *this* terminal
} else {
    Write-Step "$venvScripts already on PATH"
}

# --- 4. per-user data folder --------------------------------------------------
$dataHome = Join-Path $env:LOCALAPPDATA "nativetune"
foreach ($sub in @("llama", "models", "cache", "data")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $dataHome $sub) | Out-Null
}
[Environment]::SetEnvironmentVariable("NATIVETUNE_HOME", $dataHome, "User")
$env:NATIVETUNE_HOME = $dataHome

Write-Host ""
Write-Host "Installed. Open a NEW terminal (so PATH picks up the change), then:" -ForegroundColor Green
Write-Host ""
Write-Host "  optimum probe"
Write-Host ""
Write-Host "Before 'tune'/'default'/'serve' will work, put these here yourself:"
Write-Host "  $dataHome\llama\    <- llama.cpp binaries (llama-bench.exe, llama-server.exe, ...)"
Write-Host "  $dataHome\models\   <- your .gguf model files"
Write-Host "  $dataHome\data\calib-50kb.txt   <- ~50KB plain text for quality measurement"
Write-Host ""
Write-Host "Full setup details: https://github.com/RushiAgrawal2004/optimum#readme"
