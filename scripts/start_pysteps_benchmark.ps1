param(
    [switch]$DryRun,
    [switch]$FetchMissing,
    [switch]$RunStepsEnsemble,
    [string]$Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Invoke-Checked {
    param(
        [string]$Description,
        [string]$Command,
        [string[]]$Arguments
    )
    Write-Step $Description
    Write-Host "$Command $($Arguments -join ' ')"
    if ($DryRun) {
        return
    }
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command $($Arguments -join ' ')"
    }
}

function Import-VsBuildEnvironment {
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cl) {
        Write-Step "MSVC compiler already visible"
        Write-Host $cl.Source
        return
    }

    $vswhereCandidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    $vswhere = $vswhereCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($null -eq $vswhere) {
        throw "vswhere.exe was not found. Finish installing Visual Studio Build Tools, then restart this shell."
    }

    $installationPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $installationPath) {
        throw "Visual Studio Build Tools with MSVC x64/x86 tools were not found."
    }

    $devCmd = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $devCmd)) {
        throw "VsDevCmd.bat was not found under $installationPath."
    }

    Write-Step "Importing Visual Studio build environment"
    if ($DryRun) {
        Write-Host $devCmd
        return
    }

    cmd /s /c "`"$devCmd`" -arch=x64 -host_arch=x64 && set" |
        ForEach-Object {
            if ($_ -match "^(.*?)=(.*)$") {
                Set-Item -LiteralPath "env:$($matches[1])" -Value $matches[2]
            }
        }

    if ($null -eq (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "MSVC environment import completed, but cl.exe is still not visible."
    }
}

Import-VsBuildEnvironment

Invoke-Checked `
    -Description "Installing PySTEPS into the project virtual environment" `
    -Command "uv" `
    -Arguments @("pip", "install", "pysteps", "--python", $Python, "--link-mode", "copy")

$benchmarkArgs = @(
    "-m", "ontario_nowcast.benchmark",
    "--config", "configs\data\milestone_1_5.yaml",
    "--include-pysteps"
)
if ($RunStepsEnsemble) {
    $benchmarkArgs += "--include-steps-ensemble"
}
if ($FetchMissing) {
    $benchmarkArgs += "--fetch-missing"
}

Invoke-Checked `
    -Description "Running Milestone 1.5 benchmark with PySTEPS" `
    -Command $Python `
    -Arguments $benchmarkArgs

Invoke-Checked `
    -Description "Running unit tests" `
    -Command $Python `
    -Arguments @("-m", "pytest", "-q")

Invoke-Checked `
    -Description "Running Ruff" `
    -Command ".venv\Scripts\ruff.exe" `
    -Arguments @("check", ".")

Write-Step "Ready"
Write-Host "Report: docs\milestone_1_5_benchmark.md"
