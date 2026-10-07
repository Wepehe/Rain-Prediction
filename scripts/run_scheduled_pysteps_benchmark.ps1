param(
    [switch]$FetchMissing,
    [switch]$RunStepsEnsemble
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $RepoRoot "artifacts\milestone_1_5"
$LogPath = Join-Path $LogDir "scheduled_pysteps_benchmark.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Location $RepoRoot

try {
    "[$(Get-Date -Format o)] Starting scheduled PySTEPS benchmark" |
        Tee-Object -FilePath $LogPath -Append
    $arguments = @()
    if ($FetchMissing) {
        $arguments += "-FetchMissing"
    }
    if ($RunStepsEnsemble) {
        $arguments += "-RunStepsEnsemble"
    }
    & (Join-Path $RepoRoot "scripts\start_pysteps_benchmark.ps1") @arguments *>&1 |
        Tee-Object -FilePath $LogPath -Append
    "[$(Get-Date -Format o)] Scheduled PySTEPS benchmark completed" |
        Tee-Object -FilePath $LogPath -Append
} catch {
    "[$(Get-Date -Format o)] Scheduled PySTEPS benchmark failed: $_" |
        Tee-Object -FilePath $LogPath -Append
    exit 1
}
