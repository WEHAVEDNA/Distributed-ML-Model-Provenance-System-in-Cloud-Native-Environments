param(
    [string]$Namespace = "ml-pipeline",
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$forwards = @(
    @{ svc = "data-ingestion"; local = 8001; remote = 8001 },
    @{ svc = "preprocessing";  local = 8002; remote = 8002 },
    @{ svc = "fine-tuning";    local = 8003; remote = 8003 },
    @{ svc = "atlas-sidecar";  local = 8004; remote = 8004 }
)

if ($Stop) {
    Write-Host "Stopping all kubectl port-forward processes..." -ForegroundColor Yellow
    Get-Process -Name "kubectl" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*port-forward*" } |
        Stop-Process -Force
    Write-Host "Done." -ForegroundColor Green
    exit 0
}

Write-Host "Starting port-forwards for ml-pipeline services..." -ForegroundColor Cyan
Write-Host "(Run with -Stop to kill them later)" -ForegroundColor DarkGray

$jobs = @()
foreach ($fwd in $forwards) {
    $arg = "port-forward -n $Namespace svc/$($fwd.svc) $($fwd.local):$($fwd.remote)"
    Write-Host "  $($fwd.svc): localhost:$($fwd.local) -> $($fwd.remote)" -ForegroundColor Gray
    $jobs += Start-Job -ScriptBlock {
        param($a)
        & kubectl $a.Split(" ")
    } -ArgumentList $arg
}

Write-Host ""
Write-Host "All port-forwards running in background." -ForegroundColor Green
Write-Host "Services are now reachable at:" -ForegroundColor Green
foreach ($fwd in $forwards) {
    Write-Host "  http://localhost:$($fwd.local)/health" -ForegroundColor White
}
Write-Host ""
Write-Host "Run tests with:" -ForegroundColor Cyan
Write-Host "  pytest tests/ -v -m smoke"
Write-Host "  pytest tests/ -v -m `"wired`" --samples 200"
Write-Host ""
Write-Host "Press Ctrl+C to stop port-forwards." -ForegroundColor Yellow

try {
    Wait-Job $jobs | Out-Null
} finally {
    $jobs | Stop-Job -PassThru | Remove-Job
}
