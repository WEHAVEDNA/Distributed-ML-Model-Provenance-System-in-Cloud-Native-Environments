param(
    [string]$Namespace = "ml-pipeline"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Streaming Kubernetes pod status for namespace '$Namespace'..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop watching." -ForegroundColor Yellow
Write-Host ""
Write-Host "Open another terminal for port-forwards when you are ready:" -ForegroundColor Green
Write-Host "  pwsh scripts/port_forward.ps1" -ForegroundColor White
Write-Host ""

& kubectl get pods -n $Namespace -o wide --watch
exit $LASTEXITCODE
