param(
    [string]$Namespace = "ml-pipeline",
    [switch]$BuildImages,
    [string]$Tag = "latest",
    [string]$Registry = "",
    [switch]$Push,
    [switch]$Minikube,
    [switch]$Kind,
    [string]$KindCluster = "kind"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if ($BuildImages) {
    $buildArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "build_images.ps1"),
        "-Tag", $Tag
    )
    if ($Registry) { $buildArgs += @("-Registry", $Registry) }
    if ($Push) { $buildArgs += "-Push" }
    if ($Minikube) { $buildArgs += "-Minikube" }
    if ($Kind) { $buildArgs += @("-Kind", "-KindCluster", $KindCluster) }

    & pwsh @buildArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Applying Kubernetes manifests..." -ForegroundColor Cyan
& kubectl apply -k (Join-Path $Root "k8s")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$deployments = @("minio", "atlas-sidecar", "data-ingestion", "preprocessing", "fine-tuning")
foreach ($deployment in $deployments) {
    Write-Host "Waiting for deployment/$deployment..." -ForegroundColor Yellow
    & kubectl rollout status "deployment/$deployment" -n $Namespace --timeout=300s
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "Kubernetes deployment is ready." -ForegroundColor Green
Write-Host "Forward service ports for local testing:" -ForegroundColor Green
Write-Host "  pwsh scripts/port_forward.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Or forward manually:" -ForegroundColor DarkGray
Write-Host "  kubectl port-forward -n $Namespace svc/data-ingestion 8001:8001"
Write-Host "  kubectl port-forward -n $Namespace svc/preprocessing 8002:8002"
Write-Host "  kubectl port-forward -n $Namespace svc/fine-tuning 8003:8003"
Write-Host "  kubectl port-forward -n $Namespace svc/atlas-sidecar 8004:8004"
