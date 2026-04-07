param(
    [string]$Tag = "latest",
    [string]$Registry = "",
    [switch]$Push,
    [switch]$DockerDesktop,
    [switch]$Minikube,
    [switch]$Kind,
    [string]$KindCluster = "kind"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Services = @("data-ingestion", "preprocessing", "fine-tuning", "atlas-sidecar")

function Get-ImageName([string]$Service) {
    $base = "ml-provenance/$Service`:$Tag"
    if ([string]::IsNullOrWhiteSpace($Registry)) {
        return $base
    }
    return "$Registry/$base"
}

Write-Host "Building ML provenance service images (tag=$Tag)..." -ForegroundColor Cyan

if ($DockerDesktop) {
    Write-Host "Docker Desktop mode enabled." -ForegroundColor Yellow
    Write-Host "Images are built into the local store used by the docker-desktop cluster." -ForegroundColor Yellow
}

foreach ($service in $Services) {
    $image = Get-ImageName $service
    $context = Join-Path $Root "services/$service"

    Write-Host ""
    Write-Host "== Building $service ==" -ForegroundColor Cyan
    & docker build -t $image $context
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($Minikube) {
        Write-Host "Loading $image into minikube..." -ForegroundColor Yellow
        & minikube image load $image
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if ($Kind) {
        Write-Host "Loading $image into kind cluster $KindCluster..." -ForegroundColor Yellow
        & kind load docker-image $image --name $KindCluster
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if ($Push -and -not [string]::IsNullOrWhiteSpace($Registry)) {
        Write-Host "Pushing $image..." -ForegroundColor Yellow
        & docker push $image
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

Write-Host ""
Write-Host "All images built successfully." -ForegroundColor Green
if ($DockerDesktop) {
    Write-Host "Next: pwsh scripts/deploy_k8s.ps1 -DockerDesktop" -ForegroundColor Green
} elseif ($Minikube -or $Kind) {
    Write-Host "Next: kubectl apply -k k8s" -ForegroundColor Green
}
