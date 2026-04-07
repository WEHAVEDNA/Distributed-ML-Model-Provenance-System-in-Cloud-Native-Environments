param(
    [string]$Namespace = "ml-pipeline",
    [switch]$BuildImages,
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

function Get-KubectlCurrentContext() {
    $context = (& kubectl config current-context 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not determine the current kubectl context."
    }
    return ($context | Out-String).Trim()
}

function Test-NamespaceExists([string]$Namespace) {
    & kubectl get namespace $Namespace 1>$null 2>$null
    return $LASTEXITCODE -eq 0
}

function Wait-ForNoPods([string]$Namespace, [string]$Selector, [int]$TimeoutSeconds = 180) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $pods = & kubectl get pods -n $Namespace -l $Selector -o name 2>$null
        if ($LASTEXITCODE -ne 0) {
            Start-Sleep -Seconds 2
            continue
        }

        $activePods = @($pods | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($activePods.Count -eq 0) {
            return
        }

        Start-Sleep -Seconds 2
    }

    throw "Pods matching selector '$Selector' in namespace '$Namespace' did not terminate within $TimeoutSeconds seconds."
}

function Wait-ForJobCompletion([string]$Namespace, [string]$JobName, [int]$TimeoutSeconds = 180) {
    Write-Host "Waiting for job/$JobName..." -ForegroundColor Yellow
    & kubectl wait --for=condition=complete "job/$JobName" -n $Namespace "--timeout=$($TimeoutSeconds)s"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Reset-Workloads([string]$Namespace) {
    if (-not (Test-NamespaceExists -Namespace $Namespace)) {
        return
    }

    $deployments = @("minio", "atlas-sidecar", "data-ingestion", "preprocessing", "fine-tuning")
    $jobs = @("minio-init")

    Write-Host "Cleaning existing workloads so startup does not reuse old pods..." -ForegroundColor Yellow

    foreach ($job in $jobs) {
        & kubectl delete job $job -n $Namespace --ignore-not-found --wait=true
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    foreach ($deployment in $deployments) {
        & kubectl delete deployment $deployment -n $Namespace --ignore-not-found --wait=true
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    foreach ($deployment in $deployments) {
        Wait-ForNoPods -Namespace $Namespace -Selector "app=$deployment"
    }
}

if ($BuildImages) {
    $buildArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "build_images.ps1"),
        "-Tag", $Tag
    )
    if ($Registry) { $buildArgs += @("-Registry", $Registry) }
    if ($Push) { $buildArgs += "-Push" }
    if ($DockerDesktop) { $buildArgs += "-DockerDesktop" }
    if ($Minikube) { $buildArgs += "-Minikube" }
    if ($Kind) { $buildArgs += @("-Kind", "-KindCluster", $KindCluster) }

    & pwsh @buildArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$kustomizePath = Join-Path $Root "k8s"
if ($DockerDesktop) {
    $currentContext = Get-KubectlCurrentContext
    if ($currentContext -ne "docker-desktop") {
        throw "Docker Desktop mode requires kubectl context 'docker-desktop'. Current context: '$currentContext'."
    }
    Write-Host "Docker Desktop mode enabled." -ForegroundColor Yellow
    Write-Host "Using kubectl context '$currentContext'." -ForegroundColor Yellow
    Reset-Workloads -Namespace $Namespace
}

Write-Host "Applying Kubernetes manifests..." -ForegroundColor Cyan
& kubectl apply -k $kustomizePath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Waiting for deployment/minio..." -ForegroundColor Yellow
& kubectl rollout status "deployment/minio" -n $Namespace --timeout=300s
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Wait-ForJobCompletion -Namespace $Namespace -JobName "minio-init"

$deployments = @("atlas-sidecar", "data-ingestion", "preprocessing", "fine-tuning")
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
