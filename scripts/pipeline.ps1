param(
    [string]$Step     = "all",
    [int]   $Samples  = 500,
    [string]$Split    = "train",
    [string]$Text     = ""
)

$INGEST_URL   = "http://localhost:8001"
$PREPROC_URL  = "http://localhost:8002"
$TRAIN_URL    = "http://localhost:8003"
$SIDECAR_URL  = "http://localhost:8004"

function Wait-Job($BaseUrl, $JobId, $IntervalSec = 5) {
    while ($true) {
        try {
            $s = Invoke-RestMethod "$BaseUrl/jobs/$JobId" -TimeoutSec 10 -ErrorAction Stop
            Write-Host "  status: $($s.status)$(if($s.epoch){' | epoch: '+$s.epoch})$(if($s.losses){' | losses: ['+($s.losses -join ', ')+']'})"
            if ($s.status -eq "completed") { return $s }
            if ($s.status -eq "failed")    { Write-Error "Job failed: $($s.error)"; exit 1 }
        } catch {
            Write-Host "  (connection error - retrying in ${IntervalSec}s: $($_.Exception.Message.Split([char]10)[0]))" -ForegroundColor Yellow
        }
        Start-Sleep $IntervalSec
    }
}

function Check-Health {
    Write-Host "Checking services..."
    $services = @(
        @{ name = "data-ingestion"; url = "$INGEST_URL/health" },
        @{ name = "preprocessing";  url = "$PREPROC_URL/health" },
        @{ name = "fine-tuning";    url = "$TRAIN_URL/health" },
        @{ name = "atlas-sidecar";  url = "$SIDECAR_URL/health" }
    )
    $ok = $true
    foreach ($svc in $services) {
        try {
            $r = Invoke-RestMethod $svc.url -TimeoutSec 5
            Write-Host "  [OK] $($svc.name)" -ForegroundColor Green
        } catch {
            Write-Host "  [DOWN] $($svc.name)" -ForegroundColor Red
            $ok = $false
        }
    }
    if (-not $ok) {
        Write-Error "One or more services are not running. Start the stack: docker compose up --build"
        exit 1
    }
    Write-Host ""
}

function Run-Ingest {
    Write-Host "=== Stage 1: Data Ingestion (samples=$Samples split=$Split) ===" -ForegroundColor Cyan
    $r = Invoke-RestMethod -Method Post "$INGEST_URL/ingest?split=$Split&num_samples=$Samples"
    Write-Host "  Job: $($r.job_id)"
    $s = Wait-Job $INGEST_URL $r.job_id
    Write-Host "  SHA-256 : $($s.sha256)"
    Write-Host "  S3 URI  : $($s.s3_uri)"
    if ($s.manifest_id) { Write-Host "  Manifest: $($s.manifest_id)" }
    Write-Host ""
}

function Run-Preprocess {
    Write-Host "=== Stage 2: Preprocessing (split=$Split) ===" -ForegroundColor Cyan
    $r = Invoke-RestMethod -Method Post "$PREPROC_URL/preprocess?split=$Split"
    Write-Host "  Job: $($r.job_id)"
    $s = Wait-Job $PREPROC_URL $r.job_id
    Write-Host "  Samples : $($s.num_samples)"
    Write-Host "  SHA-256 : $($s.sha256)"
    Write-Host "  S3 URI  : $($s.s3_uri)"
    if ($s.manifest_id) { Write-Host "  Manifest: $($s.manifest_id)" }
    Write-Host ""
}

function Run-Train {
    Write-Host "=== Stage 3: Fine-Tuning (split=$Split) ===" -ForegroundColor Cyan
    Write-Host "  Polling every 10s - training is slow on CPU"
    $r = Invoke-RestMethod -Method Post "$TRAIN_URL/train?split=$Split"
    Write-Host "  Job: $($r.job_id) | Device: $($r.device) | Epochs: $($r.epochs)"
    $s = Wait-Job $TRAIN_URL $r.job_id 10
    Write-Host "  Epoch losses : $($s.epoch_losses -join ' -> ')"
    Write-Host "  SHA-256      : $($s.sha256)"
    Write-Host "  S3 URI       : $($s.s3_uri)"
    if ($s.manifest_id) { Write-Host "  Manifest     : $($s.manifest_id)" }
    Write-Host ""
}

function Run-Predict {
    $input = if ($Text) { $Text } else { "This was an absolutely brilliant piece of cinema!" }
    Write-Host "=== Inference ===" -ForegroundColor Cyan
    Write-Host "  Text: $input"
    $body = @{ text = $input } | ConvertTo-Json
    $r = Invoke-RestMethod -Method Post "$TRAIN_URL/predict" -ContentType "application/json" -Body $body
    Write-Host "  Label      : $($r.label)" -ForegroundColor Yellow
    Write-Host "  Confidence : $($r.confidence)"
    Write-Host "  Positive   : $($r.scores.positive)"
    Write-Host "  Negative   : $($r.scores.negative)"
    Write-Host ""
}

function Run-Provenance {
    Write-Host "=== Provenance Registry ===" -ForegroundColor Cyan
    $reg = Invoke-RestMethod "$SIDECAR_URL/registry"
    $keys = $reg.PSObject.Properties.Name
    if ($keys.Count -eq 0) {
        Write-Host "  (empty - run the pipeline first)"
    } else {
        foreach ($k in $keys) {
            $e = $reg.$k
            Write-Host "  $k"
            Write-Host "    stage       : $($e.stage)"
            Write-Host "    type        : $($e.type)"
            Write-Host "    manifest_id : $($e.manifest_id)"
        }
    }
    Write-Host ""
}

switch ($Step.ToLower()) {
    "health"     { Check-Health }
    "ingest"     { Check-Health; Run-Ingest }
    "preprocess" { Check-Health; Run-Preprocess }
    "train"      { Check-Health; Run-Train }
    "predict"    { Run-Predict }
    "provenance" { Run-Provenance }
    "all" {
        Check-Health
        Run-Ingest
        Run-Preprocess
        Run-Train
        Run-Predict
        Run-Provenance
        Write-Host "Pipeline complete." -ForegroundColor Green
        Write-Host "  Swagger docs  : http://localhost:8001/docs  8002/docs  8003/docs  8004/docs"
        Write-Host "  MinIO console : http://localhost:9001"
    }
    default {
        Write-Host "Usage: .\pipeline.ps1 [-Step <step>] [-Samples 500] [-Split train] [-Text 'your text']"
        Write-Host "Steps: all | health | ingest | preprocess | train | predict | provenance"
    }
}
