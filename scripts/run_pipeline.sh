#!/usr/bin/env bash
# run_pipeline.sh – Execute the full ML provenance pipeline end-to-end.
#
# Works in two modes:
#   1. docker-compose  (default): services are already running locally
#   2. kubernetes (--k8s):        port-forwards services from the cluster
#
# Usage:
#   ./scripts/run_pipeline.sh                    # docker-compose mode
#   ./scripts/run_pipeline.sh --k8s              # kubernetes mode
#   ./scripts/run_pipeline.sh --samples 200      # override num_samples
#   ./scripts/run_pipeline.sh --k8s --samples 200

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
INGEST_URL="http://localhost:8001"
PREPROC_URL="http://localhost:8002"
FINETUNE_URL="http://localhost:8003"
NUM_SAMPLES=500
SPLIT="train"
K8S_MODE=false
NS="ml-pipeline"

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --k8s)       K8S_MODE=true ;;
    --samples)   NUM_SAMPLES="$2"; shift ;;
    --split)     SPLIT="$2";       shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── K8s port-forward (background) ────────────────────────────────────────────
PF_PIDS=()
cleanup() { for pid in "${PF_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

if $K8S_MODE; then
  info "Starting kubectl port-forwards (namespace: $NS)…"
  kubectl port-forward -n "$NS" svc/data-ingestion 8001:8001 &>/dev/null & PF_PIDS+=($!)
  kubectl port-forward -n "$NS" svc/preprocessing  8002:8002 &>/dev/null & PF_PIDS+=($!)
  kubectl port-forward -n "$NS" svc/fine-tuning    8003:8003 &>/dev/null & PF_PIDS+=($!)
  sleep 3   # allow port-forwards to stabilise
fi

# ── Helper: poll until job completes ─────────────────────────────────────────
wait_for_job() {
  local base_url="$1"
  local job_id="$2"
  local label="$3"
  local max_wait="${4:-1800}"   # 30 min default
  local elapsed=0

  while true; do
    status=$(curl -sf "${base_url}/jobs/${job_id}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])" 2>/dev/null || echo "unknown")
    case "$status" in
      completed) info "${label}: completed ✓"; break ;;
      failed)
        detail=$(curl -sf "${base_url}/jobs/${job_id}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','unknown error'))" 2>/dev/null)
        err "${label} failed: ${detail}"
        ;;
      *)
        if (( elapsed >= max_wait )); then
          err "${label} timed out after ${max_wait}s"
        fi
        echo -ne "  ${label}: ${status} (${elapsed}s)\r"
        sleep 5; elapsed=$(( elapsed + 5 ))
        ;;
    esac
  done
}

# ── Health checks ─────────────────────────────────────────────────────────────
info "Checking service health…"
for svc in "$INGEST_URL" "$PREPROC_URL" "$FINETUNE_URL"; do
  curl -sf "${svc}/health" >/dev/null || err "Service at ${svc} is not healthy. Is it running?"
done
info "All services healthy."

echo ""
echo "════════════════════════════════════════════════════"
echo " ML Provenance Pipeline"
echo " samples=$NUM_SAMPLES  split=$SPLIT"
echo "════════════════════════════════════════════════════"

# ── Stage 1: Data Ingestion ───────────────────────────────────────────────────
echo ""
info "Stage 1 – Data Ingestion (IMDB, ${NUM_SAMPLES} samples)"
INGEST_RESP=$(curl -sf -X POST \
  "${INGEST_URL}/ingest?split=${SPLIT}&num_samples=${NUM_SAMPLES}")
INGEST_JOB=$(echo "$INGEST_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
info "Ingestion job: ${INGEST_JOB}"
wait_for_job "$INGEST_URL" "$INGEST_JOB" "Data Ingestion"

INGEST_INFO=$(curl -sf "${INGEST_URL}/jobs/${INGEST_JOB}")
echo "  samples : $(echo $INGEST_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("num_samples","?"))')"
echo "  sha256  : $(echo $INGEST_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("sha256","?"))')"
echo "  s3_uri  : $(echo $INGEST_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("s3_uri","?"))')"

# ── Stage 2: Preprocessing ────────────────────────────────────────────────────
echo ""
info "Stage 2 – Preprocessing (BERT tokenizer, max_length=128)"
PREPROC_RESP=$(curl -sf -X POST \
  "${PREPROC_URL}/preprocess?split=${SPLIT}")
PREPROC_JOB=$(echo "$PREPROC_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
info "Preprocessing job: ${PREPROC_JOB}"
wait_for_job "$PREPROC_URL" "$PREPROC_JOB" "Preprocessing"

PREPROC_INFO=$(curl -sf "${PREPROC_URL}/jobs/${PREPROC_JOB}")
echo "  samples   : $(echo $PREPROC_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("num_samples","?"))')"
echo "  max_length: $(echo $PREPROC_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("max_length","?"))')"
echo "  sha256    : $(echo $PREPROC_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("sha256","?"))')"

# ── Stage 3: Fine-Tuning ──────────────────────────────────────────────────────
echo ""
info "Stage 3 – Fine-Tuning bert-base-uncased on IMDB"
warn "This can take 10–30 min on CPU. Monitor progress with: curl ${FINETUNE_URL}/jobs/<id>"
TRAIN_RESP=$(curl -sf -X POST \
  "${FINETUNE_URL}/train?split=${SPLIT}")
TRAIN_JOB=$(echo "$TRAIN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
info "Training job: ${TRAIN_JOB}  device=$(echo $TRAIN_RESP | python3 -c 'import sys,json; print(json.load(sys.stdin).get("device","?"))')"
wait_for_job "$FINETUNE_URL" "$TRAIN_JOB" "Fine-Tuning" 3600

TRAIN_INFO=$(curl -sf "${FINETUNE_URL}/jobs/${TRAIN_JOB}")
echo "  epochs      : $(echo $TRAIN_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("epochs","?"))')"
echo "  epoch_losses: $(echo $TRAIN_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("epoch_losses","?"))')"
echo "  sha256      : $(echo $TRAIN_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("sha256","?"))')"
echo "  model_uri   : $(echo $TRAIN_INFO | python3 -c 'import sys,json; print(json.load(sys.stdin).get("s3_uri","?"))')"

# ── Smoke test: inference ─────────────────────────────────────────────────────
echo ""
info "Smoke test – Inference"
TEXTS=(
  "This movie was absolutely fantastic! The acting was superb."
  "What a terrible waste of time. I fell asleep after 20 minutes."
  "An okay film, nothing special but watchable enough."
)
for text in "${TEXTS[@]}"; do
  RESULT=$(curl -sf -X POST "${FINETUNE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"${text}\"}")
  label=$(echo "$RESULT" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["label"])')
  conf=$(echo "$RESULT"  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["confidence"])')
  printf "  %-60s → %s (%.4f)\n" "${text:0:60}" "$label" "$conf"
done

echo ""
info "Model info: curl ${FINETUNE_URL}/model/info"
echo ""
echo "════════════════════════════════════════════════════"
info "Pipeline complete ✓"
echo "════════════════════════════════════════════════════"
