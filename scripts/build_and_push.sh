#!/usr/bin/env bash
# build_and_push.sh – Build all three service images.
#
# Usage:
#   ./scripts/build_and_push.sh                # build only (for docker-compose / minikube)
#   ./scripts/build_and_push.sh --registry ecr # push to ECR (requires AWS CLI + docker login)
#   ./scripts/build_and_push.sh --minikube     # load into minikube image cache

set -euo pipefail

REGISTRY=""
MINIKUBE=false
TAG="latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift ;;
    --tag)      TAG="$2";      shift ;;
    --minikube) MINIKUBE=true ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

SERVICES=("data-ingestion" "preprocessing" "fine-tuning")
PORTS=(8001 8002 8003)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building ML provenance service images (tag=$TAG)…"

for i in "${!SERVICES[@]}"; do
  svc="${SERVICES[$i]}"
  image_name="ml-provenance/${svc}:${TAG}"
  if [[ -n "$REGISTRY" ]]; then
    full_name="${REGISTRY}/${image_name}"
  else
    full_name="$image_name"
  fi

  echo ""
  echo "── Building ${svc} ──────────────────────────────────────"
  docker build -t "$full_name" "${ROOT}/services/${svc}"

  if $MINIKUBE; then
    echo "Loading ${full_name} into minikube…"
    minikube image load "$full_name"
  fi

  if [[ -n "$REGISTRY" ]]; then
    echo "Pushing ${full_name}…"
    docker push "$full_name"
  fi
done

echo ""
echo "All images built successfully."
if $MINIKUBE; then
  echo "Images loaded into minikube. Deploy with:"
  echo "  kubectl apply -f k8s/"
fi
