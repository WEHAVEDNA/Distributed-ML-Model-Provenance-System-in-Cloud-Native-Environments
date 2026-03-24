#!/usr/bin/env bash
# build_and_push.sh – Build all three service images.
#
# Usage:
#   ./scripts/build_and_push.sh                         # build only
#   ./scripts/build_and_push.sh --kind                  # load into Kind cluster
#   ./scripts/build_and_push.sh --kind atlas-pipeline   # named Kind cluster
#   ./scripts/build_and_push.sh --minikube              # load into minikube
#   ./scripts/build_and_push.sh --registry ecr          # push to ECR

set -euo pipefail

REGISTRY=""
MINIKUBE=false
KIND=false
KIND_CLUSTER="atlas-pipeline"
TAG="latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift ;;
    --tag)      TAG="$2";      shift ;;
    --minikube) MINIKUBE=true ;;
    --kind)
      KIND=true
      if [[ "${2:-}" != "" && "${2:-}" != --* ]]; then
        KIND_CLUSTER="$2"; shift
      fi
      ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

SERVICES=("data-ingestion" "preprocessing" "fine-tuning")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building ML provenance service images (tag=$TAG)…"

for svc in "${SERVICES[@]}"; do
  image_name="ml-provenance/${svc}:${TAG}"
  full_name="${REGISTRY:+${REGISTRY}/}${image_name}"

  echo ""
  echo "── Building ${svc} ──────────────────────────────────────"
  docker build -t "$full_name" "${ROOT}/services/${svc}"

  if $KIND; then
    echo "Loading ${full_name} into Kind cluster '${KIND_CLUSTER}'…"
    kind load docker-image "$full_name" --name "${KIND_CLUSTER}"
  fi

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
echo ""
echo "Next — deploy to Kubernetes:"
echo "  kubectl apply -f ${ROOT}/k8s/services.yaml"
echo "  kubectl rollout status deployment -n ml-pipeline"
