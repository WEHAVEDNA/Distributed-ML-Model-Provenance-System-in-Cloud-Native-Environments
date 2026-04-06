#!/usr/bin/env bash
# build_and_push.sh – Build all service images for Docker or Kubernetes.
#
# Usage:
#   ./scripts/build_and_push.sh                             # build only
#   ./scripts/build_and_push.sh --registry ghcr.io/acme     # tag/push to a registry prefix
#   ./scripts/build_and_push.sh --minikube                  # load into minikube image cache
#   ./scripts/build_and_push.sh --kind                      # load into kind image cache
#   ./scripts/build_and_push.sh --kind --kind-cluster demo

set -euo pipefail

REGISTRY=""
MINIKUBE=false
KIND=false
KIND_CLUSTER="kind"
TAG="latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift ;;
    --tag)      TAG="$2";      shift ;;
    --minikube) MINIKUBE=true ;;
    --kind)     KIND=true ;;
    --kind-cluster) KIND_CLUSTER="$2"; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

SERVICES=("data-ingestion" "preprocessing" "fine-tuning" "atlas-sidecar")
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

  if $KIND; then
    echo "Loading ${full_name} into kind cluster ${KIND_CLUSTER}…"
    kind load docker-image "$full_name" --name "$KIND_CLUSTER"
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
  echo "  kubectl apply -k k8s/"
fi
if $KIND; then
  echo "Images loaded into kind cluster ${KIND_CLUSTER}. Deploy with:"
  echo "  kubectl apply -k k8s/"
fi
