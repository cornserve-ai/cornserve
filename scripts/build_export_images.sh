#!/usr/bin/env bash

# Usage:
#
#     export REGISTRY=see-below-for-local-vs-distributed
#     bash scripts/build_export_images.sh [--max-jobs N] [service1 service2 ...]
#     bash scripts/build_export_images.sh [--max-jobs N] list
#
# If no services are specified, it builds all services found in the docker/ directory except for `dev`.
# If `list` is specified, it prints all available services (from `docker/*.Dockerfile`) and exits.
#
# The `REGISTRY` environment variable:
# - If set to `local`, it builds the images directly within the local k3s containerd.
# - If set to `none`, it just builds the images with Docker without exporting them to anywhere.
# - If set to `minikube`, it builds the images with Docker and loads them into Minikube.
# - If set to a URL (e.g., `localhost:30070`), it builds the images with Docker and pushes them to that registry.
# More details: https://cornserve.ai/contributor_guide/kubernetes/
#
# The `CACHE_REGISTRY` environment variable (optional):
# - If set (e.g., `docker.io/yourusername`), build cache is pushed to and pulled from
#   `${CACHE_REGISTRY}/${SERVICE}:${CACHE_TAG}` for each service (default tag: devcache).
# - This enables sharing BuildKit layer cache across machines via a remote registry.
# CACHE_TO_TAG=devcache PUSH_CACHE=1 CACHE_REGISTRY=docker.io/cornserve REGISTRY=local bash scripts/build_export_images.sh

set -euo pipefail

NAMESPACE="cornserve"
CACHE_REGISTRY="${CACHE_REGISTRY:-docker.io/cornserve}"
CACHE_FROM_TAG="${CACHE_FROM_TAG:-buildcache}"
CACHE_TO_TAG="${CACHE_TO_TAG:-devcache}"
MAX_JOBS=2

list_services() {
  local include_dev="${1:-true}"
  local services=()
  local service
  local file

  while IFS= read -r file; do
    [[ -f "${file}" ]] || continue
    service=$(basename "${file}" .Dockerfile)
    if [[ "${include_dev}" == "false" && "${service}" == "dev" ]]; then
      continue
    fi
    services+=("${service}")
  done < <(find docker -type f -name '*.Dockerfile' | sort)

  printf '%s\n' "${services[@]}"
}

# Parse --max-jobs flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# `list` mode: print available images/services and exit.
if [[ "$#" -eq 1 && "$1" == "list" ]]; then
  mapfile -t SERVICES < <(list_services true)
  echo "Available services:"
  echo "${SERVICES[*]}"
  exit 0
fi

# Ensure the REGISTRY environment variable is set.
if [ -z "${REGISTRY:-}" ]; then
  echo "The REGISTRY environment variable is not set."
  echo "The REGISTRY environment variable:"
  echo "- If set to 'local', it builds the images directly within the local k3s containerd."
  echo "- If set to 'none', it just builds the images with Docker without exporting them to anywhere."
  echo "- If set to 'minikube', it builds the images with Docker and loads them into Minikube."
  echo "- If set to a URL (e.g., 'localhost:30070'), it builds the images with Docker and pushes them to that registry."
  echo "More details: https://cornserve.ai/contributor_guide/kubernetes/"
  exit 1
fi

# Get the user to type their password if they want local k3s containerd export
if [[ "${REGISTRY}" == "local" ]]; then
  # Ensure k3s is installed and list existing images
  echo "Building image directly within local k3s containerd"
  k3s_bin="$(which k3s)"
  sudo "${k3s_bin}" ctr images ls | grep -i "${NAMESPACE}" || true

  # Ensure nerdctl is installed
  if ! command -v nerdctl &> /dev/null; then
    echo "nerdctl is not installed. Please install it to build images for local k3s containerd."
    exit 1
  fi

  # Ensure buildkit is configured
  nerdctl_output=$(sudo nerdctl build 2>&1 || true)
  if echo "${nerdctl_output}" | grep -q "buildkit"; then
    echo "It seems like buildkit is not configured. Please configure it to build images for local k3s containerd."
    echo "Nerdctl output:"
    echo "${nerdctl_output}"
    exit 1
  fi
fi

# If service names are provided as arguments, use them.
# Otherwise, find all Dockerfiles in the "docker" directory.
if [ "$#" -ge 1 ]; then
  BUILD_LIST=("$@")
else
  echo "Building all services found in the docker directory."
  mapfile -t BUILD_LIST < <(list_services false)
fi

echo "Building: ${BUILD_LIST[@]}"
sleep 1

# Generate profobuf Python bindings
bash scripts/generate_pb.sh

# Function to build and export the image for a single service
build_and_export() {
  local SERVICE="$1"
  echo "Building and exporting image for: ${SERVICE}"

  DOCKERFILE=$(find docker -type f -name "${SERVICE}.Dockerfile" | head -n 1)
  if [[ -z "${DOCKERFILE}" ]]; then
    echo "Warning: Dockerfile for ${SERVICE} not found. Skipping."
    return
  fi

  IMAGE="${NAMESPACE}/${SERVICE}:latest"
  PUSH_IMAGE="${REGISTRY}/${NAMESPACE}/${SERVICE}:latest"

  # Remote build cache flags (requires CACHE_REGISTRY to be set)
  # Set PUSH_CACHE=1 to also push cache to the remote registry (default: pull only)
  CACHE_FLAGS=()
  if [[ -n "${CACHE_REGISTRY:-}" ]]; then
    CACHE_FLAGS=(
      --cache-from "type=registry,ref=${CACHE_REGISTRY}/${SERVICE}:${CACHE_FROM_TAG}"
    )
    if [[ "${PUSH_CACHE:-0}" == "1" ]]; then
      CACHE_FLAGS+=(--cache-to "type=registry,ref=${CACHE_REGISTRY}/${SERVICE}:${CACHE_TO_TAG},mode=max")
    fi
  fi

  if [[ "${REGISTRY}" == "local" ]]; then
    echo "Building image directly within local k3s containerd..."
    sudo nerdctl build --progress=plain --build-arg max_jobs="${MAX_JOBS}" \
      "${CACHE_FLAGS[@]}" \
      -f "${DOCKERFILE}" -t "${IMAGE}" .
  elif [[ "${REGISTRY}" == "none" ]]; then
    docker buildx build --progress=plain --build-arg max_jobs="${MAX_JOBS}" \
      "${CACHE_FLAGS[@]}" \
      -f "${DOCKERFILE}" -t "${IMAGE}" --load .
  elif [[ "${REGISTRY}" == "minikube" ]]; then
    docker buildx build --progress=plain --build-arg max_jobs="${MAX_JOBS}" \
      "${CACHE_FLAGS[@]}" \
      -f "${DOCKERFILE}" -t "${IMAGE}" --load .
    echo "Loading ${IMAGE} into Minikube..."
    minikube image load "${IMAGE}"
  else
    docker buildx build --progress=plain --build-arg max_jobs="${MAX_JOBS}" \
      "${CACHE_FLAGS[@]}" \
      -f "${DOCKERFILE}" -t "${PUSH_IMAGE}" --push .
  fi

  echo "Successfully built ${IMAGE}"
}

# Run all builds in parallel
for SERVICE in "${BUILD_LIST[@]}"; do
  build_and_export "${SERVICE}" &
done

wait

echo "Successfully built and exported images for: ${BUILD_LIST[@]}"
