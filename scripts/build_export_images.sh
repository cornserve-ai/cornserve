#!/usr/bin/env bash

# Usage:
#
#     export REGISTRY=see-below-for-local-vs-distributed
#     bash scripts/build_export_images.sh [service1 service2 ...]
#
# If no services are specified, it builds all services found in the docker/ directory except for `dev`.
#
# The `REGISTRY` environment variable:
# - If set to `local`, it builds the images directly within the local k3s containerd.
# - If set to `none`, it just builds the images with Docker without exporting them to anywhere.
# - If set to `minikube`, it builds the images with Docker and loads them into Minikube.
# - If set to a URL (e.g., `localhost:30070`), it builds the images with Docker and pushes them to that registry.
# More details: https://cornserve.ai/contributor_guide/kubernetes/

set -euo pipefail

NAMESPACE="cornserve"

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
  echo "Building iamge directly within local k3s containerd"
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
  BUILD_LIST=()
  while IFS= read -r file; do
    if [[ -f "$file" ]]; then
      service=$(basename "$file" .Dockerfile)
      if [[ "$service" != "dev" ]]; then
        BUILD_LIST+=("$service")
      fi
    fi
  done < <(find docker -type f -name '*.Dockerfile')
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
  
  if [[ "${REGISTRY}" == "local" ]]; then
    echo "Building image directly within local k3s containerd..."
    sudo nerdctl build --progress=plain -f "${DOCKERFILE}" -t "${IMAGE}" .
  elif [[ "${REGISTRY}" == "none" ]]; then
    docker build --progress=plain -f "${DOCKERFILE}" -t "${IMAGE}" .
  elif [[ "${REGISTRY}" == "minikube" ]]; then
    docker build --progress=plain -f "${DOCKERFILE}" -t "${IMAGE}" .
    echo "Loading ${IMAGE} into Minikube..."
    minikube image load "${IMAGE}"
  else
    docker build --progress=plain -f "${DOCKERFILE}" -t "${PUSH_IMAGE}" .
    docker push "${PUSH_IMAGE}"
  fi

  echo "Successfully built ${IMAGE}"
}

# Run all builds in parallel
for SERVICE in "${BUILD_LIST[@]}"; do
  build_and_export "${SERVICE}" &
done

wait

echo "Successfully built and exported images for: ${BUILD_LIST[@]}"
