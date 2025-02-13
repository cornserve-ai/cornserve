#!/usr/bin/env bash
set -euo pipefail

# Ensure the REGISTRY environment variable is set.
if [ -z "${REGISTRY:-}" ]; then
    echo "Error: The REGISTRY environment variable is not set. Please set it and try again."
    exit 1
fi

# Generate protocol buffers
bash scripts/generate_pb.sh

NAMESPACE="cornserve"

# If a service name is provided as the first argument, use it.
# Otherwise, recursively find all Dockerfiles in the "docker" directory.
if [ "$#" -ge 1 ]; then
    BUILD_LIST=("$1")
else
    echo "Building all services found recursively in the docker directory."
    BUILD_LIST=()
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            service=$(basename "$file" .Dockerfile)
            BUILD_LIST+=("$service")
        fi
    done < <(find docker -type f -name '*.Dockerfile')
fi

# Iterate over each service in the build list
for SERVICE in "${BUILD_LIST[@]}"; do
    echo "Building Docker image for: ${SERVICE}"
    # Locate the Dockerfile for the service recursively.
    DOCKERFILE=$(find docker -type f -name "${SERVICE}.Dockerfile" | head -n 1)
    if [[ -z "${DOCKERFILE}" ]]; then
        echo "Warning: Dockerfile for ${SERVICE} not found. Skipping."
        continue
    fi
    IMAGE="${REGISTRY}/${NAMESPACE}/${SERVICE}:latest"
    docker build -f "${DOCKERFILE}" -t "${IMAGE}" .
    docker push "${IMAGE}"
    echo "Successfully built and pushed ${IMAGE}"
done
