#!/usr/bin/env bash

set -euo pipefail

# Ensure the REGISTRY environment variable is set.
if [ -z "${REGISTRY:-}" ]; then
    REGISTRY="$(hostname):5000"
    echo "Warning: The REGISTRY environment variable is not set. Defaulting to ${REGISTRY}."
fi

# Generate protocol buffers
bash scripts/generate_pb.sh

NAMESPACE="cornserve"

# If service names are provided as arguments, use them.
# Otherwise, recursively find all Dockerfiles in the "docker" directory.
if [ "$#" -ge 1 ]; then
    BUILD_LIST=("$@")
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

# Function to build and push a service
build_and_push() {
    local SERVICE="$1"
    echo "Building Docker image for: ${SERVICE}"

    DOCKERFILE=$(find docker -type f -name "${SERVICE}.Dockerfile" | head -n 1)
    if [[ -z "${DOCKERFILE}" ]]; then
        echo "Warning: Dockerfile for ${SERVICE} not found. Skipping."
        return
    fi

    IMAGE="${REGISTRY}/${NAMESPACE}/${SERVICE}:latest"
    
    docker build -f "${DOCKERFILE}" -t "${IMAGE}" . && docker push "${IMAGE}" && echo "Successfully built and pushed ${IMAGE}"
}

# Run all builds in parallel
for SERVICE in "${BUILD_LIST[@]}"; do
    build_and_push "${SERVICE}" &
done

wait
echo "All builds and pushes completed."
