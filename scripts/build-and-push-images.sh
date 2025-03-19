#!/usr/bin/env bash

set -euo pipefail

# Ensure the REGISTRY environment variable is set.
if [ -z "${REGISTRY:-}" ]; then
    REGISTRY="$(hostname -f):5000"
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

    IMAGE="${NAMESPACE}/${SERVICE}:latest"
    PUSH_IMAGE="${REGISTRY}/${NAMESPACE}/${SERVICE}:latest"
    
    docker build -f "${DOCKERFILE}" -t "${IMAGE}" .

    if [[ "${REGISTRY}" == "local" ]]; then
        echo "Exporting image to local k3s containerd..."
        docker save "${IMAGE}" | invoke_k3s ctr images import -
    else
        echo "Pushing to ${REGISTRY}..."
        docker tag "${IMAGE}" "${PUSH_IMAGE}"
        docker push "${PUSH_IMAGE}"
    fi

    echo "Successfully built and exported ${IMAGE}"
}

invoke_k3s() {
    k3s_bin="$(which k3s)"
    sudo "${k3s_bin}" "$@"
}

# Get the user to type their password if they want local k3s containerd export
if [[ "${REGISTRY}" == "local" ]]; then
    # Ensure k3s is installed and record its binary location
    echo "Exporting image to local k3s containerd. Existing images are:"
    invoke_k3s ctr images ls | grep -i "${NAMESPACE}"
fi

# Run all builds in parallel
for SERVICE in "${BUILD_LIST[@]}"; do
    build_and_push "${SERVICE}" &
done

wait
echo "All builds and pushes completed."
