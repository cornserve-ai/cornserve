#!/usr/local/env bash

# Usage:
#
#     REGISTRY=<your_registry> bash kubernetes/set_registry.sh
#
# Update the private registry URL in the cornserve dev overlay kustomization file.
# Override `KUSTOMIZATION_FILE` to change the file path.

set -evo pipefail

# Ensure the environment variable REGISTRY is set
if [[ -z "${REGISTRY}" ]]; then
  echo "Error: REGISTRY environment variable is not set."
  exit 1
fi

KUSTOMIZATION_FILE=${KUSTOMIZATION_FILE:-"kubernetes/kustomize/cornserve/overlays/dev/kustomization.yaml"}
if [[ ! -f "$KUSTOMIZATION_FILE" ]]; then
  echo "Error: Kustomization file '$KUSTOMIZATION_FILE' does not exist."
  exit 1
fi

K3S_REGISTRIES_FILE=${K3S_REGISTRIES_FILE:-"kubernetes/k3s/registries.yaml"}
if [[ ! -f "$K3S_REGISTRIES_FILE" ]]; then
  echo "Error: K3s registries file '$K3S_REGISTRIES_FILE' does not exist."
  exit 1
fi

echo "Editing registry in $KUSTOMIZATION_FILE and $K3S_REGISTRIES_FILE to $REGISTRY"

sed -i.bak -e "s#localhost:5000#$REGISTRY#g" "$KUSTOMIZATION_FILE"
rm "$KUSTOMIZATION_FILE.bak"

sed -i.bak -e "s#localhost:5000#$REGISTRY#g" "$K3S_REGISTRIES_FILE"
rm "$K3S_REGISTRIES_FILE.bak"
