#!/usr/bin/env bash

# This script spins up a Docker registry container.
# You potentially want to tweak the volume path to match your setup.

set -evo pipefail

docker run -d \
  -p 5000:5000 \
  -e REGISTRY_STORAGE_DELETE_ENABLED=true \
  --restart=always \
  --name cornserve-registry \
  --volume /data/cornserve/registry:/var/lib/registry \
  registry:2
