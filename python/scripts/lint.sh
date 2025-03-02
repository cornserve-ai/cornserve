#!/usr/bin/env bash

set -evo pipefail

echo ${BASH_SOURCE[0]}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ruff format
ruff check cornserve
pyright cornserve
