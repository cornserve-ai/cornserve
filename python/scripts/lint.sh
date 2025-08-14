#!/usr/bin/env bash

set -ev

echo ${BASH_SOURCE[0]}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z $GITHUB_ACTION ]]; then
  ruff format --target-version py311 cornserve tests ../examples ../benchmark
  ruff check --fix-only --select I cornserve tests ../examples ../benchmark
else
  ruff format --target-version py311 --check cornserve tests ../examples ../benchmark
  ruff check --select I cornserve tests ../examples ../benchmark
fi

ruff check --target-version py311 cornserve ../examples ../benchmark
pyright cornserve ../examples ../benchmark
