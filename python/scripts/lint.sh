#!/usr/bin/env bash


echo ${BASH_SOURCE[0]}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z $GITHUB_ACTION ]]; then
  black cornserve
else
  set -e
  black --check cornserve
fi

ruff check cornserve
pyright cornserve
