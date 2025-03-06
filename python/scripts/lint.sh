#!/usr/bin/env bash


echo ${BASH_SOURCE[0]}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z $GITHUB_ACTION ]]; then
  ruff format --target-version py311 cornserve tests
else
  ruff format --target-version py311 --check cornserve tests
fi

ruff check cornserve
pyright cornserve
