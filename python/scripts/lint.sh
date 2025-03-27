#!/usr/bin/env bash


echo ${BASH_SOURCE[0]}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z $GITHUB_ACTION ]]; then
  ruff format --target-version py311 cornserve tests
  ruff check --fix-only --select I cornserve tests
else
  ruff format --target-version py311 --check cornserve tests
  ruff check --select I cornserve tests
fi

ruff check --target-version py311 cornserve
pyright cornserve
