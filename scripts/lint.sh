#!/usr/bin/env bash

set -ev

function do_format() {
  if [[ -z $GITHUB_ACTION ]]; then
    ruff format --target-version py311 $@
    ruff check --fix-only --select I $@
  else
    ruff format --target-version py311 --check $@
    ruff check --select I $@
  fi
}

function do_check() {
  ruff check --target-version py311 $@
  pyright $@
}

pushd python
do_format cornserve tests ../examples
do_check cornserve ../examples
popd

pushd tasklib
do_format cornserve_tasklib
do_check cornserve_tasklib
popd

# if [[ -z $GITHUB_ACTION ]]; then
#   ruff format --target-version py311 cornserve tests ../examples
#   ruff check --fix-only --select I cornserve tests ../examples
# else
#   ruff format --target-version py311 --check cornserve tests ../examples
#   ruff check --select I cornserve tests ../examples
# fi
#
# ruff check --target-version py311 cornserve ../examples 
# pyright cornserve ../examples
