#!/usr/bin/env bash

set -e

ORG_PATH=$(pwd)
SCRIPT_BASE_DIR=$(cd "$(dirname $0)"; pwd)
cd "${SCRIPT_BASE_DIR}"

if [ ! -d build ]; then
    mkdir build
fi

make all cxx -j8

cd "${ORG_PATH}"