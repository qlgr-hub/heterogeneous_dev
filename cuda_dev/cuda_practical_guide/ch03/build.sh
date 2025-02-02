#!/usr/bin/env bash

set -e

ORG_PATH=$(pwd)
SCRIPT_BASE_DIR=$(cd "$(dirname $0)"; pwd)
cd "${SCRIPT_BASE_DIR}"

if [ ! -d build ]; then
    mkdir build
fi

cd build

cmake ..

make -j8

cd "${ORG_PATH}"