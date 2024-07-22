#!/usr/bin/env bash

set -e

TOPDIR=$(pwd)

mkdir -p "${TOPDIR}/build"
cd "${TOPDIR}/build"
rm -rf ./*

cmake ..
make