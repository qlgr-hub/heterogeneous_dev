#!/usr/bin/env bash

set -e

cd /workspace || exit
jupyter lab --allow-root --no-browser --ip=0.0.0.0 &

sudo /usr/sbin/sshd -D
