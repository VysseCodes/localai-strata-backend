#!/bin/bash
set -e

# The Strata backend is a thin gRPC shim: it needs only grpcio/protobuf. The
# model, its engine and its CUDA libraries live in the Strata folder that
# Strata's own setup prepares (see strata-setup.sh), so this install is the
# same on every build profile and needs no GPU packages.

backend_dir=$(dirname $0)
if [ -d $backend_dir/common ]; then
    source $backend_dir/common/libbackend.sh
else
    source $backend_dir/../common/libbackend.sh
fi

# A bare-metal install may have no uv; the standard venv works just as well
# for this backend's three packages.
if [ "x${USE_PIP:-}" != "xtrue" ] && ! command -v uv >/dev/null 2>&1; then
    echo "uv not found: using python -m venv and pip"
    USE_PIP=true
fi

installRequirements
