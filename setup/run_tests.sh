#!/usr/bin/env bash

export http_proxy=http://127.0.0.1:12333
export https_proxy=http://127.0.0.1:12333

set -euxo pipefail

pwd  # FYI
pytest .

pushd $QSIM_CPP_PATH/build
ctest -V .
popd
