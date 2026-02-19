#!/bin/bash
# Copyright 2026 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Test script for GitHub Actions — aligned with build_versioned.sh
# Expects env vars:
#   TF_VERSION_SPEC, KERAS_VERSION_SPEC — pip version specifiers
#   TF_USE_LEGACY_KERAS — '0' or '1'
#   BUILD_TAG_FILTERS, TEST_TAG_FILTERS — bazel tag filters
#   NIGHTLY_ENV — 'true'/'false'
#   WHEEL_DIR — path to directory containing the .whl
set -e
set -x

echo "=== Installing dependencies ==="
pip install --upgrade pip

if [[ "$TF_USE_LEGACY_KERAS" == 1 ]]; then
  pip install --group test-tf216plus --progress-bar off --upgrade
else
  pip install --group test-pre-tf216 --progress-bar off --upgrade
fi

echo "=== Installing wheel ==="
pip install "${WHEEL_DIR}"/tensorflow_gnn-*.whl
pip list

echo "=== Verifying nightly (if applicable) ==="
if [[ "${NIGHTLY_ENV}" == "true" ]]; then
  pip freeze | grep -q tf-nightly= && ! pip freeze | grep -q tensorflow=
  pip freeze | grep -q tf-keras-nightly= && ! pip freeze | grep -q tf-keras=
fi

echo "=== Running Bazel tests (build_versioned.sh approach) ==="
export PYTHON_BIN_PATH=$(which python3)

# Setup bazel_pip/ symlink (matching build_versioned.sh lines 45-48)
PIP_TEST_PREFIX=bazel_pip
TEST_ROOT=$(pwd)/${PIP_TEST_PREFIX}
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"
ln -s "$(pwd)"/tensorflow_gnn "$TEST_ROOT"/tensorflow_gnn

# Tag filters (matching build_versioned.sh line 56)
tag_filters="${BUILD_TAG_FILTERS:--no_oss,-oss_excluded}"

# Fix CRLF line endings on shell scripts (Windows git autocrlf breaks sh_test)
find . -name '*.sh' -exec sed -i 's/\r$//' {} +

bazel clean

# Test command aligned with build_versioned.sh line 77
bazel test \
  --test_env="TF_USE_LEGACY_KERAS=${TF_USE_LEGACY_KERAS}" \
  --build_tag_filters="${tag_filters}" \
  --test_tag_filters="${TEST_TAG_FILTERS:--no_oss,-oss_excluded}" \
  --test_output=errors \
  --verbose_failures=true \
  --build_tests_only \
  --define=no_tfgnn_py_deps=true \
  --keep_going \
  --experimental_repo_remote_exec \
  //bazel_pip/tensorflow_gnn/...
