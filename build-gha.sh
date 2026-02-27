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

# Build script for GitHub Actions — extracted from build-reusable.yml
# Expects env vars: TF_VERSION_SPEC, KERAS_VERSION_SPEC, NIGHTLY_BUILD
set -e
set -x

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install . --progress-bar off
pip install ${TF_VERSION_SPEC} ${KERAS_VERSION_SPEC} --progress-bar off --upgrade

echo "=== Building wheel ==="
if [[ "${NIGHTLY_BUILD}" == "true" ]]; then
  perl -i -lpe '$k+= s/tensorflow>=2\.[0-9]+\.[0-9]+(,<=?[0-9.]+)?;/tf-nightly;/g; END{exit($k != 1)}' setup.py
fi
python setup.py bdist_wheel

echo "=== Build complete ==="
ls -la dist/*.whl
