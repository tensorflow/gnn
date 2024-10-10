#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -x

cd "${KOKORO_ARTIFACTS_DIR}/github/gnn/"

sudo apt-get install -y "python${PYTHON_VERSION}"

# Update alternatives, taken from corresponding Keras OSS test script
sudo update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python$PYTHON_VERSION" 1
sudo apt-get install -y python$PYTHON_VERSION-venv

PYTHON_BINARY="/usr/bin/python${PYTHON_VERSION}"
PIP_TEST_PREFIX=bazel_pip

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate

# Debug messages to indicate the python version
python --version
python3 --version

TEST_ROOT=$(pwd)/${PIP_TEST_PREFIX}
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"
ln -s "$(pwd)"/tensorflow_gnn "$TEST_ROOT"/tensorflow_gnn

# Print the OS version
cat /etc/os-release

# Prepend common tag filters to a defined env_var
# For example, tests for TF 2.8 shouldn't run RNG-dependent tests
# These tag filters are enforced to start with a comma for separation
tag_filters="-no_oss,-oss_excluded${TAG_FILTERS}"

bazel clean
pip install -r requirements-dev.txt --progress-bar off
pip install tensorflow=="${TF_VERSION}" --progress-bar off --upgrade
if [[ "$TF_USE_LEGACY_KERAS" == 1 ]]; then
  pip install tf-keras=="${TF_VERSION}" --progress-bar off --upgrade
fi
python3 setup.py bdist_wheel
pip uninstall -y tensorflow_gnn
pip install dist/tensorflow_gnn-*.whl

bazel test --test_env=TF_USE_LEGACY_KERAS --build_tag_filters="${tag_filters}" --test_tag_filters="${tag_filters}" --test_output=errors --verbose_failures=true --build_tests_only --define=no_tfgnn_py_deps=true --keep_going --experimental_repo_remote_exec //bazel_pip/tensorflow_gnn/...
