#!/bin/bash
# Copyright 2021 Google LLC
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

PYENV_ROOT="/home/kbuilder/.pyenv"
PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}

echo "Installing pyenv.."
git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
export PATH="/home/kbuilder/.local/bin:$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

echo "Python setup..."
pyenv install -s "$PYTHON_VERSION"
pyenv global "$PYTHON_VERSION"

cd "${KOKORO_ARTIFACTS_DIR}/github/gnn/"

PIP_TEST_PREFIX=bazel_pip

python -m venv build_venv
source build_venv/bin/activate

# Check the python version
python --version

# update pip
pip install --upgrade pip

# Install build
pip install build

TEST_ROOT=$(pwd)/${PIP_TEST_PREFIX}
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"
ln -s "$(pwd)"/tensorflow_gnn "$TEST_ROOT"/tensorflow_gnn
tag_filters="-no_oss,-oss_excluded"

# Check that `bazel` does version selection as expected.
if [[ -n "${USE_BAZEL_VERSION}" && $(bazel --version) != *${USE_BAZEL_VERSION}* ]]; then
  echo "Mismatch of configured and actual bazel version (see logged [[ command)"
  exit 1
fi

bazel clean
pip install --group test-nightly --progress-bar off --upgrade
python3 -m build --wheel
deactivate

# Start the test environment.
python3 -m venv test_venv
source test_venv/bin/activate

# Check the python version
python --version

pip install --upgrade pip
pip install dist/tensorflow_gnn-*.whl
pip uninstall -y tensorflow tf-keras ai-edge-litert
pip install --group test-nightly --progress-bar off --upgrade

echo "Final packages after all pip commands:"
pip list

# Check that tf-nightly is installed but tensorflow is not
# Also check that tf-keras-nightly is installed.
if [[ $(pip freeze | grep -q tf_nightly=; echo $?) -eq 0 && $(pip freeze | grep -q tensorflow=; echo $?) -eq 0 ]]; then
  echo "Found tensorflow and tf_nightly in the environment."
  exit 1
fi
if [[ $(pip freeze | grep -q tf_keras-nightly=; echo $?) -eq 0 && $(pip freeze | grep -q tf_keras=; echo $?) -eq 0 ]]; then
  echo "Found tf_keras and tf_keras-nightly in the environment."
  exit 1
fi
# The env variable is needed to ensure that TF keras still behaves like keras 2
bazel test --test_env="TF_USE_LEGACY_KERAS=1" --build_tag_filters="${tag_filters}" --test_tag_filters="${tag_filters}" --test_output=errors --verbose_failures=true --build_tests_only --define=no_tfgnn_py_deps=true --keep_going --experimental_repo_remote_exec //bazel_pip/tensorflow_gnn/...
