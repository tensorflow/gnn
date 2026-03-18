#!/bin/bash
# Copyright 2026 Google LLC
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

# For pyenv python installation on ml-build image
sudo apt-get update
sudo apt-get install -y libbz2-dev liblzma-dev libncurses-dev libffi-dev libssl-dev libreadline-dev libsqlite3-dev zlib1g-dev

PYENV_ROOT="$HOME/.pyenv"
PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}

function force_tensorflow_version() {
  if [[ -z "${TF_VERSION}" ]]; then
    echo "TF_VERSION is not set. Not forcing tensorflow version."
    return
  fi

  pip install tensorflow=="${TF_VERSION}" --progress-bar off --upgrade
  if [[ "$TF_USE_LEGACY_KERAS" == 1 ]]; then
    pip install tf-keras=="${TF_VERSION}" --progress-bar off --upgrade
  fi
}

echo "Installing pyenv.."
git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
export PATH="$HOME/.local/bin:$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

echo "Python setup..."
pyenv install -s "$PYTHON_VERSION"
pyenv global "$PYTHON_VERSION"

PIP_TEST_PREFIX=bazel_pip

python -m venv build_venv
source build_venv/bin/activate

# Debug messages to indicate the python version
python --version

# update pip
pip install --upgrade pip

# Install build
pip install build

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

# Check that `bazel` does version selection as expected.
if [[ -n "${USE_BAZEL_VERSION}" && $(bazel --version) != *${USE_BAZEL_VERSION}* ]]; then
  echo "Mismatch of configured and actual bazel version (see logged [[ command)"
  exit 1
fi

bazel clean

if [[ "$TEST_TF_NIGHTLY" == "true" ]]; then
  pip install --group test-nightly --progress-bar off --upgrade
else
  force_tensorflow_version
fi

python3 -m build --wheel
deactivate

# Start the test environment.
python3 -m venv test_venv
source test_venv/bin/activate

# Check the python version
python --version

# update pip
pip install --upgrade pip

if [[ "$TEST_TF_NIGHTLY" == "true" ]]; then
  pip install dist/tensorflow_gnn-*.whl
  pip uninstall -y tensorflow tf-keras ai-edge-litert
  pip install --group test-nightly --progress-bar off --upgrade

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

else
  force_tensorflow_version

  if [[ "$TF_USE_LEGACY_KERAS" == 1 ]]; then
    pip install --group test-tf216plus --progress-bar off --upgrade
  else
    pip install --group test-pre-tf216 --progress-bar off --upgrade
  fi

  pip install dist/tensorflow_gnn-*.whl
fi

echo "Final packages after all pip commands:"
pip list

bazel test --test_env=TF_USE_LEGACY_KERAS --build_tag_filters="${tag_filters}" --test_tag_filters="${tag_filters}" --test_output=errors --verbose_failures=true --build_tests_only --define=no_tfgnn_py_deps=true --keep_going --experimental_repo_remote_exec //bazel_pip/tensorflow_gnn/...
