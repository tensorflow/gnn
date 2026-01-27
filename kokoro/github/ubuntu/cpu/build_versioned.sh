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

# --- START Bazel Version Management ---
USE_BAZEL_VERSION=${USE_BAZEL_VERSION:-"7.4.1"}
BAZEL_INSTALL_DIR="/tmp/bazel/${USE_BAZEL_VERSION}"
BAZEL_BIN="${BAZEL_INSTALL_DIR}/bin/bazel"

if ! [[ -x "${BAZEL_BIN}" && $(${BAZEL_BIN} --version) == *${USE_BAZEL_VERSION}* ]]; then
  echo "Bazel ${USE_BAZEL_VERSION} not found or version mismatch. Installing..."
  mkdir -p "${BAZEL_INSTALL_DIR}/bin"
  curl -fLo "${BAZEL_INSTALL_DIR}/bin/bazel" "https://releases.bazel.build/${USE_BAZEL_VERSION}/release/bazel-${USE_BAZEL_VERSION}-linux-x86_64"
  chmod +x "${BAZEL_INSTALL_DIR}/bin/bazel"
  echo "Bazel ${USE_BAZEL_VERSION} installed."
else
  echo "Using cached Bazel ${USE_BAZEL_VERSION}"
fi

export PATH="${BAZEL_INSTALL_DIR}/bin:${PATH}"
echo "Bazel version check:"
bazel --version
# --- END Bazel Version Management ---

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
VENV_DIR="venv"

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Debug messages to indicate the python version
python --version
python3 --version

# update pip
pip install --upgrade pip

TEST_ROOT=$(pwd)/${PIP_TEST_PREFIX}
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"

# --- START Fixed Symlink ---
# Create a relative symlink.
# From $(pwd)/bazel_pip, tensorflow_gnn is one level up.
ln -s ../tensorflow_gnn "${TEST_ROOT}"/tensorflow_gnn
echo "Created symlink:"
ls -l "${TEST_ROOT}"/tensorflow_gnn
# --- END Fixed Symlink ---

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
pip install -r requirements-dev.txt --progress-bar off
pip install tensorflow=="${TF_VERSION}" --progress-bar off --upgrade
if [[ "$TF_USE_LEGACY_KERAS" == 1 ]]; then
  pip install tf-keras=="${TF_VERSION}" --progress-bar off --upgrade
fi
python3 setup.py bdist_wheel
pip uninstall -y tensorflow_gnn
pip install dist/tensorflow_gnn-*.whl

echo "Final packages after all pip commands:"
pip list

# Deactivate venv before running tests to avoid issues with test environments
deactivate

# Run tests
bazel test --test_env=TF_USE_LEGACY_KERAS --build_tag_filters="${tag_filters}" --test_tag_filters="${tag_filters}" --test_output=errors --verbose_failures=true --build_tests_only --define=no_tfgnn_py_deps=true --keep_going --experimental_repo_remote_exec //bazel_pip/tensorflow_gnn/...

# --- START Cleanup Symlinks and venv ---
echo "Removing Bazel-generated symlinks..."
rm -f bazel-bin bazel-out bazel-genfiles bazel-testlogs bazel-gnn

echo "Removing venv directory..."
rm -rf "${VENV_DIR}"
# --- END Cleanup Symlinks and venv ---
