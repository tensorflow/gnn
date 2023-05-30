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

cd "${KOKORO_ARTIFACTS_DIR}/github/gnn/"

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

PYTHON_BINARY="/usr/bin/python3.9"
PIP_TEST_PREFIX=bazel_pip

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate

# Check the python version
python --version
python3 --version

TEST_ROOT=$(pwd)/${PIP_TEST_PREFIX}
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"
ln -s "$(pwd)"/tensorflow_gnn "$TEST_ROOT"/tensorflow_gnn
tag_filters="-no_oss,-oss_excluded"

bazel clean
pip install --upgrade pip
pip install -r requirements-dev.txt --progress-bar off
pip install tf-nightly --progress-bar off --upgrade
# We need to remove the dependency on tensorflow to test nightly
# The dependencies will be provided by tf-nightly
perl -i  -lpe '$k+= s/tensorflow>=2\.[0-9]+\.[0-9]+/tf-nightly/g; END{exit($k != 1)}' setup.py
python3 setup.py bdist_wheel
pip uninstall -y tensorflow_gnn
pip install dist/tensorflow_gnn-*.whl
# Check that tf-nightly is installed but tensorflow is not
pip freeze | grep -q tf-nightly= && ! pip freeze | grep -q tensorflow=
bazel test --build_tag_filters="${tag_filters}" --test_tag_filters="${tag_filters}" --test_output=errors --verbose_failures=true --build_tests_only --define=no_tfgnn_py_deps=true --keep_going --experimental_repo_remote_exec //bazel_pip/tensorflow_gnn/...
