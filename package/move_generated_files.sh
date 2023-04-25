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

# Moves the bazel generated files needed for packaging the wheel to the source
# tree.

function _is_windows() {
  [[ "$(uname -s | tr 'A-Z' 'a-z')" =~ (cygwin|mingw32|mingw64|msys)_nt* ]]
}

function tfgnn::move_generated_files() {
  if _is_windows; then
    # See https://github.com/bazelbuild/bazel/issues/6761 for bazel-bin.
    GENFILES=${BUILD_WORKSPACE_DIRECTORY}/bazel-genfiles
    if [[ ! -d ${GENFILES} ]]; then
      GENFILES=${BUILD_WORKSPACE_DIRECTORY}/bazel-bin
    fi
  else
    # If run by "bazel run", $(pwd) is the .runfiles dir that contains all the
    # data dependencies.
    GENFILES=$(pwd)
  fi

  FILES="
    tensorflow_gnn/experimental/sampler/eval_dag_pb2.py
    tensorflow_gnn/proto/graph_schema_pb2.py
    tensorflow_gnn/proto/examples_pb2.py
    tensorflow_gnn/sampler/sampling_spec_pb2.py
    tensorflow_gnn/sampler/subgraph_pb2.py
    tensorflow_gnn/tools/sampled_stats_pb2.py
  "
  for FILE in ${FILES}; do
    cp -f ${GENFILES}/${FILE} ${BUILD_WORKSPACE_DIRECTORY}/${FILE}
  done
}

tfgnn::move_generated_files
