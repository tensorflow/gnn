# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""The protocol message (protobuf) types defined by TF-GNN Sampler."""

from tensorflow_gnn.experimental.sampler.proto import eval_dag_pb2
from tensorflow_gnn.utils import api_utils


# Program computation DAG, its stages and layers.
Program = eval_dag_pb2.Program
EvalDAG = eval_dag_pb2.EvalDAG
Stage = eval_dag_pb2.Stage
Layer = eval_dag_pb2.Layer

# Specifications of input/output values of layers.
ValueSpec = eval_dag_pb2.ValueSpec
TensorSpec = eval_dag_pb2.TensorSpec
RaggedTensorSpec = eval_dag_pb2.RaggedTensorSpec
FlattenedSpec = eval_dag_pb2.FlattenedSpec

# Layer configs.
EdgeSamplingConfig = eval_dag_pb2.EdgeSamplingConfig
IOFeatures = eval_dag_pb2.IOFeatures


# Remove all names added by module imports, unless explicitly allowed here.
api_utils.remove_submodules_except(__name__, [])
# LINT.ThenChange()../api_def/sampler-symbols.txt)
