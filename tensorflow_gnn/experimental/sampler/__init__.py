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
"""Public interface for GNN Sampler."""

from tensorflow_gnn.experimental.sampler import core
from tensorflow_gnn.experimental.sampler import eval_dag
from tensorflow_gnn.experimental.sampler import ext_ops
from tensorflow_gnn.experimental.sampler import interfaces
from tensorflow_gnn.experimental.sampler import proto  # Exposed as submodule. pylint: disable=unused-import
from tensorflow_gnn.experimental.sampler import subgraph_pipeline
from tensorflow_gnn.utils import api_utils

# NOTE: This package is covered by tensorflow_gnn/api_def/api_symbols_test.py.
# Please see there for instructions how to reflect API changes.
# LINT.IfChange

# Helpers.
set_ext_ops_implementation = ext_ops.set_ops_implementation
ragged_lookup = ext_ops.ragged_lookup
ragged_unique = ext_ops.ragged_unique
ragged_choice = ext_ops.ragged_choice

build_graph_tensor = core.build_graph_tensor

create_sampling_model_from_spec = (
    subgraph_pipeline.create_sampling_model_from_spec
)

create_link_sampling_model_from_spec = (
    subgraph_pipeline.create_link_sampling_model_from_spec
)

# Export.
create_program = eval_dag.create_program
save_model = eval_dag.save_model
Artifacts = eval_dag.Artifacts

# Sampling layers.
InMemUniformEdgesSampler = core.InMemUniformEdgesSampler
InMemIndexToFeaturesAccessor = core.InMemIndexToFeaturesAccessor
InMemIntegerKeyToBytesAccessor = core.InMemIntegerKeyToBytesAccessor
InMemStringKeyToBytesAccessor = core.InMemStringKeyToBytesAccessor
KeyToTfExampleAccessor = core.KeyToTfExampleAccessor
TfExamplesParser = core.TfExamplesParser
UniformEdgesSampler = core.UniformEdgesSampler
CompositeLayer = core.CompositeLayer

# Interfaces.
ConnectingEdgesSampler = interfaces.ConnectingEdgesSampler
OutgoingEdgesSampler = interfaces.OutgoingEdgesSampler
KeyToFeaturesAccessor = interfaces.KeyToFeaturesAccessor
KeyToBytesAccessor = interfaces.KeyToBytesAccessor

# Remove all names added by module imports, unless explicitly allowed here.
api_utils.remove_submodules_except(__name__, ['proto'])
# LINT.ThenChange(../../api_def/sampler-symbols.txt)
