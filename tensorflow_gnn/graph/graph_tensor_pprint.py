# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Routines to pretty-print the contents of eager GraphTensor instances."""

from typing import Any, Dict

import numpy

from tensorflow_gnn.graph import graph_tensor as gt


# NOTE(blais): Recursive type annotations not supported yet so we use Any.
def _get_tensor_data(listorarray: Any) -> Any:
  """Convert a tensor to plain-old data types."""
  if isinstance(listorarray, (int, float, str)):
    return listorarray
  elif isinstance(listorarray, bytes):
    try:
      return listorarray.decode('utf8')
    except UnicodeDecodeError:
      # Note: This may be useful for images and other non-text binary data.
      return listorarray
  elif isinstance(listorarray, list):
    return [_get_tensor_data(elem) for elem in listorarray]
  elif isinstance(listorarray, numpy.ndarray):
    return [_get_tensor_data(elem) for elem in listorarray.tolist()]
  else:
    raise TypeError(f'Unsupported type: {listorarray}')


def _get_features_data(features: gt.Fields) -> Dict[str, Any]:
  """Convert a tensor to plain-old data types."""
  return {name: _get_tensor_data(tensor.numpy())
          for name, tensor in features.items()}


def graph_tensor_to_values(graph: gt.GraphTensor) -> Dict[str, Any]:
  """Convert an eager `GraphTensor` to a mapping of mappings of PODTs.

  This is used for pretty-printing. Convert your graph tensor with this and run
  the result through `pprint.pprint()` or `pprint.pformat()` for display of its
  contents.

  Args:
    graph: An eager `GraphTensor` instance to be pprinted.

  Returns:
    A dict of plain-old data types that can be run through `pprint.pprint()` or
    a JSON conversion library.
  """
  context_data = _get_features_data(graph.context.features)
  node_data = {name: _get_features_data(nodeset.features)
               for name, nodeset in graph.node_sets.items()}
  edge_data = {name: _get_features_data(edgeset.features)
               for name, edgeset in graph.edge_sets.items()}
  return {'context': context_data,
          'node_sets': node_data,
          'edge_sets': edge_data}
