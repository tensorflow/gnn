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
"""KerasTensor specializations for GraphTensor pieces."""
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tf_internal


class GraphPieceKerasTensor(tf_internal.KerasTensor):
  """Base class for graph pieces Keras tensors.

  Each graph piece (e.g. `tfgnn.Context`, `tfgnn.NodeSet`, etc.) must define
  corresponding Keras tensor and register them with the
  `register_keras_tensor_specialization`.

  Keras tensors specialization is done according to the rules:
  1. Graph piece methods and properties that return static values (e.g. python
     scalars) must be explicitly mirrored in the corresponding Keras tensors
     See, for example, `rank` and  `indices_dtype` below.

  2. Methods and properties that return dynamic values (e.g. tf.Tensor) must be
     deligated with the `_delegate_method` and `_delegate_property`.

  3. Class methods (e.g. class factory methods) must be deligated with
     TFClassMethodDispatcher. NOTE: for deligation to work class methods are
     assumed to raise `TypeError` on unsupported arguments.
  """

  @property
  def rank(self):
    return self.shape.rank

  @property
  def indices_dtype(self):
    return self.spec.indices_dtype

  @property
  def spec(self):
    return self._type_spec


class GraphKerasTensor(GraphPieceKerasTensor):
  pass


class ContextKerasTensor(GraphPieceKerasTensor):
  pass


class NodeSetKerasTensor(GraphPieceKerasTensor):
  pass


class EdgeSetKerasTensor(GraphPieceKerasTensor):
  pass


class HyperAdjacencyKerasTensor(GraphPieceKerasTensor):

  def node_set_name(self,
                    node_set_tag: const.IncidentNodeTag) -> const.NodeSetName:
    return self.spec.node_set_name(node_set_tag)


class AdjacencyKerasTensor(HyperAdjacencyKerasTensor):

  @property
  def source_name(self) -> const.NodeSetName:
    return self.spec.source_name

  @property
  def target_name(self) -> const.NodeSetName:
    return self.spec.target_name


# pylint: disable=protected-access
ContextKerasTensor._overload_operator(gt.Context, '__getitem__')
NodeSetKerasTensor._overload_operator(gt.NodeSet, '__getitem__')
EdgeSetKerasTensor._overload_operator(gt.EdgeSet, '__getitem__')
HyperAdjacencyKerasTensor._overload_operator(adj.HyperAdjacency, '__getitem__')

GRAPH_PIECE_PROPERTIES = ('_data',)
GRAPH_PIECE_WITH_FEATURES_PROPERTIES = GRAPH_PIECE_PROPERTIES + (
    'features',
    '_get_features_ref',
    'sizes',
    'total_size',
    'num_components',
    'total_num_components',
)

for cls, gt_properties in [
    (EdgeSetKerasTensor, GRAPH_PIECE_WITH_FEATURES_PROPERTIES + ('adjacency',)),
    (NodeSetKerasTensor, GRAPH_PIECE_WITH_FEATURES_PROPERTIES),
    (ContextKerasTensor, GRAPH_PIECE_WITH_FEATURES_PROPERTIES),
    (GraphKerasTensor, (
        'node_sets',
        'edge_sets',
        'context',
        'num_components',
        'total_num_components',
    ) + GRAPH_PIECE_PROPERTIES),
    (HyperAdjacencyKerasTensor, GRAPH_PIECE_PROPERTIES),
    (AdjacencyKerasTensor, GRAPH_PIECE_PROPERTIES + ('source', 'target'))
]:
  for gt_property in gt_properties:
    tf_internal.delegate_property(cls, gt_property)

GRAPH_PIECE_WITH_FEATURES_METHODS = ('get_features_dict', 'replace_features')
for cls, gt_methods in [(EdgeSetKerasTensor, GRAPH_PIECE_WITH_FEATURES_METHODS),
                        (NodeSetKerasTensor, GRAPH_PIECE_WITH_FEATURES_METHODS),
                        (ContextKerasTensor, GRAPH_PIECE_WITH_FEATURES_METHODS),
                        (GraphKerasTensor, ('remove_features',
                                            'replace_features',
                                            'merge_batch_to_components')),
                        (HyperAdjacencyKerasTensor, ('get_indices_dict',))]:
  for gt_method in gt_methods:
    tf_internal.delegate_method(cls, gt_method)

# TODO(b/191957072): dispatch class methods.
# for cls, class_methods in [
#     (adj.Adjacency, ('from_indices',)),
#     (adj.HyperAdjacency, ('from_indices',)),
#     (gt.Context, ('from_fields',)),
#     (gt.NodeSet, ('from_fields',)),
#     (gt.EdgeSet, ('from_fields',)),
#     (gt.GraphTensor, ('from_pieces',)),
# ]:
#   for class_method in class_methods:
#     tf_internal.TFClassMethodDispatcher(cls, class_method).register(
#         getattr(cls, class_method))

tf_internal.register_keras_tensor_specialization(adj.Adjacency,
                                                 AdjacencyKerasTensor)
tf_internal.register_keras_tensor_specialization(adj.HyperAdjacency,
                                                 HyperAdjacencyKerasTensor)
tf_internal.register_keras_tensor_specialization(gt.GraphTensor,
                                                 GraphKerasTensor)
tf_internal.register_keras_tensor_specialization(gt.Context, ContextKerasTensor)
tf_internal.register_keras_tensor_specialization(gt.NodeSet, NodeSetKerasTensor)
tf_internal.register_keras_tensor_specialization(gt.EdgeSet, EdgeSetKerasTensor)
