"""KerasTensor specializations for GraphTensor pieces."""
from keras.engine import keras_tensor as kt
from keras.layers import core
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt


class GraphPieceKerasTensor(kt.KerasTensor):
  """Base class for graph pieces Keras tensors.

  Each graph piece (e.g. Context, NodeSet, etc.) must define corresponding Keras
  tensor and register them with the `register_keras_tensor_specialization`.

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

for cls, gt_properties in [
    (EdgeSetKerasTensor, ('features', 'sizes', 'adjacency', 'total_size',
                          '_get_features_ref', '_data')),
    (NodeSetKerasTensor, ('features', 'sizes', 'total_size',
                          '_get_features_ref', '_data')),
    (ContextKerasTensor, ('features', '_get_features_ref', '_data')),
    (GraphKerasTensor, ('node_sets', 'edge_sets', 'context',
                        'total_num_components', '_data')),
    (HyperAdjacencyKerasTensor, ('_data',)),
    (AdjacencyKerasTensor, ('source', 'target'))
]:
  for gt_property in gt_properties:
    core._delegate_property(cls, gt_property)

for cls, gt_methods in [
    (EdgeSetKerasTensor, ('get_features_dict', 'replace_features')),
    (NodeSetKerasTensor, ('get_features_dict', 'replace_features')),
    (ContextKerasTensor, ('get_features_dict', 'replace_features')),
    (GraphKerasTensor, ('replace_features', 'merge_batch_to_components')),
    (HyperAdjacencyKerasTensor, ('get_indices_dict',))
]:
  for gt_method in gt_methods:
    core._delegate_method(cls, gt_method)

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
#     core.TFClassMethodDispatcher(cls, class_method).register(
#         getattr(cls, class_method))

kt.register_keras_tensor_specialization(adj.Adjacency, AdjacencyKerasTensor)
kt.register_keras_tensor_specialization(adj.HyperAdjacency,
                                        HyperAdjacencyKerasTensor)

kt.register_keras_tensor_specialization(gt.GraphTensor, GraphKerasTensor)
kt.register_keras_tensor_specialization(gt.Context, ContextKerasTensor)
kt.register_keras_tensor_specialization(gt.NodeSet, NodeSetKerasTensor)
kt.register_keras_tensor_specialization(gt.EdgeSet, EdgeSetKerasTensor)
