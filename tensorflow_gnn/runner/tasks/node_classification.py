"""Tasks for node classification."""
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner.tasks import classification


class RootNodeMulticlassClassification(classification.MulticlassClassification):
  """Multiclass classification by root node label."""

  def __init__(self,
               num_classes: int,
               *,
               node_set_name: str,
               state_name: str = tfgnn.DEFAULT_STATE_NAME):
    """Multiclass classification by root node label.

    This task defines classification of a rooted graph by a label found
    on its unique root node. By convention, root nodes are stored as the
    first node of each graph component in the respective node set.
    Typically, such graphs are created by go/graph-sampler by sampling the
    neighborhoods of root nodes (or "seed nodes") in large graphs to
    aggregate information relevant to the root node's classification.

    Any labels are expected to be sparse, i.e.: scalars.

    Args:
      num_classes: The number of target classes.
      node_set_name: The node set containing the root node.
      state_name: The feature name for activations
        (typically: tfgnn.DEFAULT_STATE_NAME).
    """
    super(RootNodeMulticlassClassification, self).__init__(num_classes)
    self._node_set_name = node_set_name
    self._state_name = state_name

  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    """Gather activations from root nodes."""
    if gt.rank != 0:
      raise ValueError(
          f"Expected a scalar (rank=0) GraphTensor, received rank={gt.rank}")
    node_set = gt.node_sets[self._node_set_name]
    node_value = node_set.features[self._state_name]
    components_starts = tf.math.cumsum(node_set.sizes, exclusive=True)
    return tf.gather(node_value, components_starts)


class RootNodeBinaryClassification(classification.BinaryClassification):
  """Binary classification by root node label."""

  def __init__(self,
               *,
               node_set_name: str,
               state_name: str = tfgnn.DEFAULT_STATE_NAME):
    """Binary classification by root node label.

    This task defines classification of a rooted graph by a label found
    on its unique root node. By convention, root nodes are stored as the
    first node of each graph component in the respective node set.
    Typically, such graphs are created by go/graph-sampler by sampling the
    neighborhoods of root nodes (or "seed nodes") in large graphs to
    aggregate information relevant to the root node's classification.

    Any labels are expected to be binary i.e. 0s and 1s.

    Args:
      node_set_name: The node set containing the root node.
      state_name: The feature name for activations
        (typically: tfgnn.DEFAULT_STATE_NAME).
    """
    super(RootNodeBinaryClassification, self).__init__()
    self._node_set_name = node_set_name
    self._state_name = state_name

  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    """Gather activations from root nodes."""
    return tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(gt)
