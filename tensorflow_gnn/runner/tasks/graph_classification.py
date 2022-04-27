"""Tasks for graph-level classification."""
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner.tasks import classification


class GraphMulticlassClassification(classification.MulticlassClassification):
  """Multiclass classification by the label provided in the graph context."""

  def __init__(self,
               num_classes: int,
               *,
               node_set_name: str,
               state_name: str = tfgnn.DEFAULT_STATE_NAME):
    """Multiclass classification by the label provided in the graph context.

    This task defines classification of a graph by a label found in its unique
    context. The representations are mean-pooled from a single set of nodes.

    Labels are expected to be sparse, i.e.: scalars.

    Args:
      num_classes: The number of target classes.
      node_set_name: Node set to pool representations from.
      state_name: The feature name for node activations
          (typically: tfgnn.DEFAULT_STATE_NAME).
    """
    super(GraphMulticlassClassification, self).__init__(num_classes)
    self._node_set_name = node_set_name
    self._state_name = state_name

  def gather_activations(self, gt: tfgnn.GraphTensor):
    if gt.rank != 0:
      raise ValueError(
          f"Expected a scalar (rank=0) GraphTensor, received rank={gt.rank}")
    return tfgnn.pool_nodes_to_context(
        gt, self._node_set_name, "mean", feature_name=self._state_name)
