"""Classification tasks."""
import abc
from typing import Callable, Optional, Sequence, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn

Tensor = Union[tf.Tensor, tf.RaggedTensor]


class _FromLogitsMixIn(tf.keras.metrics.Metric):
  """Mixin for `tf.keras.metrics.Metric` with a from_logits option."""

  def __init__(self, from_logits: bool, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._from_logits = from_logits

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> None:
    return super().update_state(
        y_true,
        tf.nn.sigmoid(y_pred) if self._from_logits else y_pred,
        sample_weight)


class _Precision(_FromLogitsMixIn, tf.keras.metrics.Precision):
  pass


class _Recall(_FromLogitsMixIn, tf.keras.metrics.Recall):
  pass


class _Classification(abc.ABC):
  """Abstract classification class.

  Any subclass must implement all of `gather_activations`, `losses` and
  `metrics`, usually by inheriting from the classes below.
  """

  def __init__(self, units: int):
    self._units = units

  @abc.abstractmethod
  def gather_activations(self, gt: tfgnn.GraphTensor) ->  Tensor:
    raise NotImplementedError()

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Append a linear head with `units` output units.

    Multiple `tf.keras.Model` outputs are not supported.

    Args:
      model: A `tf.keras.Model` to adapt.

    Returns:
      An adapted `tf.keras.Model.`
    """
    tfgnn.check_scalar_graph_tensor(model.output, name="Classification")
    activations = self.gather_activations(model.output)
    logits = tf.keras.layers.Dense(
        self._units,
        name="logits")(activations)  # Name seen in SignatureDef.
    return tf.keras.Model(model.inputs, logits)

  def preprocessors(self) -> Sequence[Callable[..., tf.data.Dataset]]:
    return tuple()

  @abc.abstractmethod
  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    raise NotImplementedError()

  @abc.abstractmethod
  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    raise NotImplementedError()


class _BinaryClassification(_Classification):
  """Binary classification."""

  def __init__(self, *args, units: int = 1, **kwargs):
    super().__init__(units, *args, **kwargs)

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.BinaryCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (_Precision(from_logits=True),
            _Recall(from_logits=True),
            tf.keras.metrics.AUC(from_logits=True, name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", from_logits=True, name="auc_pr"),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True))


class _MulticlassClassification(_Classification):
  """Multiclass classification."""

  def __init__(self, num_classes: int, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super().__init__(num_classes, *args, **kwargs)

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Sparse categorical crossentropy loss."""
    return (tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Sparse categorical metrics."""
    return (tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True))


class _GraphClassification(_Classification):
  """Classification by the label provided in the graph context."""

  def __init__(self,
               *args,
               node_set_name: str,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               **kwargs):
    """Classification by the label provided in the graph context.

    This task defines classification of a graph by a label found in its unique
    context. The representations are mean-pooled from a single set of nodes.

    Labels are expected to be sparse, i.e.: scalars.

    Args:
      *args: Additional positional arguments.
      node_set_name: Node set to pool representations from.
      state_name: The feature name for node activations
        (typically: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(*args, **kwargs)
    self._node_set_name = node_set_name
    self._state_name = state_name
    self._reduce_type = reduce_type

  def gather_activations(self, gt: tfgnn.GraphTensor) ->  Tensor:
    return tfgnn.pool_nodes_to_context(
        gt,
        self._node_set_name,
        self._reduce_type,
        feature_name=self._state_name)


class _RootNodeClassification(_Classification):
  """Classification by root node label."""

  def __init__(self,
               *args,
               node_set_name: str,
               state_name: str = tfgnn.HIDDEN_STATE,
               **kwargs):
    """Classification by root node label.

    This task defines classification of a rooted graph by a label found
    on its unique root node. By convention, root nodes are stored as the
    first node of each graph component in the respective node set.
    Typically, such graphs are created by go/graph-sampler by sampling the
    neighborhoods of root nodes (or "seed nodes") in large graphs to
    aggregate information relevant to the root node's classification.

    Any labels are expected to be sparse, i.e.: scalars.

    Args:
      *args: Additional positional arguments.
      node_set_name: The node set containing the root node.
      state_name: The feature name for activations
        (typically: tfgnn.HIDDEN_STATE).
      **kwargs: Additional keyword arguments.
    """
    super().__init__(*args, **kwargs)
    self._node_set_name = node_set_name
    self._state_name = state_name

  def gather_activations(self, gt: tfgnn.GraphTensor) ->  Tensor:
    """Gather activations from root nodes."""
    return tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(gt)


class GraphBinaryClassification(_GraphClassification, _BinaryClassification):
  pass


class GraphMulticlassClassification(_GraphClassification,
                                    _MulticlassClassification):
  pass


class RootNodeBinaryClassification(_RootNodeClassification,
                                   _BinaryClassification):
  pass


class RootNodeMulticlassClassification(_RootNodeClassification,
                                       _MulticlassClassification):
  pass
