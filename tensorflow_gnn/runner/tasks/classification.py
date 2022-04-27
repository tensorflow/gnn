"""Abstract classification tasks."""
import abc
from typing import Callable, Optional, Sequence

import tensorflow as tf
import tensorflow_gnn as tfgnn


class MulticlassClassification(abc.ABC):
  """Multiclass classification abstract class."""

  def __init__(self, num_classes: int):
    self._num_classes = num_classes

  @abc.abstractmethod
  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    raise NotImplementedError()

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Append a linear head with `num_classes` units.

    Multiple `tf.keras.Model` outputs are not supported.

    Args:
      model: A `tf.keras.Model` to adapt.

    Returns:
      An adapted `tf.keras.Model.`
    """
    if not tfgnn.is_graph_tensor(model.output):
      raise ValueError(f"Expected a GraphTensor, received {type(model.output)}")
    activations = self.gather_activations(model.output)
    # TODO(b/196880966): Add more flexibility — regularization, etc.
    logits = tf.keras.layers.Dense(
        self._num_classes,
        name="logits")(activations)  # Name seen in SignatureDef.
    return tf.keras.Model(model.inputs, logits)

  def preprocessors(self) -> Sequence[Callable[..., tf.data.Dataset]]:
    return tuple()

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Sparse categorical crossentropy loss."""
    return (tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Sparse categorical metrics."""
    return (tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True))


class FromLogitsMixin(tf.keras.metrics.Metric):
  """Mixin for `tf.keras.metrics.Metric` with a from_logits option."""

  def __init__(self, from_logits: bool, *args, **kwargs) -> None:
    super(FromLogitsMixin, self).__init__(*args, **kwargs)
    self._from_logits = from_logits

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> None:
    if self._from_logits:
      y_pred = tf.nn.sigmoid(y_pred)
    return super(FromLogitsMixin, self).update_state(
        y_true,
        y_pred,
        sample_weight)


class Precision(FromLogitsMixin, tf.keras.metrics.Precision):
  pass


class Recall(FromLogitsMixin, tf.keras.metrics.Recall):
  pass


class BinaryClassification(abc.ABC):
  """Binary classification abstract class."""

  @abc.abstractmethod
  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    raise NotImplementedError()

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Append a linear head with 1 unit.

    Multiple `tf.keras.Model` outputs are not supported.

    Args:
      model: A `tf.keras.Model` to adapt.

    Returns:
      An adapted `tf.keras.Model.`
    """
    if not tfgnn.is_graph_tensor(model.output):
      raise ValueError(f"Expected a GraphTensor, received {type(model.output)}")
    activations = self.gather_activations(model.output)
    logits = tf.keras.layers.Dense(
        1,
        name="logits")(activations)  # Name seen in SignatureDef.
    return tf.keras.Model(model.inputs, logits)

  def preprocessors(self) -> Sequence[Callable[..., tf.data.Dataset]]:
    return tuple()

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.BinaryCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.metrics.AUC(from_logits=True),
            Precision(from_logits=True),
            Recall(from_logits=True),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True))
