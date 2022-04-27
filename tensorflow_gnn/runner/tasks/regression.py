"""Regression tasks."""
import abc
from typing import Callable, Sequence

import tensorflow as tf
import tensorflow_gnn as tfgnn


class Regression(abc.ABC):
  """Regression abstract class."""

  def __init__(self, units: int):
    self._units = units

  @abc.abstractmethod
  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    raise NotImplementedError()

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Append a linear head with `self._units` units.

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
        self._units,
        name="logits")(activations)  # Name seen in SignatureDef.
    return tf.keras.Model(model.inputs, logits)

  def preprocessors(self) -> Sequence[Callable[..., tf.data.Dataset]]:
    return tuple()

  @abc.abstractmethod
  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    raise NotImplementedError()

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Regression metrics."""
    return (tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredLogarithmicError(),
            tf.keras.metrics.MeanAbsolutePercentageError())


class GraphRegression(Regression):
  """Graph context regression abstract class."""

  def __init__(self,
               units: int = 1,
               *,
               node_set_name: str,
               state_name: str = tfgnn.DEFAULT_STATE_NAME,
               reduce_type: str = "mean"):
    super(GraphRegression, self).__init__(units)
    self._node_set_name = node_set_name
    self._state_name = state_name
    self._reduce_type = reduce_type

  def gather_activations(self, gt: tfgnn.GraphTensor) -> tfgnn.Field:
    tfgnn.check_scalar_graph_tensor(gt, name="GraphRegression")
    return tfgnn.pool_nodes_to_context(
        gt,
        self._node_set_name,
        self._reduce_type,
        feature_name=self._state_name)


class GraphMeanAbsoluteError(GraphRegression):
  """Graph context mean absolute error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanAbsoluteError(),)


class GraphMeanAbsolutePercentageError(GraphRegression):
  """Graph context mean absolute percentage error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanAbsolutePercentageError(),)


class RootNodeRegression(Regression):
  """Root node regression abstract class."""

  def __init__(self,
               units: int = 1,
               *,
               node_set_name: str,
               state_name: str = tfgnn.DEFAULT_STATE_NAME):
    super(RootNodeRegression, self).__init__(units)
    self._node_set_name = node_set_name
    self._state_name = state_name

  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    return tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(gt)


class RootNodeMeanAbsoluteError(RootNodeRegression):
  """Root node mean asbsolute error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanAbsoluteError(),)


class RootNodeMeanSquaredError(RootNodeRegression):
  """Root node mean squared error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanSquaredError(),)


class RootNodeMeanSquaredLogarithmicError(RootNodeRegression):
  """Root node mean squared logarithmic error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanSquaredLogarithmicError(),)


class RootNodeMeanSquaredLogScaledError(RootNodeRegression):
  """Root node mean squared log scaled error task, see: go/xtzqv."""

  def __init__(self,
               units: int = 1,
               *,
               node_set_name: str,
               state_name: str = tfgnn.DEFAULT_STATE_NAME,
               alpha_loss_param: float = 5.,
               epsilon_loss_param: float = 1e-8):
    super(RootNodeMeanSquaredLogScaledError, self).__init__(
        units,
        node_set_name=node_set_name,
        state_name=state_name,)
    self._alpha_loss_param = alpha_loss_param
    self._epsilon_loss_param = epsilon_loss_param

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (_MeanSquaredLogScaledError(
        self._alpha_loss_param,
        self._epsilon_loss_param),)


class _MeanSquaredLogScaledError(tf.keras.losses.Loss):
  """Mean squared log-scaled error as described in: go/xtzqv."""

  def __init__(self,
               alpha_loss_param: float,
               epsilon_loss_param: float,
               *args,
               **kwargs):
    super(_MeanSquaredLogScaledError, self).__init__(*args, **kwargs)
    self._alpha_loss_param = alpha_loss_param
    self._epsilon_loss_param = epsilon_loss_param

  def call(self, y_true, y_pred):
    """See tf.keras.losses.Loss."""
    y_pred = tf.cast(tf.keras.activations.relu(y_pred), tf.float64)
    y_true = tf.cast(y_true, tf.float64)

    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    msle = tf.math.reduce_mean(
        tf.math.squared_difference(
            tf.math.log(y_pred + self._epsilon_loss_param),
            tf.math.log(y_true + self._epsilon_loss_param)))

    mse = tf.debugging.check_numerics(mse, "mse")
    msle = tf.debugging.check_numerics(msle, "msle")

    return mse + self._alpha_loss_param * msle

  def get_config(self):
    config = super(_MeanSquaredLogScaledError, self).get_config()
    config.update({
        "alpha_loss_param": self._alpha_loss_param,
        "epsilon_loss_param": self._epsilon_loss_param
    })
    return config
