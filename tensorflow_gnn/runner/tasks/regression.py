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
"""Regression tasks."""
import abc
from typing import Callable, Optional, Sequence

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces

AUTO = tf.keras.losses.Reduction.AUTO


class _Regression(interfaces.Task):
  """Regression abstract class.

  Any subclass must implement both `gather_activations` and `losses`, usually
  by inheriting from the below mix ins.
  """

  def __init__(self, units: int):
    self._units = units

  @abc.abstractmethod
  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    raise NotImplementedError()

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Append a linear head with `units` output units.

    Multiple `tf.keras.Model` outputs are not supported.

    Args:
      model: A `tf.keras.Model` to adapt.

    Returns:
      An adapted `tf.keras.Model.`
    """
    tfgnn.check_scalar_graph_tensor(model.output, name="Regression")
    activations = self.gather_activations(model.output)
    logits = tf.keras.layers.Dense(
        self._units,
        name="logits")(activations)  # Name seen in SignatureDef.
    return tf.keras.Model(model.input, logits)

  def preprocess(self, gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    return gt

  @abc.abstractmethod
  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    raise NotImplementedError()

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Regression metrics."""
    return (tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredLogarithmicError(),
            tf.keras.metrics.MeanAbsolutePercentageError())


class _GraphRegression(_Regression):
  """Graph context regression abstract class."""

  def __init__(self,
               units: int = 1,
               *,
               node_set_name: str,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean"):
    super().__init__(units)
    self._node_set_name = node_set_name
    self._state_name = state_name
    self._reduce_type = reduce_type

  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    return tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT,
        self._reduce_type,
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(gt)


class _RootNodeRegression(_Regression):
  """Root node regression abstract class."""

  def __init__(self,
               units: int = 1,
               *,
               node_set_name: str,
               state_name: str = tfgnn.HIDDEN_STATE):
    super().__init__(units)
    self._node_set_name = node_set_name
    self._state_name = state_name

  def gather_activations(self, gt: tfgnn.GraphTensor) -> tf.Tensor:
    return tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(gt)


class _MeanAbsoluteErrorLossMixIn:
  """Mean absolute error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanAbsoluteError(),)


class _MeanAbsolutePercentageErrorLossMixIn:
  """Mean absolute percentage error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanAbsolutePercentageError(),)


class _MeanSquaredErrorLossMixIn:
  """Mean squared error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanSquaredError(),)


class _MeanSquaredLogarithmicErrorLossMixIn:
  """Mean squared logarithmic error task."""

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanSquaredLogarithmicError(),)


class _MeanSquaredLogScaledError(tf.keras.losses.Loss):
  """Mean squared log scaled error task, see: go/xtzqv."""

  def __init__(self,
               reduction: tf.keras.losses.Reduction,
               name: str,
               *,
               alpha_loss_param: float,
               epsilon_loss_param: float):
    super().__init__(reduction, name)
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
    config = super().get_config()
    config.update({
        "alpha_loss_param": self._alpha_loss_param,
        "epsilon_loss_param": self._epsilon_loss_param
    })
    return config


class _MeanSquaredLogScaledErrorLossMixIn:
  """Mean squared log scaled error task."""

  def __init__(self,
               *args,
               alpha_loss_param: float = 5.,
               epsilon_loss_param: float = 1e-8,
               reduction: tf.keras.losses.Reduction = AUTO,
               name: Optional[str] = None,
               **kwargs):
    super().__init__(*args, **kwargs)
    self._alpha_loss_param = alpha_loss_param
    self._epsilon_loss_param = epsilon_loss_param
    self._reduction = reduction
    self._name = name

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (_MeanSquaredLogScaledError(
        self._reduction,
        self._name,
        alpha_loss_param=self._alpha_loss_param,
        epsilon_loss_param=self._epsilon_loss_param),)


class GraphMeanAbsoluteError(_MeanAbsoluteErrorLossMixIn, _GraphRegression):
  pass


class GraphMeanAbsolutePercentageError(_MeanAbsolutePercentageErrorLossMixIn,
                                       _GraphRegression):
  pass


class GraphMeanSquaredError(_MeanSquaredErrorLossMixIn, _GraphRegression):
  pass


class GraphMeanSquaredLogarithmicError(_MeanSquaredLogarithmicErrorLossMixIn,
                                       _GraphRegression):
  pass


class GraphMeanSquaredLogScaledError(_MeanSquaredLogScaledErrorLossMixIn,
                                     _GraphRegression):
  pass


class RootNodeMeanAbsoluteError(_MeanAbsoluteErrorLossMixIn,
                                _RootNodeRegression):
  pass


class RootNodeMeanAbsolutePercentageError(_MeanAbsolutePercentageErrorLossMixIn,
                                          _RootNodeRegression):
  pass


class RootNodeMeanSquaredError(_MeanSquaredErrorLossMixIn, _RootNodeRegression):
  pass


class RootNodeMeanSquaredLogarithmicError(_MeanSquaredLogarithmicErrorLossMixIn,
                                          _RootNodeRegression):
  pass


class RootNodeMeanSquaredLogScaledError(_MeanSquaredLogScaledErrorLossMixIn,
                                        _RootNodeRegression):
  pass
