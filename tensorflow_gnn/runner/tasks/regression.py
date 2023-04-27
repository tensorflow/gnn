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
from __future__ import annotations

import abc
from typing import Callable, Optional, Sequence, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces

AUTO = tf.keras.losses.Reduction.AUTO
Field = tfgnn.Field
GraphTensor = tfgnn.GraphTensor
# TODO(b/274672364): make this tuple[...] in Python 3.9 style
# when we drop py38 support.
LabelFn = Callable[[GraphTensor], Tuple[GraphTensor, Field]]


class _Regression(interfaces.Task):
  """Regression abstract class.

  Any subclass must implement both `gather_activations` and `losses`, usually
  by inheriting from the below mix ins.
  """

  def __init__(
      self,
      units: int,
      *,
      name: str = "regression_logits",
      label_fn: Optional[LabelFn] = None,
      label_feature_name: Optional[str] = None):
    """Sets `Task` parameters.

    Args:
      units: The units for the regression head.
      name: The regression head's layer name. This name typically appears in
        the exported model's SignatureDef.
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    if (label_fn is None) == (label_feature_name is None):
      raise ValueError(
          "Exactly one of `label_fn` or `label_feature_name` may be specified"
          f" (got label_fn={label_fn} and"
          f" label_feature_name={label_feature_name})"
      )
    self._units = units
    self._name = name
    self._label_fn = label_fn
    self._label_feature_name = label_feature_name

  @abc.abstractmethod
  def gather_activations(self, inputs: GraphTensor) -> Field:
    raise NotImplementedError()

  def predict(self, inputs: tfgnn.GraphTensor) -> tf.Tensor:
    """Apply a linear head for regression.

    Args:
      inputs: A `tfgnn.GraphTensor` for regression.

    Returns:
      The regression logits.
    """
    tfgnn.check_scalar_graph_tensor(inputs, name="_Regression")
    activations = self.gather_activations(inputs)
    logits = tf.keras.layers.Dense(
        self._units,
        name=self._name)(activations)  # Name seen in SignatureDef.
    return logits

  def preprocess(self, inputs: GraphTensor) -> tuple[GraphTensor, Field]:
    if self._label_fn is not None:
      return self._label_fn(inputs)
    x = inputs
    y = tfgnn.keras.layers.Readout(
        feature_name=self._label_feature_name,
        node_set_name="_readout")(inputs)
    return x, y

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
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               **kwargs):
    super().__init__(units, **kwargs)
    self._node_set_name = node_set_name
    self._state_name = state_name
    self._reduce_type = reduce_type

  def gather_activations(self, inputs: GraphTensor) -> tf.Tensor:
    return tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT,
        self._reduce_type,
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(inputs)


class _RootNodeRegression(_Regression):
  """Root node regression abstract class."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               **kwargs):
    super().__init__(units, **kwargs)
    self._node_set_name = node_set_name
    self._state_name = state_name

  def gather_activations(self, inputs: GraphTensor) -> tf.Tensor:
    return tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(inputs)


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


class MeanSquaredLogScaledError(tf.keras.losses.Loss):
  """Mean squared log scaled error task, see: go/xtzqv."""

  def __init__(self,
               reduction: tf.keras.losses.Reduction = AUTO,
               name: Optional[str] = None,
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


class MeanAbsoluteLogarithmicErrorLoss(tf.keras.losses.Loss):
  """Mean absolute log scaled error task."""

  def call(self, y_true, y_pred):
    return _mean_absolute_logarithmic_error(y_true, y_pred)


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
    return (MeanSquaredLogScaledError(
        self._reduction,
        self._name,
        alpha_loss_param=self._alpha_loss_param,
        epsilon_loss_param=self._epsilon_loss_param),)


def _mean_absolute_logarithmic_error(y_true, y_pred):
  """Computes the mean absolute logarithmic error between `y_true` and `y_pred`.

  loss = mean((log(y_true + 1) - log(y_pred + 1)), axis=-1)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean absolute logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tf.math.log1p(tf.convert_to_tensor(y_pred))
  y_true = tf.math.log1p(tf.cast(y_true, y_pred.dtype))
  return tf.math.reduce_mean(tf.abs(y_pred - y_true), axis=-1)


class _MeanAbsoluteLogarithmicErrorLossMixIn:
  """Mean absolute logarithmic error task."""

  def __init__(
      self,
      reduction: tf.keras.losses.Reduction = AUTO,
      name: Optional[str] = None,
      **kwargs):
    super().__init__(**kwargs)
    self._reduction = reduction
    self._name = name

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (MeanAbsoluteLogarithmicErrorLoss(self._reduction, self._name),)


class RootNodeMeanAbsoluteLogarithmicError(
    _MeanAbsoluteLogarithmicErrorLossMixIn, _RootNodeRegression
):
  """Root node mean absolute logarithmic error task."""

  def predict(self, inputs: tfgnn.GraphTensor) -> tf.Tensor:
    """Apply a head with ReLU for nonnegative regression.

    Args:
      inputs: A `tfgnn.GraphTensor` use for prediction.

    Returns:
      The nonnegative logits.
    """
    tfgnn.check_scalar_graph_tensor(
        inputs,
        name="RootNodeMeanAbsoluteLogarithmicError")
    activations = self.gather_activations(inputs)
    logits = tf.keras.layers.Dense(
        self._units,
        activation="relu",
        name="logits")(activations)  # Name seen in SignatureDef.
    return logits


# TODO(dzelle): Add an `__init__` with parameters and doc for all of the below.
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
