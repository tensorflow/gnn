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
from typing import Callable, Optional, Tuple

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
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
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

  def predict(self, inputs: tfgnn.GraphTensor) -> interfaces.Predictions:
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
        name=self._name)(activations)
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
  def losses(self) -> interfaces.Losses:
    raise NotImplementedError()

  def metrics(self) -> interfaces.Metrics:
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


class _NodeRegression(_Regression):
  """Node regression (via structured readout) abstract class."""

  def __init__(self,
               key: str = "seed",
               *,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True,
               **kwargs):
    """Regression of node(s) via structured readout.

    This task defines regression via structured readout (see
    `tfgnn.keras.layers.StructuredReadout`).  Structured readout addresses the
    need to read out final hidden states from a GNN computation to make
    predictions for some nodes (or edges) of interest. To add auxiliary node
    (and edge) sets for structured readout see, e.g.:
    `tfgnn.keras.layers.AddReadoutFromFirstNode`.

    Args:
      key: A string key to select between possibly multiple named readouts.
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self._key = key
    self._feature_name = feature_name
    self._readout_node_set = readout_node_set
    self._validate = validate

  def gather_activations(self, inputs: GraphTensor) -> Field:
    """Gather activations from auxiliary node (and edge) sets."""
    try:
      return tfgnn.keras.layers.StructuredReadout(
          self._key,
          feature_name=self._feature_name,
          readout_node_set=self._readout_node_set,
          validate=self._validate)(inputs)
    except (KeyError, ValueError) as e:
      raise ValueError(
          "This NodeRegression task failed in StructuredReadout("
          f"{self._key}, feature_name={self._feature_name}, "
          f"readout_node_set={self._readout_node_set}).\n"
          "For a dataset of sampled subgraphs that does not provide a readout "
          "structure but follows the conventional placement of root nodes "
          "first in their node set, consider using a RootNodeClassification "
          "task or tfgnn.keras.layers.AddReadoutFromFirstNode."
      ) from e


class _MeanAbsoluteErrorLossMixIn:
  """Mean absolute error task."""

  def losses(self) -> interfaces.Losses:
    return tf.keras.losses.MeanAbsoluteError()


class _MeanAbsolutePercentageErrorLossMixIn:
  """Mean absolute percentage error task."""

  def losses(self) -> interfaces.Losses:
    return tf.keras.losses.MeanAbsolutePercentageError()


class _MeanSquaredErrorLossMixIn:
  """Mean squared error task."""

  def losses(self) -> interfaces.Losses:
    return tf.keras.losses.MeanSquaredError()


class _MeanSquaredLogarithmicErrorLossMixIn:
  """Mean squared logarithmic error task."""

  def losses(self) -> interfaces.Losses:
    return tf.keras.losses.MeanSquaredLogarithmicError()


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

  def losses(self) -> interfaces.Losses:
    return MeanSquaredLogScaledError(
        self._reduction,
        self._name,
        alpha_loss_param=self._alpha_loss_param,
        epsilon_loss_param=self._epsilon_loss_param,
    )


class GraphMeanAbsoluteError(_MeanAbsoluteErrorLossMixIn, _GraphRegression):
  """Regression from pooled node states with mean absolute error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Regression from pooled node states with mean absolute error.

    Args:
      node_set_name: The node set to pool.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        reduce_type=reduce_type,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class GraphMeanAbsolutePercentageError(_MeanAbsolutePercentageErrorLossMixIn,
                                       _GraphRegression):
  """Regression from pooled node states with mean absolute percentage error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Regression from pooled node states with mean absolute percentage error.

    Args:
      node_set_name: The node set to pool.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        reduce_type=reduce_type,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class GraphMeanSquaredError(_MeanSquaredErrorLossMixIn, _GraphRegression):
  """Regression from pooled node states with mean squared error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Regression from pooled node states with mean squared error.

    Args:
      node_set_name: The node set containing to pool.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        reduce_type=reduce_type,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class GraphMeanSquaredLogarithmicError(_MeanSquaredLogarithmicErrorLossMixIn,
                                       _GraphRegression):
  """Regression from pooled node states with mean squared logarithmic error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Regression from pooled node states with mean squared logarithmic error.

    Args:
      node_set_name: The node set to pool.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        reduce_type=reduce_type,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class GraphMeanSquaredLogScaledError(_MeanSquaredLogScaledErrorLossMixIn,
                                     _GraphRegression):
  """Regression from pooled node states with mean squared log scaled error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None,
               alpha_loss_param: float = 5.,
               epsilon_loss_param: float = 1e-8,
               reduction: tf.keras.losses.Reduction = AUTO):
    """Regression from pooled node states with mean squared log scaled error.

    Args:
      node_set_name: The node set to pool.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
      alpha_loss_param: Alpha for the mean squared log scaled error.
      epsilon_loss_param: Epsilon for the mean squared log scaled error.
      reduction: Reduction for the mean squared log scaled error.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        reduce_type=reduce_type,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name,
        alpha_loss_param=alpha_loss_param,
        epsilon_loss_param=epsilon_loss_param,
        reduction=reduction)


class RootNodeMeanAbsoluteError(_MeanAbsoluteErrorLossMixIn,
                                _RootNodeRegression):
  """Root node regression with mean absolute error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Root node regression with mean absolute error.

    Args:
      node_set_name: The node set containing the root node.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class RootNodeMeanAbsolutePercentageError(_MeanAbsolutePercentageErrorLossMixIn,
                                          _RootNodeRegression):
  """Root node regression with mean absolute percentage error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Root node regression with mean absolute percentage error.

    Args:
      node_set_name: The node set containing the root node.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class RootNodeMeanSquaredError(_MeanSquaredErrorLossMixIn, _RootNodeRegression):
  """Root node regression with mean squared error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Root node regression with mean squared error.

    Args:
      node_set_name: The node set containing the root node.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class RootNodeMeanSquaredLogarithmicError(_MeanSquaredLogarithmicErrorLossMixIn,
                                          _RootNodeRegression):
  """Root node regression with mean squared logarithmic error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Root node regression with mean squared logarithmic error.

    Args:
      node_set_name: The node set containing the root node.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class RootNodeMeanSquaredLogScaledError(_MeanSquaredLogScaledErrorLossMixIn,
                                        _RootNodeRegression):
  """Root node regression with mean squared log scaled error."""

  def __init__(self,
               node_set_name: str,
               *,
               units: int = 1,
               state_name: str = tfgnn.HIDDEN_STATE,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None,
               alpha_loss_param: float = 5.,
               epsilon_loss_param: float = 1e-8,
               reduction: tf.keras.losses.Reduction = AUTO):
    """Root node regression with mean squared log scaled error.

    Args:
      node_set_name: The node set containing the root node.
      units: The units for the regression head.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
      alpha_loss_param: Alpha for the mean squared log scaled error.
      epsilon_loss_param: Epsilon for the mean squared log scaled error.
      reduction: Reduction for the mean squared log scaled error.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name,
        alpha_loss_param=alpha_loss_param,
        epsilon_loss_param=epsilon_loss_param,
        reduction=reduction)


class NodeMeanAbsoluteError(_MeanAbsoluteErrorLossMixIn, _NodeRegression):
  """Node regression with mean absolute error via structured readout."""

  def __init__(self,
               key: str = "seed",
               *,
               units: int = 1,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True):
    """Node regression with mean absolute error via structured readout.

    This task defines regression via structured readout (see
    `tfgnn.keras.layers.StructuredReadout`).  Structured readout addresses the
    need to read out final hidden states from a GNN computation to make
    predictions for some nodes of interest. To add auxiliary node
    (and edge) sets for structured readout see, e.g.:
    `tfgnn.keras.layers.AddReadoutFromFirstNode`.

    Args:
      key: A string key to select between possibly multiple named readouts.
      units: The units for the regression head.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
    """
    super().__init__(
        key,
        units=units,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name,
        feature_name=feature_name,
        readout_node_set=readout_node_set,
        validate=validate)


class NodeMeanAbsolutePercentageError(_MeanAbsolutePercentageErrorLossMixIn,
                                      _NodeRegression):
  """Node regression with mean absolute percentage error via structured readout.
  """

  def __init__(self,
               key: str = "seed",
               *,
               units: int = 1,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True):
    """Node regression with mean absolute percentage error via structured readout.

    This task defines regression via structured readout (see
    `tfgnn.keras.layers.StructuredReadout`).  Structured readout addresses the
    need to read out final hidden states from a GNN computation to make
    predictions for some nodes of interest. To add auxiliary node
    (and edge) sets for structured readout see, e.g.:
    `tfgnn.keras.layers.AddReadoutFromFirstNode`.

    Args:
      key: A string key to select between possibly multiple named readouts.
      units: The units for the regression head.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
    """
    super().__init__(
        key,
        units=units,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name,
        feature_name=feature_name,
        readout_node_set=readout_node_set,
        validate=validate)


class NodeMeanSquaredError(_MeanSquaredErrorLossMixIn, _NodeRegression):
  """Node regression with mean squared error via structured readout."""

  def __init__(self,
               key: str = "seed",
               *,
               units: int = 1,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True):
    """Node regression with mean squared error via structured readout.

    This task defines regression via structured readout (see
    `tfgnn.keras.layers.StructuredReadout`).  Structured readout addresses the
    need to read out final hidden states from a GNN computation to make
    predictions for some nodes of interest. To add auxiliary node
    (and edge) sets for structured readout see, e.g.:
    `tfgnn.keras.layers.AddReadoutFromFirstNode`.

    Args:
      key: A string key to select between possibly multiple named readouts.
      units: The units for the regression head.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
    """
    super().__init__(
        key,
        units=units,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name,
        feature_name=feature_name,
        readout_node_set=readout_node_set,
        validate=validate)


class NodeMeanSquaredLogarithmicError(_MeanSquaredLogarithmicErrorLossMixIn,
                                      _NodeRegression):
  """Node regression with mean squared log error via structured readout."""

  def __init__(self,
               key: str = "seed",
               *,
               units: int = 1,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True):
    """Node regression with mean squared log error via structured readout.

    This task defines regression via structured readout (see
    `tfgnn.keras.layers.StructuredReadout`).  Structured readout addresses the
    need to read out final hidden states from a GNN computation to make
    predictions for some nodes of interest. To add auxiliary node
    (and edge) sets for structured readout see, e.g.:
    `tfgnn.keras.layers.AddReadoutFromFirstNode`.

    Args:
      key: A string key to select between possibly multiple named readouts.
      units: The units for the regression head.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
    """
    super().__init__(
        key,
        units=units,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name,
        feature_name=feature_name,
        readout_node_set=readout_node_set,
        validate=validate)


class NodeMeanSquaredLogScaledError(_MeanSquaredLogScaledErrorLossMixIn,
                                    _NodeRegression):
  """Node regression with mean squared log scaled error via structured readout.
  """

  def __init__(self,
               key: str = "seed",
               *,
               units: int = 1,
               name: str = "regression_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True,
               alpha_loss_param: float = 5.,
               epsilon_loss_param: float = 1e-8,
               reduction: tf.keras.losses.Reduction = AUTO):
    """Node regression with mean squared log scaled error via structured readout.

    This task defines regression via structured readout (see
    `tfgnn.keras.layers.StructuredReadout`).  Structured readout addresses the
    need to read out final hidden states from a GNN computation to make
    predictions for some nodes of interest. To add auxiliary node
    (and edge) sets for structured readout see, e.g.:
    `tfgnn.keras.layers.AddReadoutFromFirstNode`.

    Args:
      key: A string key to select between possibly multiple named readouts.
      units: The units for the regression head.
      name: The regression head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
      alpha_loss_param: Alpha for the mean squared log scaled error.
      epsilon_loss_param: Epsilon for the mean squared log scaled error.
      reduction: Reduction for the mean squared log scaled error.
    """
    super().__init__(
        key,
        units=units,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name,
        feature_name=feature_name,
        readout_node_set=readout_node_set,
        validate=validate,
        alpha_loss_param=alpha_loss_param,
        epsilon_loss_param=epsilon_loss_param,
        reduction=reduction)
