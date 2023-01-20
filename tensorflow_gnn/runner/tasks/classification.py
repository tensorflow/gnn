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
"""Classification tasks."""
import abc
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces

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

  def get_config(self) -> Mapping[Any, Any]:
    return dict(from_logits=self._from_logits, **super().get_config())


@tf.keras.utils.register_keras_serializable(package="GNN")
class FromLogitsPrecision(_FromLogitsMixIn, tf.keras.metrics.Precision):
  pass


@tf.keras.utils.register_keras_serializable(package="GNN")
class FromLogitsRecall(_FromLogitsMixIn, tf.keras.metrics.Recall):
  pass


class _PerClassMetricMixIn(tf.keras.metrics.Metric):
  """Mixin for `tf.keras.metrics.Metric` with a sparse_class_id option.

  This Mixin is needed because ground truths come as class id integer, which is
  incompatible with tf.keras.metrics.Precision.
  """

  def __init__(self, sparse_class_id: int, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._sparse_class_id = sparse_class_id

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> None:
    return super().update_state(
        (y_true == self._sparse_class_id),
        (tf.argmax(y_pred, -1) == self._sparse_class_id),
        sample_weight)

  def get_config(self) -> Mapping[Any, Any]:
    return dict(sparse_class_id=self._sparse_class_id, **super().get_config())


@tf.keras.utils.register_keras_serializable(package="GNN")
class PerClassPrecision(_PerClassMetricMixIn, tf.keras.metrics.Precision):
  pass


@tf.keras.utils.register_keras_serializable(package="GNN")
class PerClassRecall(_PerClassMetricMixIn, tf.keras.metrics.Recall):
  pass


class _Classification(interfaces.Task):
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
    return tf.keras.Model(model.input, logits)

  @abc.abstractmethod
  def preprocess(self, gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    raise NotImplementedError()

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

  def preprocess(self, gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    return gt

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.BinaryCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (FromLogitsPrecision(from_logits=True),
            FromLogitsRecall(from_logits=True),
            tf.keras.metrics.AUC(from_logits=True, name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", from_logits=True, name="auc_pr"),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True))


class _MulticlassClassification(_Classification):
  """Multiclass classification."""

  def __init__(self,
               *args,
               num_classes: Optional[int] = None,
               class_names: Optional[Sequence[str]] = None,
               per_class_statistics: bool = False,
               **kwargs):  # pylint: disable=useless-super-delegation
    if (num_classes is None) == (class_names is None):
      raise ValueError(
          "Exactly one of `num_classes` or `class_names` must be specified")
    if num_classes is None:
      num_classes = len(class_names)
    super().__init__(num_classes, *args, **kwargs)
    if class_names is None:
      self._class_names = [f"class_{i}" for i in range(num_classes)]
    else:
      self._class_names = class_names
    self._per_class_statistics = per_class_statistics

  def preprocess(self, gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    return gt

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Sparse categorical crossentropy loss."""
    return (tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Sparse categorical metrics."""
    metric_objs = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)]

    if self._per_class_statistics:
      for i, class_name in enumerate(self._class_names):
        metric_objs.append(
            PerClassPrecision(
                sparse_class_id=i,
                name=f"precision_for_{class_name}"))
        metric_objs.append(
            PerClassRecall(
                sparse_class_id=i,
                name=f"recall_for_{class_name}"))
    return metric_objs


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
    return tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT,
        self._reduce_type,
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(gt)


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
