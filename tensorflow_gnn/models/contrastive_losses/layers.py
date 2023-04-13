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
"""Perturbation layers."""
from __future__ import annotations

import dataclasses
import functools
from typing import Callable, List, Mapping, Optional, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn

_PACKAGE = "GNN>models>contrastive_losses"
_WILDCARD = "*"

FieldCorruptionSpec = Mapping[tfgnn.FieldName, float]
NodeCorruptionSpec = Mapping[tfgnn.SetName, FieldCorruptionSpec]
EdgeCorruptionSpec = Mapping[tfgnn.SetName, FieldCorruptionSpec]
ContextCorruptionSpec = FieldCorruptionSpec


# TODO(tsitsulin,dzelle): revisit * pattern after cl/515352301 is in.
@dataclasses.dataclass
class CorruptionSpec:
  """Class for defining corruption specification.

  This has three fields for specifying the corruption behavior of node-, edge-,
  and context sets.

  A value of the key "*" is a wildcard value that is used for either all
  features or all node/edge sets.

  Some example usages:

  Want: corrupt everything with parameter 1.0.
  Solution: either set defalut_rate to 1.0 or set all corruption specs to
    `{"*": 1.}`.

  Want: corrupt all context features with parameter 1.0 except for "feat", which
    should not be corrupted.
  Solution: set `context_corruption` to `{"feat": 0., "*": 1.}`
  """

  node_set_corruption: NodeCorruptionSpec = dataclasses.field(
      default_factory=dict
  )
  edge_set_corruption: EdgeCorruptionSpec = dataclasses.field(
      default_factory=dict
  )
  context_corruption: ContextCorruptionSpec = dataclasses.field(
      default_factory=dict
  )

  def with_default_rate(self, default_rate: float):
    node_set_corruption = {_WILDCARD: {_WILDCARD: default_rate}}
    edge_set_corruption = {_WILDCARD: {_WILDCARD: default_rate}}
    context_corruption = {_WILDCARD: default_rate}

    for k, v in self.node_set_corruption.items():
      node_set_corruption[k] = {_WILDCARD: default_rate, **v}
    for k, v in self.edge_set_corruption.items():
      edge_set_corruption[k] = {_WILDCARD: default_rate, **v}
    context_corruption.update(self.context_corruption)

    return CorruptionSpec(
        node_set_corruption=node_set_corruption,
        edge_set_corruption=edge_set_corruption,
        context_corruption=context_corruption,
    )


class _Corruptor(tfgnn.keras.layers.MapFeatures):
  """Base class for graph corruptor."""

  def __init__(
      self,
      corruption_spec: Optional[CorruptionSpec] = None,
      *,
      corruption_fn: Callable[[tfgnn.Field, float], tfgnn.Field],
      default_rate: Optional[float] = None,
      **kwargs,
  ):
    """Captures arguments for `call`.

    Args:
      corruption_spec: A spec for corruption application.
      corruption_fn: Corruption function.
      default_rate: Global application rate of the corruptor. This is only used
        when `corruption_spec` is None.
      **kwargs: Additional keyword arguments.
    """
    if corruption_spec is None and default_rate is None:
      raise ValueError(
          "At least one of `corruption_spec` or `default_rate` must be set."
      )
    if corruption_spec is None:
      corruption_spec = CorruptionSpec()
    if default_rate is not None:
      corruption_spec = corruption_spec.with_default_rate(default_rate)
    self._default_rate = default_rate
    self._corruption_fn = corruption_fn
    self._node_corruption_spec = corruption_spec.node_set_corruption
    self._edge_corruption_spec = corruption_spec.edge_set_corruption
    self._context_corruption_spec = corruption_spec.context_corruption

    def fn(inputs, *, node_set_name=None, edge_set_name=None):
      if node_set_name is not None:
        spec = self._node_corruption_spec.get(
            node_set_name, self._node_corruption_spec[_WILDCARD]
        )
      elif edge_set_name is not None:
        spec = self._edge_corruption_spec.get(
            edge_set_name, self._edge_corruption_spec[_WILDCARD]
        )
      else:
        spec = self._context_corruption_spec
      return _corrupt_features(
          inputs.features,
          self._corruption_fn,
          corruption_spec=spec,
      )

    super().__init__(fn, fn, fn, **kwargs)

  def get_config(self):
    return dict(
        default_rate=self._default_rate,
        corruption_fn=self._corruption_fn,
        node_corruption_spec=self._node_corruption_spec,
        edge_corruption_spec=self._edge_corruption_spec,
        context_corruption_spec=self._context_corruption_spec,
        **super().get_config(),
    )

  @classmethod
  def from_config(cls, config):
    config["corruption_spec"] = CorruptionSpec(
        config.pop("node_corruption_spec"),
        config.pop("edge_corruption_spec"),
        config.pop("context_corruption_spec"),
    )
    return cls(**config)


def _seed_wrapper(
    fn: Callable[..., tfgnn.Field], seed: Optional[int] = None
) -> Callable[..., tfgnn.Field]:
  @functools.wraps(fn)
  def wrapper_fn(tensor, rate):
    return fn(tensor, rate, seed=seed)

  return wrapper_fn


@tf.keras.utils.register_keras_serializable(package=_PACKAGE)
class ShuffleFeaturesGlobally(_Corruptor):
  """A corruptor that shuffles features."""

  def __init__(self, *args, seed: Optional[float] = None, **kwargs):
    self._seed = seed
    seeded_fn = _seed_wrapper(_shuffle_tensor, seed=seed)
    super().__init__(*args, corruption_fn=seeded_fn, default_rate=1.0, **kwargs)

  def get_config(self):
    return dict(seed=self._seed, **super().get_config())


@tf.keras.utils.register_keras_serializable(package=_PACKAGE)
class DropoutFeatures(_Corruptor):

  def __init__(self, *args, seed: Optional[float] = None, **kwargs):
    self._seed = seed
    seeded_fn = _seed_wrapper(tf.nn.dropout, seed=seed)
    super().__init__(*args, corruption_fn=seeded_fn, default_rate=0.0, **kwargs)

  def get_config(self):
    return dict(seed=self._seed, **super().get_config())


def _ragged_dim_list(tensor: tf.RaggedTensor) -> List[Union[int, tf.Tensor]]:
  """Lists ragged tensor dimensions with a preference for static sizes."""
  static = tensor.row_lengths().shape.as_list()
  if None not in static:
    return static
  dynamic = tf.unstack(tf.shape(tensor.row_lengths()))
  return [(d if s is None else s) for s, d in zip(static, dynamic)]


# TODO(tsitsulin,dzelle): consider moving into core library and replace
# `tfgnn.shuffle_features_globally`.
def _shuffle_tensor(
    tensor: tfgnn.Field, rate: float = 1.0, *, seed: Optional[int] = None
) -> tfgnn.Field:
  """Randomly permutes a fixed `rate` percentage of rows in the input tensor.

  This function shuffles tensors across the second (feature) dimension. For
  example, tensor [node_batch, feature_dim, ...] is shuffled across feature_dim.

  Args:
    tensor: Input Tensor. Both dense and ragged tensors are supported.
    rate: Percentage of the rows to be shuffled.
    seed: A seed for random uniform shuffle.

  Returns:
    A shuffled tensor.
  """
  if rate < 0.0 or rate > 1.0:
    raise ValueError(f"Shuffle rate should be in [0, 1] (got {rate}).")
  if rate == 0.0:
    return tensor
  # Empty tensors can not get shuffled.
  if any(i == 0 for i in tensor.get_shape().as_list()):
    return tensor
  if tfgnn.is_dense_tensor(tensor):
    if rate == 1.0:
      return tf.random.shuffle(tensor, seed=seed)
    batch_size = tf.shape(tensor)[0]
    num_rows_to_shuffle = tf.cast(
        tf.math.ceil(tf.cast(batch_size, tf.float32) * rate), tf.int32
    )
    random_rows = tf.random.shuffle(tf.range(0, batch_size), seed=seed)[
        :num_rows_to_shuffle
    ]
    random_rows_shuffled = tf.random.shuffle(random_rows, seed=seed)
    return tf.tensor_scatter_nd_update(
        tensor,
        tf.expand_dims(random_rows, 1),
        tf.gather(tensor, random_rows_shuffled),
    )
  if tfgnn.is_ragged_tensor(tensor):
    # Not sure this is optimal.
    batch_size = _ragged_dim_list(tensor)[0]
    num_rows_to_shuffle = tf.cast(
        tf.math.ceil(tf.cast(batch_size, tf.float32) * rate), tf.int64
    )
    row_to_newrow = tf.range(batch_size, dtype=tensor.row_splits.dtype)
    random_rows = tf.random.shuffle(row_to_newrow, seed=seed)[
        :num_rows_to_shuffle
    ]
    row_to_newrow = tf.tensor_scatter_nd_update(
        row_to_newrow,
        tf.expand_dims(random_rows, 1),
        tf.sort(random_rows),
    )
    new_rowids = tf.gather(row_to_newrow, tensor.value_rowids())
    new_values = tf.gather(tensor.values, tf.argsort(new_rowids))
    return tf.RaggedTensor.from_value_rowids(
        new_values, tf.sort(new_rowids), nrows=tensor.nrows()
    )
  raise ValueError(
      "Operation is currently supported only for dense or ragged tensors."
  )


def _corrupt_features(
    features: tfgnn.Fields,
    corruption_fn: Callable[[tfgnn.Field, float], tfgnn.Field],
    corruption_spec: FieldCorruptionSpec,
):
  """Corrupts all features given a corruption function and corruption spec.

  Both `features` and `corruption_spec` are mappings. If a key is not
  present in the `corruption_spec` we use the wildcard key `_WILDCARD`.

  Args:
    features: A mapping from feature name to tensor to corrupt.
    corruption_fn: A function that corrupts a tensor given a corruption rate.
    corruption_spec: Corruption specification.

  Returns:
    The corrupted `features`.
  """
  output = {}
  for feature_name, feature_value in features.items():
    rate = corruption_spec.get(feature_name, corruption_spec[_WILDCARD])
    output[feature_name] = (
        feature_value if not rate else corruption_fn(feature_value, rate)
    )
  return output


@tf.keras.utils.register_keras_serializable(package=_PACKAGE)
class DeepGraphInfomaxLogits(tf.keras.layers.Layer):
  """Computes clean and corrupted logits for Deep Graph Infomax (DGI)."""

  def build(self, input_shape: tf.TensorShape) -> None:
    """Builds a bilinear layer."""
    if not isinstance(input_shape, tf.TensorShape):
      raise ValueError(f"Expected `TensorShape` (got {type(input_shape)})")
    units = input_shape.as_list()[-1]
    if units is None:
      raise ValueError(f"Expected a defined inner dimension (got {units})")
    # Bilinear layer.
    self._bilinear = tf.keras.layers.Dense(units, use_bias=False)

  def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Computes clean and corrupted logits for DGI.

    Args:
      inputs: A stacked tensor with clean and corrupted (in order)
        representations stacked like `[batch, 2, representation dim]`.

    Returns:
      A concatenated (clean, corrupted) logits, respectively.
    """
    x_clean, x_corrupted = tf.unstack(inputs, axis=1)
    # Summary.
    summary = tf.math.reduce_mean(x_clean, axis=0, keepdims=True)
    # Clean logits.
    logits_clean = tf.matmul(x_clean, self._bilinear(summary), transpose_b=True)
    # Corrupted logits.
    logits_corrupted = tf.matmul(
        x_corrupted, self._bilinear(summary), transpose_b=True
    )
    return tf.keras.layers.Concatenate()((logits_clean, logits_corrupted))
