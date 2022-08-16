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
"""Common data classes and functions for all preprocessing operations."""

import math
from typing import Any, Callable, Optional, Mapping, NamedTuple, Tuple, Union
import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const


class SizeConstraints(NamedTuple):
  """Constraints on the number of entities in the graph."""
  total_num_components: Union[int, tf.Tensor]
  total_num_nodes: Mapping[const.NodeSetName, Union[int, tf.Tensor]]
  total_num_edges: Mapping[const.EdgeSetName, Union[int, tf.Tensor]]
  min_nodes_per_component: Union[Mapping[const.NodeSetName,
                                         Union[int, tf.Tensor]], Tuple[()]] = ()


class FeatureDefaultValues(NamedTuple):
  """Default values for graph context, node sets and edge sets features."""
  context: Optional[const.Fields] = None
  node_sets: Optional[Mapping[const.NodeSetName, const.Fields]] = None
  edge_sets: Optional[Mapping[const.EdgeSetName, const.Fields]] = None


class BasicStats(NamedTuple):
  """The mininum, maximum and mean values of a nest of numeric values."""
  minimum: Union[tf.Tensor, Any]
  maximum: Union[tf.Tensor, Any]
  mean: Union[tf.Tensor, Any]


def compute_basic_stats(dataset: tf.data.Dataset) -> BasicStats:
  """Evaluates basic statistics for each nested tensor from the dataset.

  Args:
    dataset: Dataset containing nest of numeric values.

  Returns:
    BasicStats for dataset elements. Each evaluated statistics has the same
    structure as `dataset.element_spec`.
  """

  def reduce_fn(old_state: Tuple[tf.Tensor, Any],
                new_element) -> Tuple[tf.Tensor, Any]:
    old_count, old_stats = old_state
    new_count = old_count + 1
    alpha = 1.0 / tf.cast(new_count, tf.float64)
    new_stats = BasicStats(
        minimum=tf.nest.map_structure(tf.minimum, old_stats.minimum,
                                      new_element),
        maximum=tf.nest.map_structure(tf.maximum, old_stats.maximum,
                                      new_element),
        # From Donald Knuth’s Art of Computer Programming, 3rd edition, Vol 2,
        # 4.2.2, (15), B. P. Welford algorithm. Also Ling, R.F., “Comparison of
        # Several Algorithms for Computing Sample Means and Variances”, 1974.
        mean=tf.nest.map_structure(
            lambda s, e: s + (tf.cast(e, tf.float64) - s) * alpha,
            old_stats.mean, new_element),
    )
    return (new_count, new_stats)

  def cast_to_float32_or_float64(value_spec, value):
    if value_spec.dtype == value.dtype:
      return value
    # Prefer float32 type as it is, generally, better supported.
    return tf.cast(value, tf.float32)

  stats0 = BasicStats(
      minimum=tf.nest.map_structure(
          lambda spec: tf.fill(spec.shape, spec.dtype.max),
          dataset.element_spec),
      maximum=tf.nest.map_structure(
          lambda spec: tf.fill(spec.shape, spec.dtype.min),
          dataset.element_spec),
      mean=tf.nest.map_structure(
          lambda spec: tf.fill(spec.shape, tf.constant(0.0, tf.float64)),
          dataset.element_spec),
  )

  count0 = tf.constant(0, tf.int64)
  _, stats = dataset.reduce((count0, stats0), reduce_fn)
  return BasicStats(
      minimum=stats.minimum,
      maximum=stats.maximum,
      mean=tf.nest.map_structure(cast_to_float32_or_float64,
                                 dataset.element_spec, stats.mean))


def dataset_filter_with_summary(
    dataset: tf.data.Dataset,
    predicate: Callable[[Any], tf.Tensor],
    *,
    summary_name: str = 'dataset_removed_fraction',
    summary_steps: int = 1000,
    summary_decay: Optional[float] = None):
  """Dataset filter with a summary for the fraction of dataset elements removed.

  The fraction of removed elements is computed using exponential moving average.
  See https://en.wikipedia.org/wiki/Moving_average.

  The summary is reported each `summary_steps` elements in the input dataset
  before filtering. Statistics are reported using `tf.summary.scalar()` with
  `step` set to the element index in the result (filtered) dataset, see
  https://tensorflow.org/tensorboard/scalars_and_keras#logging_custom_scalars
  for how to write and retreive them.

  Args:
    dataset: An input dataset.
    predicate: A function mapping a dataset element to a boolean.
    summary_name: A name for this summary.
    summary_steps: Report summary for this number of elements in the input
      dataset before filtering.
    summary_decay: An exponential moving average decay factor. If not set,
      defaults to the `exp(- 1 / summary_steps)`.

  Returns:
    Thed dataset containing the elements of this dataset for which predicate is
    `True`.
  """
  if not summary_steps > 0:
    raise ValueError(('Expected summary_steps > 0,'
                      f' actual summary_steps={summary_steps}.'))

  if summary_decay is None:
    summary_decay = math.exp(-1. / float(summary_steps))
  else:
    if not 0. < summary_decay < 1.:
      raise ValueError(('Expected 0 < summary_decay < 1,'
                        f' actual summary_decay={summary_decay}.'))

  def report(value: tf.Tensor, step: tf.Tensor) -> tf.Tensor:
    ops = tf.summary.scalar(summary_name, value, step=step)
    with tf.control_dependencies([ops]):
      return tf.identity(value)

  def scan_fn(
      state: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], value: Any
  ) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Any]]:
    old_in_count, old_out_count, old_ema = state
    pred = predicate(value)
    ema_update = tf.cast(tf.logical_not(pred), old_ema.dtype)

    in_count = old_in_count + 1
    out_count = old_out_count + tf.cast(pred, old_out_count.dtype)

    # Exponential moving average (EMA) is updated according to the rule
    #    ema_{t + 1} = ema_{t} + alpha * (value_{t} - ema_{t})  (1),
    # where alpha = 1 - decay. This formula suggests that an effective
    # smoothing window size is O(1 / alpha), so for small alpha values a good
    # initial estimate for ema_{0} becomes very important as it could take
    # O(1 / alpha) steps to recover. The typical solution is to have some
    # warm-up period t0, t0 ~ 1 / alpha, to estimate ema_{t0} e.g using simple
    # average and only then use apply (1) for follow up updates:
    #
    # ema_{t + 1} = ema_{t} + alpha_{t} * (value_{t} - ema_{t}),
    # with alpha_{t} = max(1 / (t + 1),  1 - decay).
    #
    # See https://en.wikipedia.org/wiki/Exponential_smoothing.
    alpha = tf.maximum(
        tf.constant(1.0 - summary_decay, old_ema.dtype),
        1.0 / tf.cast(in_count, old_ema.dtype))
    ema = old_ema + alpha * (ema_update - old_ema)
    report_step = tf.maximum(out_count - 1, 0)
    ema = tf.cond(in_count % summary_steps == 0,
                  lambda: report(ema, step=report_step), lambda: ema)

    return (in_count, out_count, ema), (pred, value)

  state0 = (tf.constant(0, tf.int64), tf.constant(0, tf.int64),
            tf.constant(0., tf.float64))
  dataset = dataset.scan(state0, scan_fn)
  dataset = dataset.filter(lambda predicate, value: predicate)
  dataset = dataset.map(lambda predicate, value: value)
  return dataset
