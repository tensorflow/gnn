"""Common data classes and functions for all preprocessing operations."""

from typing import Any, Optional, Mapping, NamedTuple, Tuple, Union
import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const


class SizesConstraints(NamedTuple):
  """Constraints on the number of entities in the graph."""
  total_num_components: Union[int, tf.Tensor]
  total_num_nodes: Mapping[const.NodeSetName, Union[int, tf.Tensor]]
  total_num_edges: Mapping[const.EdgeSetName, Union[int, tf.Tensor]]


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
