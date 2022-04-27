"""Wrappers of Tensorflow distributed strategies."""
from typing import Optional
import tensorflow as tf

MinSizePartitioner = tf.distribute.experimental.partitioners.MinSizePartitioner


class ParameterServerStrategy(tf.distribute.ParameterServerStrategy):
  """A `ParameterServerStrategy` convenience wrapper."""

  def __init__(self, min_shard_bytes: Optional[int] = None):
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    num_ps = cluster_resolver.cluster_spec().num_tasks("ps")
    # If min_shard_bytes is not supplied, use the recommended default.
    if min_shard_bytes is None:
      variable_partitioner = MinSizePartitioner(max_shards=num_ps)
    else:
      variable_partitioner = MinSizePartitioner(min_shard_bytes, num_ps)
    super(ParameterServerStrategy, self).__init__(
        cluster_resolver,
        variable_partitioner)
