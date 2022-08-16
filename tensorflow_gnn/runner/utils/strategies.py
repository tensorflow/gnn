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
    super().__init__(cluster_resolver, variable_partitioner)


class TPUStrategy(tf.distribute.TPUStrategy):
  """A `TPUStrategy` convenience wrapper."""

  def __init__(self, tpu: str = ""):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    super().__init__(resolver)
