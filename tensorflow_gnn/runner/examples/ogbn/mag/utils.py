# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Utils specific to OGBN-MAG training.
"""
from typing import Any

import tensorflow as tf
import tensorflow_gnn as tfgnn


def mask_by_indices(values: tf.Tensor, indices: tf.Tensor,
                    mask_value: Any) -> tf.Tensor:
  """([[1, 2], [3, 4], [5, 6]], [0, 2], -1) -> [[-1, -1], [3, 4], [-1, -1]]."""
  mask_value = tf.convert_to_tensor(mask_value, values.dtype)
  assert mask_value.shape.rank == 0
  dim0 = tf.shape(values)[0] if values.shape[0] is None else values.shape[0]
  fdims = values.shape.as_list()[1:]
  assert None not in fdims, "Inner dims must be fully defined."
  mask_value = tf.fill([dim0] + fdims, mask_value)
  mask = tf.fill(tf.shape(indices), True)
  indices = tf.expand_dims(indices, -1)
  mask = tf.scatter_nd(indices, mask, [dim0])
  mask = tf.reshape(mask, [-1] + [1] * len(fdims))
  mask = tf.tile(mask, [1] + fdims)
  return tf.where(mask, mask_value, values)


def make_causal_mask(node_set: tfgnn.NodeSet,
                     feature_name: tfgnn.FieldName = "year") -> tf.Tensor:
  """Mask where true indicate neighbors published in same year or after seed."""
  seed_starts = tf.math.cumsum(node_set.sizes, exclusive=True)
  seed_features = tf.gather(node_set[feature_name], seed_starts)
  repeated_seed_features = tf.repeat(seed_features, node_set.sizes)
  return tf.math.greater_equal(node_set[feature_name], repeated_seed_features)


def mask_paper_labels(
    node_set: tfgnn.NodeSet,
    label_feature_name: tfgnn.FieldName,
    mask_value: Any,
    extra_label_mask=None,
) -> tfgnn.Field:
  """Masks label features of specific paper nodes with mask_values."""

  label_feature = node_set[label_feature_name]
  if extra_label_mask is not None:
    assert extra_label_mask.dtype == tf.bool
    label_feature = tf.where(
        extra_label_mask,
        mask_value * tf.ones_like(label_feature),
        label_feature,
    )
  seed_starts = tf.math.cumsum(node_set.sizes, exclusive=True)
  return mask_by_indices(label_feature, seed_starts, mask_value)

