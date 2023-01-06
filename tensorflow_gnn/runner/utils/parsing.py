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
"""Helpers for `GraphTensor` parsing."""
import tensorflow as tf
import tensorflow_gnn as tfgnn

GraphTensor = tfgnn.GraphTensor
GraphTensorSpec = tfgnn.GraphTensorSpec


def maybe_parse_graph_tensor_dataset(
    ds: tf.data.Dataset,
    gtspec: GraphTensorSpec) -> tf.data.Dataset:
  """Parse (or check the compatability of) a dataset with `GraphTensorSpec`.

  * If `ds` contains `tf.string` elements, the dataset is parsed using `gtspec`
    and returned.
  * If `ds` contains `GraphTensor` elements, the dataset is checked
    (by `tfgnn.create_schema_pb_from_graph_spec(...)`) to be compatible with
    `gtspec` and returned.
  * Otherwise, a `ValueError` is raised.

  Args:
    ds: A `tf.data.Dataset` to parse or check.
    gtspec: A `GraphTensorSpec` for parsing or checking.

  Returns:
    A `tf.data.Dataset` that has been parsed by, or checked for compatibility
    with, `gtspec`.

  Raises:
      ValueError: If `ds` does contain `tf.string` or `GraphTensor` elements.
  """
  if ds.element_spec.is_compatible_with(tf.TensorSpec((), tf.string)):
    ds = ds.map(
        tfgnn.keras.layers.ParseSingleExample(gtspec),
        deterministic=False,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif ds.element_spec.is_compatible_with(tf.TensorSpec((None,), tf.string)):
    ds = ds.map(
        tfgnn.keras.layers.ParseExample(gtspec),
        deterministic=False,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif not isinstance(ds.element_spec, tfgnn.GraphTensorSpec):
    raise ValueError(f"Expected `GraphTensorSpec` (got {ds.element_spec})")
  else:
    element_spec = ds.element_spec
    while element_spec.rank > 0:
      element_spec = element_spec._unbatch()  # pylint: disable=protected-access
    schema = tfgnn.create_schema_pb_from_graph_spec(gtspec)
    tfgnn.check_compatible_with_schema_pb(element_spec, schema)
  return ds
