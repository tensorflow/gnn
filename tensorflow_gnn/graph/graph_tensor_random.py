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
"""Generation of random-valued GraphTensor instances for testing.
"""

import string
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as gc
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tensor_utils as tu


def random_ragged_tensor(
    shape: List[Union[int, None, tf.Tensor]],
    dtype: tf.dtypes.DType,
    sample_values: Optional[List[Any]] = None,
    row_lengths_range: Tuple[int, int] = (2, 8),
    row_splits_dtype: tf.dtypes.DType = tf.int32,
    validate: bool = True) -> gc.Field:
  """Generate a ragged tensor with random values.

  NOTE: This is running in pure Python, not building a TensorFlow graph.

  Args:
    shape: The desired shape of the tensor, as a list of integers, `None` (for
      ragged dimensions) or a dynamically computed size (as a tensor of rank 0).
      Do not provide a tf.TensorShape here as it cannot hold tensors.
    dtype: Data type for the values.
    sample_values: List of example values to sample from. If not specified, some
      simple defaults are produced. The data type must match that used to
      initalize `dtype`.
    row_lengths_range: Minimum and maximum values for each row lengths in a
      ragged range.
    row_splits_dtype: Data type for row splits.
    validate: If true, then use assertions to check that the arguments form a
      valid RaggedTensor. Note: these assertions incur a runtime cost, since
      they must be checked for each tensor value.

  Returns:
    An instance of either tf.Tensor of tf.RaggedTensor.
  """

  # Allocate partitions for each , generating random row lengths where ragged.
  # This also computes the total number of values to be inserted in the final
  # tensor (`size`).
  nested_row_lengths = []
  if not isinstance(shape, list):
    raise ValueError(f"Requested shape must be a list of integers, `None` or "
                     f"rank-0 tensors: {shape}")
  if shape:
    size = tf.constant(1, tf.int32)
    for dim in shape:
      if isinstance(dim, (int, tf.Tensor)):
        if isinstance(dim, tf.Tensor):
          _assert_rank0_int(dim, "Shape dimension")
          dim = tf.cast(dim, tf.int32)
        nested_row_lengths.append(dim)
        size *= dim
      else:
        assert dim is None, f"Invalid type for dimension in shape: {dim}"
        low, high = row_lengths_range
        row_lengths = tf.random.uniform([size], low, high, row_splits_dtype)
        nested_row_lengths.append(row_lengths)
        size = tf.cast(tf.math.reduce_sum(row_lengths), tf.int32)
  else:
    size = []  # Scalar.

  # Allocate the total amount of flat values.
  if sample_values:
    # TODO(blais): Assert consistency of types from `sample_values` and `dtype`.
    indices = tf.random.uniform([size], 0, len(sample_values), tf.int32)
    sample_values_tensor = tf.convert_to_tensor(sample_values, dtype=dtype)
    flat_values = tf.gather(sample_values_tensor, indices)
  else:
    flat_values = typed_random_values(size, dtype)

  # Now, build up the ragged tensor inside out.
  #
  # NOTE(blais,edloper): The future of RaggedTensor will bring support for a
  # RowPartition representation that will make the following dispatching code
  # unnecessary.
  tensor = flat_values
  for row_lengths in reversed(nested_row_lengths[1:]):
    if isinstance(row_lengths, int):
      if isinstance(tensor, tf.RaggedTensor):
        tensor = tf.RaggedTensor.from_uniform_row_length(tensor, row_lengths,
                                                         validate=validate)
      else:
        old_shape = list(tensor.shape)
        tensor = tf.reshape(tensor, [-1, row_lengths] + old_shape[1:])
    else:
      tensor = tf.RaggedTensor.from_row_lengths(tensor, row_lengths,
                                                validate=validate)
  return tensor


def typed_random_values(size: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
  """Generate a flat array of reasonable random values of a particular type.

  Args:
    size: The number of values to generate.
    dtype: The data type of the values.
  Returns:
    A tensor of shape [size] and dtype `dtype`.
  """
  if dtype.is_integer:
    # random.uniform supports: `int32` or `int64` types.
    if dtype in (tf.dtypes.int64, tf.dtypes.uint64):
      values = tf.random.uniform([size], 100, 999, tf.int64)
    else:
      minvalue = tf.constant(max(10, dtype.min), tf.int32)
      maxvalue = tf.constant(min(100, dtype.max), tf.int32)
      values = tf.random.uniform([size], minvalue, maxvalue, tf.int32)
    values = tf.cast(values, dtype)
  elif dtype.is_floating:
    values = tf.random.uniform([size], dtype=dtype)
  elif dtype == tf.dtypes.bool:
    values = tf.random.uniform([size], dtype=tf.float32) > 0.5
  elif dtype == tf.dtypes.string:
    letters = tf.constant(list(string.ascii_uppercase))
    indices = tf.random.uniform([size], 0, len(string.ascii_uppercase),
                                dtype=tf.int32)
    values = tf.gather(letters, indices)
  else:
    raise TypeError("Unsupported type for random values: {}".format(dtype))
  return values


SampleDict = Dict[Tuple[gc.SetType, Optional[gc.SetName], gc.FieldName],
                  Union[List[str], List[int], List[float]]]


def random_graph_tensor(
    spec: gt.GraphTensorSpec,
    sample_dict: Optional[SampleDict] = None,
    row_lengths_range: Tuple[int, int] = (2, 8),
    validate: bool = True,
    num_components_range: Tuple[int, int] = (1, 2),
) -> gt.GraphTensor:
  """Generate a graph tensor from a spec, with random features.

  NOTE: This function does not (yet?) support the generation of the auxiliary
  node set for `tfgnn.structured_readout()`. It should not be included in the
  `spec`, and if needed, should be added separately in a later step.

  Args:
    spec: A GraphTensorSpec instance that describes the graph tensor. The result
      random graph tensors are generated for the relaxed number of items, as
      `spec.relax(num_components=True, num_nodes=True, num_edges=True)`.
    sample_dict: A dict of (set-type, set-name, field-name) to list-of-values to
      sample from. The intended purpose is to generate random values that are
      more realistic, more representative of what the actual dataset will
      contain. You can provide such if the values aren't provided for a feature,
      random features are inserted of the right type.
    row_lengths_range: Minimum (included) and maximum (excluded) values for each
      row lengths in a ragged range.
    validate: If true, then use assertions to check that the arguments form a
      valid RaggedTensor. Note: these assertions incur a runtime cost, since
      they must be checked for each tensor value.
    num_components_range: Minimum (included) and maximum (excluded) values for
      the number of graph components. Overrides the number of components
      from spec.

  Returns:
    A GraphTensor compatible with the given spec with relaxed number of graph
    items. The size of each node set and edge set is random within
    `row_lengths_range`. The number of components is random within
    `num_components_range`.
  """
  gt.check_scalar_graph_tensor(spec, "random_graph_tensor()")
  spec = spec.relax(num_components=True, num_nodes=True, num_edges=True)

  if sample_dict is None:
    sample_dict = {}
  if not 0 <= row_lengths_range[0] < row_lengths_range[1]:
    raise ValueError(
        "Expected 0 <= row_lengths_range[0] < row_lengths_range[1], got"
        f" {row_lengths_range}"
    )

  def _gen_features(
      set_type: gc.SetType,
      set_name: Optional[gc.SetName],
      features_spec: gc.FieldsSpec,
      num_items: tf.Tensor,
  ):
    """Generate a random feature tensor dict."""
    tensors = {}
    for fname, feature_spec in features_spec.items():
      shape = feature_spec.shape.as_list()
      if shape[0] is None:
        shape[0] = num_items
      key = (set_type, set_name, fname)
      sample_values = sample_dict.get(key, None)
      tensors[fname] = random_ragged_tensor(
          shape=shape,
          dtype=feature_spec.dtype,
          sample_values=sample_values,
          row_splits_dtype=spec.row_splits_dtype,
          validate=validate,
      )
    return tensors

  num_components = _get_num_components(spec, num_components_range)

  # Create random context features.
  context = gt.Context.from_fields(
      features=_gen_features(
          gc.CONTEXT, None, spec.context_spec.features_spec, num_components
      )
  )

  # Create random node-set features.
  node_sets = {}
  for set_name, node_set_spec in spec.node_sets_spec.items():
    min_nodes, max_nodes = row_lengths_range
    sizes = _random_sizes(
        num_components, min_nodes, max_nodes, spec.indices_dtype
    )
    node_sets[set_name] = gt.NodeSet.from_fields(
        sizes=sizes,
        features=_gen_features(
            gc.NODES,
            set_name,
            node_set_spec.features_spec,
            tf.math.reduce_sum(sizes),
        ),
    )

  # Create random edge-set features.
  edge_sets = {}
  for set_name, edge_set_spec in spec.edge_sets_spec.items():
    # Choose the number of edges for each component at random.
    # To keep it reasonable, we stay within a small constant factor of
    # the number of nodes (unless there are zero nodes on one side).
    adj_spec = edge_set_spec.adjacency_spec
    source_sizes = node_sets[adj_spec.source_name].sizes
    target_sizes = node_sets[adj_spec.target_name].sizes

    has_edges = tf.math.logical_and(source_sizes > 0, target_sizes > 0)
    sum_sizes = source_sizes + target_sizes
    sum_sizes = tf.where(has_edges, sum_sizes, tf.zeros_like(sum_sizes))
    sum_sizes = tf.cast(sum_sizes, tf.float32)
    min_edges = tf.cast(sum_sizes / 1.5, spec.indices_dtype)
    max_edges = tf.cast(sum_sizes * 2.25, spec.indices_dtype)
    sizes = _random_sizes(
        num_components, min_edges, max_edges, spec.indices_dtype
    )

    # Randomly generate the actual node indices.
    source_indices = _random_edge_indices(sizes, source_sizes)
    target_indices = _random_edge_indices(sizes, target_sizes)
    adjacency = adj.Adjacency.from_indices(
        source=(adj_spec.source_name, source_indices),
        target=(adj_spec.target_name, target_indices),
    )

    # Create the edge set.
    edge_sets[set_name] = gt.EdgeSet.from_fields(
        sizes=sizes,
        features=_gen_features(
            gc.EDGES,
            set_name,
            edge_set_spec.features_spec,
            tf.math.reduce_sum(sizes),
        ),
        adjacency=adjacency,
    )

  result = gt.GraphTensor.from_pieces(
      context=context, node_sets=node_sets, edge_sets=edge_sets
  )
  if not result.spec.is_compatible_with(spec):
    raise ValueError(
        f"Internal error: result {result.spec} is not compatible with relaxed"
        f" spec {spec}."
    )
  return result


def _get_feature_values(feature: tf.train.Feature) -> Union[List[str],
                                                            List[int],
                                                            List[float]]:
  """Return the values from a TF feature proto."""
  if feature.HasField("float_list"):
    return list(feature.float_list.value)
  elif feature.HasField("int64_list"):
    return list(feature.int64_list.value)
  elif feature.HasField("bytes_list"):
    return list(feature.bytes_list.value)
  return []


def _assert_rank0_int(t: tf.Tensor, tensor_name: Text) -> None:
  if t.shape.rank != 0 or t.dtype not in (tf.int32, tf.int64):
    raise ValueError(f"Expected `{tensor_name}` as rank-1 integer tensor,"
                     f" got rank={t.shape.rank}, dtype={t.dtype.name}")


def _random_edge_indices(
    num_edges: tf.Tensor, num_nodes: tf.Tensor
) -> tf.Tensor:
  """Random edge indices for the number of edges and nodes in each component."""
  assert num_edges.dtype == num_nodes.dtype
  dtype = num_edges.dtype
  offsets = tf.cumsum(num_nodes, exclusive=True)
  indices = tu.repeat(tf.stack([offsets, num_nodes], axis=-1), num_edges)
  offsets, lengths = tf.unstack(indices, axis=-1)
  alpha = tf.random.uniform(tf.shape(lengths), dtype=tf.float64)
  return offsets + tf.cast(alpha * tf.cast(lengths, tf.float64), dtype)


def _random_sizes(
    num_components: Union[int, tf.Tensor],
    num_items_min: Union[int, tf.Tensor],
    num_items_max: Union[int, tf.Tensor],
    dtype: tf.DType,
) -> tf.Tensor:
  """Random sizes with constraints on the number of items in each component."""
  minval = tf.convert_to_tensor(num_items_min, dtype)
  length = tf.convert_to_tensor(num_items_max - num_items_min, dtype)
  alpha = tf.random.uniform([num_components], dtype=tf.float64)
  return minval + tf.cast(alpha * tf.cast(length, tf.float64), dtype)


def _get_num_components(
    spec: gt.GraphTensorSpec, num_components_range: Tuple[int, int]
) -> Union[int, tf.Tensor]:
  """Chooses the number of components based on spec and allowed range."""
  if not 0 <= num_components_range[0] < num_components_range[1]:
    raise ValueError(
        "Expected 0 <= num_components_range[0] < num_components_range[1],"
        f" got {num_components_range}"
    )

  if num_components_range[0] == num_components_range[1]:
    return num_components_range[0]
  else:
    return tf.random.uniform(
        (),
        num_components_range[0],
        num_components_range[1],
        dtype=spec.indices_dtype,
    )
