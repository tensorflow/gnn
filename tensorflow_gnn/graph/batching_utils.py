"""Defines advanced batching operations for GraphTensor."""

from typing import cast, NamedTuple, Tuple

import tensorflow as tf

from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import padding_ops
from tensorflow_gnn.graph import preprocessing_common

SizesConstraints = preprocessing_common.SizesConstraints


class _ScanState(NamedTuple):
  """The state used by `dynamic_batch` in `tf.data.Dataset.scan()`."""
  budget_left: SizesConstraints
  accumulator: Tuple[tf.TensorArray]


def dynamic_batch(dataset: tf.data.Dataset,
                  constraints: SizesConstraints) -> tf.data.Dataset:
  """Batches as many consecutive graphs as allowed by the `constraints`.

  Each result batch can have variable number of graphs. Batches are returned as
  graph tensors of rank 1 with the first dimension that indexes individual
  examples in the batch. The result graphs could be converted to scalar graph
  tensors using `.merge_batch_to_components()` and then padded to the target
  sizes with `pad_to_total_sizes()`.

  TODO(b/212274918): add support for non-scalar input graph tensors.

  NOTE: this operation is more expensive compared to the fixed size batching.
  The overhead is mainly due to `tf.data.Dataset.scan()` and grows with the
  average number of graphs in the result batches. This overhead is small when
  only a few examples are batched on evarage. When ~100 examples are combined on
  average the operation could become 3-4x slower. For the latter case consider
  static batching as by the law of large numbers it should create comparable
  results over such a large sample size. Another alternative is a mixed
  strategy: if on average N >> 10 graphs are batched, first use fixed size
  batching with sqrt(N) batch size, convert rank-1 results into scalar graphs
  using `.merge_batch_to_components()` and then apply dynamic batching.

  Args:
    dataset: dataset of scalar graph tensors.
    constraints: the size contrains for the graph tensor. Must define the
      maximum number of graph components (`.total_num_components`), the maximum
      total number of nodes in each node set (`.total_num_nodes[node_set_name]`)
      and likewise for each edge set (`.total_num_edges[edge_set_name]`).

  Returns:
    The dataset of rank-1 graph tensors compatible with the `constraints`.

  Raises:
    ValueError: if the `constraints` are not defined for some node sets or edges
      sets defined by the graph tensors type specification.
    tf.errors.InvalidArgumentError: if any of the input graph tensor instances
      are not compatible with the `constraints` so batching is not possible. For
      example, if some graph tensor has more nodes then it is allowed.
  """
  # pylint: disable=protected-access
  #
  # The implementation relies on `._to_tensor_list()` from the composite tensor
  # API to convert graph pieces into the flat list of variant tensors. The API
  # guarantees that those components could be stacked independetly and with
  # `._from_tensor_list()` combined into the rank+1 graph piece. Those stackable
  # components are accumulated in the TensorArray containers. This allows the
  # components to be stacked only once using `TensorArray.stack()` when result
  # batch is finalized.
  input_spec = dataset.element_spec
  if not isinstance(input_spec, gt.GraphTensorSpec):
    raise ValueError('The element of dataset must be scalar GraphTensor.')
  gt.check_scalar_graph_tensor(
      cast(gt.GraphTensorSpec, input_spec), 'dynamic_batch()')

  output_spec = input_spec._batch(None)
  constraints = _validate_and_prepare_constraints(constraints, input_spec)

  # A terminating element is needed at the end of a finite dataset to flush
  # accumulated inputs even if the size budget is not exhausted yet. We mark it
  # with a boolean flag. This should not create any visible overhead compared to
  # the graph tensor itself.
  def add_eod_flag(dataset: tf.data.Dataset, flag: bool) -> tf.data.Dataset:
    return dataset.map(lambda g: (g, flag))

  has_infinite_cardinality = dataset.cardinality(
  ) == tf.data.INFINITE_CARDINALITY
  if has_infinite_cardinality:
    # For known-infinite datasets (like the repeated training data), we can take
    # a shortcut, because there is no last element.
    dataset = add_eod_flag(dataset, False)
  else:
    # For datasets with known-finite or unknown cardinality, we attach an extra
    # copy of the first element with EOD flag at the end. (It's up to the code
    # below to not output that.) Repeating a genuine input instead of an empty
    # value is meant to avoid potential special cases elsewhere.
    dataset_end = dataset.take(1)
    dataset = add_eod_flag(dataset, False)
    dataset = dataset.concatenate(add_eod_flag(dataset_end, True))

  def get_empty_value() -> gt.GraphTensor:
    return output_spec._create_empty_value()

  def get_initial_state() -> _ScanState:
    accumulator = list()
    for spec in input_spec._flat_tensor_specs:
      accumulator.append(
          tf.TensorArray(
              spec.dtype, size=0, dynamic_size=True, clear_after_read=True))
    accumulator = tuple(accumulator)
    return _ScanState(budget_left=constraints, accumulator=accumulator)

  def extract_value(state: _ScanState) -> gt.GraphTensor:
    value = tf.nest.map_structure(lambda t: t.stack(), state.accumulator)
    value = output_spec._from_tensor_list(list(value))
    return value

  def get_next_state(state: _ScanState,
                     graph_tensor: gt.GraphTensor) -> _ScanState:
    budget_left = tf.nest.map_structure(tf.math.subtract, state.budget_left,
                                        _get_total_sizes(graph_tensor))
    accumulator = tf.nest.map_structure(
        lambda ta, spec: ta.write(ta.size(), spec), state.accumulator,
        tuple(graph_tensor.spec._to_tensor_list(graph_tensor)))
    return _ScanState(budget_left=budget_left, accumulator=accumulator)

  def exceeds_budget(state: _ScanState,
                     graph_tensor: gt.GraphTensor) -> tf.Tensor:
    within_budget = padding_ops.satisfies_total_sizes(graph_tensor,
                                                      state.budget_left)
    return tf.math.logical_not(within_budget)

  def scan_func(
      state: _ScanState, value: Tuple[gt.GraphTensor, tf.Tensor]
  ) -> Tuple[_ScanState, Tuple[tf.Tensor, gt.GraphTensor]]:
    graph_tensor, eod_flag = value

    def flush():
      with tf.control_dependencies(
          padding_ops.assert_satisfies_total_sizes(
              graph_tensor, target_total_sizes=constraints)):
        # For simplicity, next_state remembers the graph_tensor in all cases.
        # If graph_tensor comes with eod_flag=True, there will be no further
        # call to flush(), and this artificially added graph_tensor is omitted
        # from the output, as it should.
        next_state = get_next_state(get_initial_state(), graph_tensor)
        return (next_state, (True, extract_value(state)))

    def accumulate():
      next_state = get_next_state(state, graph_tensor)
      return (next_state, (False, get_empty_value()))

    should_flush = tf.math.logical_or(
        exceeds_budget(state, graph_tensor), eod_flag)
    return tf.cond(should_flush, flush, accumulate)

  dataset = dataset.scan(get_initial_state(), scan_func)
  dataset = dataset.filter(lambda has_value, _: has_value)
  dataset = dataset.map(lambda _, value: value)
  if has_infinite_cardinality and dataset.cardinality(
  ) != tf.data.INFINITE_CARDINALITY:
    # The Dataset.filter() always sets cardinality to the UNKNOWN_CARDINALITY.
    # In our case the `filter()` operation from above could only filter up to
    # `constraints.total_num_components` consecutive elements, so if the input
    # dataset is INFINITE_CARDINALITY so should be the output.
    dataset = dataset.repeat()

  return dataset


def _get_total_sizes(graph_tensor: gt.GraphTensor) -> SizesConstraints:
  """Returns the total number of items in the `graph_tensor`."""
  return SizesConstraints(
      total_num_components=graph_tensor.total_num_components,
      total_num_nodes={
          name: node_set.total_size
          for name, node_set in graph_tensor.node_sets.items()
      },
      total_num_edges={
          name: edge_set.total_size
          for name, edge_set in graph_tensor.edge_sets.items()
      })


def _validate_and_prepare_constraints(
    constraints: SizesConstraints,
    graph_tensor_spec: gt.GraphTensorSpec) -> SizesConstraints:
  """Checks constraints and matches value types to the `graph_tensor_spec`."""
  total_num_nodes = {}
  for node_set_name, node_set_spec in graph_tensor_spec.node_sets_spec.items():
    if node_set_name not in constraints.total_num_nodes:
      raise ValueError(
          f'The maximum total number of <{node_set_name}> nodes must be'
          f' specified as `constraints.total_num_nodes[<{node_set_name}>]`.')
    total_num_nodes[node_set_name] = tf.cast(
        constraints.total_num_nodes[node_set_name],
        node_set_spec.sizes_spec.dtype)

  total_num_edges = {}
  for edge_set_name, edge_set_spec in graph_tensor_spec.edge_sets_spec.items():
    if edge_set_name not in constraints.total_num_edges:
      raise ValueError(
          f'The maximum total number of <{edge_set_name}> edges must be'
          f' specified as `constraints.total_num_edges[<{edge_set_name}>]`.')
    total_num_edges[edge_set_name] = tf.cast(
        constraints.total_num_edges[edge_set_name],
        edge_set_spec.sizes_spec.dtype)

  return SizesConstraints(
      total_num_components=tf.cast(
          constraints.total_num_components,
          graph_tensor_spec.context_spec.sizes_spec.dtype),
      total_num_nodes=total_num_nodes,
      total_num_edges=total_num_edges)
