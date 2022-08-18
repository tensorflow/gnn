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
"""Defines advanced batching operations for GraphTensor."""
import functools

from typing import Any, cast, Iterable, List, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow_gnn.graph import adjacency
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import padding_ops
from tensorflow_gnn.graph import preprocessing_common

SizeConstraints = preprocessing_common.SizeConstraints


class _ScanState(NamedTuple):
  """The state used by `dynamic_batch` in `tf.data.Dataset.scan()`."""
  budget_left: SizeConstraints  # Upper bounds only.
  accumulator: Tuple[tf.TensorArray]


def dynamic_batch(dataset: tf.data.Dataset,
                  constraints: SizeConstraints) -> tf.data.Dataset:
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
  budget, min_nodes_per_component = _validate_and_prepare_constraints(
      constraints, input_spec)

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
    return _ScanState(budget_left=budget, accumulator=accumulator)

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
    budget = _set_min_nodes_per_component(state.budget_left,
                                          min_nodes_per_component)
    within_budget = padding_ops.satisfies_size_constraints(graph_tensor, budget)
    return tf.math.logical_not(within_budget)

  def scan_func(
      state: _ScanState, value: Tuple[gt.GraphTensor, tf.Tensor]
  ) -> Tuple[_ScanState, Tuple[tf.Tensor, gt.GraphTensor]]:
    graph_tensor, eod_flag = value

    def flush():
      with tf.control_dependencies(
          padding_ops.assert_satisfies_size_constraints(
              graph_tensor, size_constraints=budget)):
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


def find_tight_size_constraints(
    dataset: tf.data.Dataset,
    *,
    min_nodes_per_component: Optional[Mapping[const.NodeSetName, int]] = None,
    target_batch_size: Optional[Union[int, tf.Tensor]] = None,
) -> SizeConstraints:
  """Returns smallest possible size constraints that allow dataset padding.

  Evaluated constraints are intended to be used when it is required that all
  elements of the `dataset` can be padded, e.g., when evaluating models.

  Typically, this function is used on a dataset of individual examples (that is,
  not batched), and the `target_batch_size` is passed as an argument. The
  returned constraints will work for all possible batches up to that size drawn
  from the dataset.

  Alternatively, this function can be used on a dataset that is already batched,
  passing `target_batch_size=None`. The returned constraints will work for the
  batches exactly as seen in the dataset. However, note that many performance-
  optimized ways of building a Dataset (like parallel .map() and .interleave()
  calls before .batch()) introduce nondeterminism and may not deliver the exact
  same batches again.

  Note that this function iterates over all elements of the input dataset, so
  its execution time is proportional to the dataset's cardinality.

  Args:
    dataset: finite dataset of graph tensors of any rank.
    min_nodes_per_component: mapping from a node set name to a minimum number of
      nodes in each graph component. Defaults to 0.
    target_batch_size: if not `None`, an integer for multiplying the sizes
      measured from dataset before making room for padding.

  Returns:
    Smalles possible size constraints that allows padding of all graph tensors
    in the input dataset.

  Raises:
    ValueError: if dataset elements are not GraphTensors or its cardinality
      is `tf.data.INFINITE_CARDINALITY`.
  """
  graph_tensor_spec = dataset.element_spec
  if not isinstance(dataset.element_spec, gt.GraphTensorSpec):
    raise ValueError('The element of dataset must be GraphTensor.')
  if dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
    raise ValueError('The dataset must be finite.')

  ds = dataset.map(_get_total_sizes_int64)
  size_contraints = preprocessing_common.compute_basic_stats(ds).maximum
  assert isinstance(size_contraints, SizeConstraints)

  if target_batch_size is not None:
    def multiply_by_batch_size(size):
      if isinstance(size, tf.Tensor):
        # Avoid errors from int32/64 mismatch.
        return size * tf.cast(target_batch_size, size.dtype)
      else:
        return size * target_batch_size

    size_contraints = tf.nest.map_structure(multiply_by_batch_size,
                                            size_contraints)

  return _fine_tune_learned_constraints(
      size_contraints, graph_tensor_spec=graph_tensor_spec,
      min_nodes_per_component=min_nodes_per_component)


def learn_fit_or_skip_size_constraints(
    dataset: tf.data.Dataset,
    batch_size: Union[int, Iterable[int]],
    *,
    min_nodes_per_component: Optional[Mapping[const.NodeSetName, int]] = None,
    success_ratio: Union[float, Iterable[float]] = 1.0,
    sample_size: int = 100_000,
    num_thresholds: int = 1_000) -> Union[SizeConstraints, List[Any]]:
  """Learns the optimal size constraints for the fixed size batching with retry.

  The function estimates the smallest possible size constraints so that a random
  sample of `batch_size` graph tensors meets those constraints with probability
  no less than `success_ratio`. The success ratio is treated as a hard
  constraint, up to sampling error. The constraints can be used for graph tensor
  padding to the fully defined shapes required by XLA.

  Example:

  ```python
  # Learn size constraints for a given dataset of graph tensors and the target
  # batch size(s). The constraints could be learned once and then reused.
  constraints = tfgnn.learn_fit_or_skip_size_constraints(dataset, batch_size)

  # Batch merge contained graphs into scalar graph tensors.
  if training:
    # Randomize and repeat dataset for training. Note that the fit-or-skip
    # technique is only applicable for randomizer infinite datasets. It is
    # incorrect to apply it during models evaluation because some input
    # examples may be filtered out.
    dataset = dataset.shuffle(shuffle_size).repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda graph: graph.merge_batch_to_components())

  if training:
    # Remove all batches that do not satisfy the learned constraints.
    dataset = dataset.filter(
        functools.partial(
            tfgnn.satisfies_size_constraints,
            total_sizes=constraints))

    # Pad graph to the learned size constraints.
    dataset = dataset.map(
        functools.partial(
            tfgnn.pad_to_total_sizes,
            size_constraints=constraints,
            validate=False))
  ```

  The learned constraints are intend to be used only with randomized repeated
  dataset. This dataset are first batched using `tf.data.Dataset.batch()`, the
  batches that are too large to fit the learned contraints are filtered using
  `tfgnn.satisfies_size_constraints()` and then padded
  `tfgnn.pad_to_total_sizes()`.

  This approach, if applicable, is more efficient compared to padding to the
  maximum possible sizes. It is also simpler and faster compared to the dynamic
  batching, especially for the large batch sizes (>10).  To illustrate the main
  point, consider graphs containing only 0 or 1 nodes. A random batch of 1000 of
  those graphs could contain 1000 nodes in the worst case. If this maximum limit
  is used to reseve space for random 1000 graphs, the space of 425 nodes is used
  only in 1:1000_000 cases. It is >40% more efficient to reserve space only for
  575 nodes and resample batches in the rare cases when they do not fit.


  Args:
    dataset: dataset of graph tensors that is intended to be batched.
    batch_size: the target batch size(s). Could be a single positive integer
      value or any iterable. For the latter case the result is reported for each
      requested value.
    min_nodes_per_component: mapping from a node set name to a minimum number of
      nodes in each graph component. Defaults to 0.
    success_ratio: the target probability(s) that a random batch of graph tensor
      satisfies the learned constraints. Could be a single float value between 0
      and 1 or any iterable. For the latter case the result is reported for
      each requested value. NOTE: setting success_ratio to 1 only guarantees
      that all sampled graphs are satisfy the learned constraints. This does not
      in general apply to an arbitrary sample. When `sample_size` tends to
      infinity, the 1 ratio corresponds to the "almost surely satisfies" event.
    sample_size: the number of the first dataset examples to use for inference.
    num_thresholds: the number of quantiles to use to approximate probability
      distributions.

  Returns:
    Learned size constraints. If both `batch_size` and `success_ratio` are
    iterables, the result is returned as a nested lists, were `result[b][r]`
    is a size constraints for `batch_size[b]` and `success_ratio[r]`. If any of
    `batch_size` or/and `success_ratio` are scalars the corresponding dimension
    is squeezed in the output.
  """

  # Convert `batch_size` and `success_ratio` to lists:
  batch_sizes = _convert_to_list(batch_size, int, 'batch_size')
  success_ratios = _convert_to_list(success_ratio, float, 'success_ratio')

  # Validate parameters:
  if not all(b > 0 for b in batch_sizes):
    raise ValueError(f'The `batch_size` must be positive, got {batch_size}')

  if not all((0. <= r <= 1.) for r in success_ratios):
    raise ValueError(
        f'The `success_ratio` must be between 0 and 1, got {success_ratio}')

  if not sample_size > 0:
    raise ValueError(f'The `sample_size` must be positive, got {sample_size}')

  if not num_thresholds > 0:
    raise ValueError(
        f'The `num_thresholds` must be positive, got {num_thresholds}')

  if not isinstance(dataset.element_spec, gt.GraphTensorSpec):
    raise ValueError('Expected dataset with GraphTensor elements,'
                     f' got dataset.element_spec={dataset.element_spec}.')

  # Extract graph piece sizes and flatten them as a rank=1 int64 tensor.
  graph_tensor_spec = dataset.element_spec
  dataset = dataset.map(_get_total_sizes_int64)
  result_type_spec = dataset.element_spec
  dataset = dataset.map(lambda t: tf.stack(tf.nest.flatten(t)))

  # Cache `sample_size` samples into memory for better perfromance.
  dataset = dataset.take(sample_size)
  dataset = dataset.cache()

  # It’s easy enough to sample the distribution of sizes for each graph piece,
  # but the key question is how to distribute the available “error budget”, that
  # is `1 - success_ratio`, between them. The following algorithm uses a scoring
  # approach for graph pieces to turn that multivariate optimization problem
  # into a data fitting problem in a single parameter.
  #
  # Notation:
  # The algorithm accepts scalar hyperparameters
  #   N = sample_size (the number of sampled batches) and
  #   T = num_thresholds (the number of sampled size thresholds)
  # and a graph with
  #   graph pieces g, for 0 <= g < G = #node sets + # edge sets + 1 (context)
  # and then executes the following in parallel for all combinations of
  #   batch_size[b] for 0 <= b < B and
  #   success_ratio[r] for 0 <= r < R.
  #
  # 1) Take a random sample of N_0 input examples. For simplicity, we choose
  #    N_0 = N.
  # 2) From these examples, create a random sample of N possible batches for
  #    each batch size.
  # 3) For each graph piece g in each sample batch, compute the actual size
  #    (total number of elements).
  # 4) For each graph piece and each batch size b from step 3,
  #    compute the standard deviation and quantiles 0 <= t < T of actual sizes.
  # 5) Compute an importance score in range [0,1] for each graph piece g and
  #    batch size b based on the statistics computed in step 4.
  # 6) The candidate constraint for graph piece g, batch size b and threshold
  #    value t is generated as ⌈t * importance_score⌉-th largest quantile.
  #    So graph pieces with 0 importance have maximum possible constraint value,
  #    graph pieces with the largest importance 1 have the tightest possible
  #    constraints for the given threshold t.
  # 7) Select the largest index t such that the fraction of sample batches from
  #    step 2 that satisfy the t-th size threshold for every graph piece g is at
  #    least the requested success ratio r. Output this selection of constraints
  #    for the graph pieces as the result for the requested batch size b and
  #    success ratio r.
  #
  # By construction, the result has the required success probability on the
  # sample of batches. It depends on the scoring function for graph pieces in
  # step 5 and its use in step 6 how tight the size constraints are.
  #
  # The scoring function for a graph piece used in the implementation below is
  # the ratio of standard deviations between this graph piece and the maximal
  # one. Intuitively, the effect is to err on the size of over-budgeting for
  # low-variance graph pieces (because they are unlikely to under-utilize it a
  # lot) and focus the parameter fitting on high-variance graph pieces (because
  # they are likely to dominate the tradeoff between under-utilization and
  # failure rate).
  #
  # Summay of notations used for tensor indices:
  #   g in range(G): graph pieces.
  #   b in range(B): requested batch sizes, B = len(batch_sizes).
  #   t in range(T): sampled thresholds, T = num_thresholds.
  #   r in range(R): requesxted success ratios, R = len(success_ratios).
  @tf.function
  def sample_sizes_for_each_batch_size(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Creates ramdomized dataset with actual sizes of graph pieces in batches.

    The result dataset has exactly `sample_size` sampled at random elements with
    sizes for each batch size and graph piece. Note that result has the same
    number of elements as the original dataset. In principle we could sample
    less having the same statistical power, but in order to keep things simple
    we use the same sample size parameter. Generally we do not have a exact
    formula to estimate  `sample_size` and we simply rely that it is big enough.

    Args:
      ds: dataset containing flattened graph piece sizes as rank=1 tf.Tensor.

    Returns:
      dataset containing exactly `sample_size` elements. Each element is rank-2
      integer tensor with shape [G, B].
    """

    def extract_sizes_fn(sizes_ig: tf.Tensor):
      """Exracts sizes of graph pieces for each `batch_sizes` value.

      Args:
        sizes_ig: rank=2 integer tensor cotaining graph piece sizes. The first
          `i` dimension has size max(batch_sizes). It index sizes that belong to
          the same graphs. The second `g` dimension index graph entities
          (context, node sets, edge sets).

      Returns:
        Rank=2 integer tensor with shape [G, B].
      """
      # Notation: i index all possible batch sizes from [1, max(batch_sizes)].
      sizes_ig.shape.assert_is_fully_defined()
      sizes_gi = tf.transpose(sizes_ig)
      sizes_gi = tf.cumsum(sizes_gi, axis=-1, exclusive=False)
      # Compute indices in sizes_gi for batch sizes of interest.
      indices_1b = tf.reshape(batch_sizes, [1, -1]) - 1
      indices_gb = tf.tile(indices_1b, [sizes_gi.shape[0], 1])
      # Gather sizes for each requested batch size.
      sizes_gb = tf.gather(sizes_gi, indices_gb, batch_dims=1)
      return sizes_gb

    assert isinstance(ds.element_spec, tf.TensorSpec)
    assert ds.element_spec.shape.rank == 1
    ds = ds.repeat()
    ds = ds.shuffle(sample_size)
    ds = ds.batch(max(batch_sizes), drop_remainder=True)
    ds = ds.map(extract_sizes_fn)
    ds = ds.take(sample_size)
    return ds

  dataset = sample_sizes_for_each_batch_size(dataset)

  @tf.function
  def get_constraints_for_thresholds(ds: tf.data.Dataset) -> tf.Tensor:
    """Estimates size constraints for each quantiles' threshold value.

    NOTE: this algorithm consumes O(B * G * sample_size) memory. For simplicity,
    the implementation computes exact quantiles by sorting values. We can
    significantly reduce memory usage by using approximate quantiles algorithms,
    e.g. https://arxiv.org/abs/1603.05346.

    Args:
      ds: dataset containing flattened graph entities sizes for each
        `batch_sizes` value. Has a shape [G, B].

    Returns:
      Estimated size constraints for each `batch_sizes` and thresholds values.
      Has a shape [B, G, T].
    """
    # Notation: `i` index samples in the range [0, sample_size).
    #
    # Extract `sample_size` to compute quantiles:
    sizes_igb = ds.batch(
        sample_size, drop_remainder=True).take(1).get_single_element()
    sizes_igb.shape.assert_is_fully_defined()
    sizes_bgi = tf.transpose(sizes_igb)
    # Compute standard deviation for each batch size and graph piece:
    std_bg = tf.math.reduce_std(tf.cast(sizes_bgi, tf.float32), axis=-1)
    # Compute the largest graph piece standard deviation for each batch size:
    max_std_b1 = tf.reduce_max(std_bg, axis=-1, keepdims=True)

    # Compute importance coefficients as sizes std to the maximum std ratio. We
    # are considering only size constraints with probabilities of violation that
    # is proportional to the importance coefficients (see theory above).
    importance_bg = tf.math.divide_no_nan(std_bg, max_std_b1)
    importance_bg1 = tf.expand_dims(importance_bg, axis=-1)

    sorted_sizes_bgi = tf.sort(sizes_bgi, axis=-1, direction='DESCENDING')
    indices_t = tf.convert_to_tensor(
        np.linspace(
            start=0, stop=sample_size, endpoint=False, num=num_thresholds),
        tf.float32)
    indices_bgt = tf.cast(indices_t * importance_bg1, tf.int32)
    tf.debugging.assert_less(indices_bgt, sample_size)
    constraints_bgt = tf.gather(sorted_sizes_bgi, indices_bgt, batch_dims=2)
    return constraints_bgt

  constraints_bgt = get_constraints_for_thresholds(dataset)

  @tf.function
  def eval_success_ratios(ds: tf.data.Dataset,
                          constraints_bgt: tf.Tensor) -> tf.Tensor:
    """Computes fraction of examples that satisfy size constraints.

    Args:
      ds: dataset containing graph entities sizes for different batch sizes.
        integer tensors with a shape [G, B].
      constraints_bgt: size constraint value for each batch size, graph entity
        and thresholds value. Has a shape [B, G, T].

    Returns:
      Fraction of graphs from the `ds` that sattisfy `constraints_bgt` for
      each batch size and threshold as a float tensor with [B, T] shape.
    """

    def map_fn(sizes_gb: tf.Tensor):
      sizes_bg1 = tf.expand_dims(tf.transpose(sizes_gb), axis=-1)
      satisfied_bgt = sizes_bg1 <= constraints_bgt
      satisfied_bt = tf.math.reduce_all(satisfied_bgt, axis=1)
      satisfied_bt = tf.cast(satisfied_bt, tf.int32)
      return satisfied_bt

    ds = ds.map(map_fn)
    return preprocessing_common.compute_basic_stats(ds).mean

  success_ratios_bt = eval_success_ratios(dataset, constraints_bgt)

  @tf.function
  def get_constraints_matching_input_succes_ratios(
      constraints_bgt: tf.Tensor, success_ratios_bt: tf.Tensor) -> tf.Tensor:
    """Matches constraint values to the `batch_sizes` and `success_ratios`.

    Args:
      constraints_bgt: size constraint for each `batch_sizes` value, graph
        entity and threshold value. Has a shape [B, G, T].
      success_ratios_bt: the estimated probabilities that graph satisfies
        matching `constraints_bgt` value. Has a shape [B, T].

    Returns:
       Size constraints for each `batch_sizes`, `success_ratios` values and
       graph piece. Has a shape [B, R, G].
    """
    b_size, t_size = success_ratios_bt.shape.as_list()
    g_size = constraints_bgt.shape[1]
    reversed_indices_br = tf.searchsorted(
        tf.reverse(success_ratios_bt, axis=[-1]),
        values=tf.tile(tf.expand_dims(success_ratios, axis=0), [b_size, 1]))
    indices_br = tf.maximum((t_size - 1) - reversed_indices_br, 0)

    indices_bgr = tf.tile(tf.expand_dims(indices_br, axis=1), [1, g_size, 1])
    constraints_bgr = tf.gather(constraints_bgt, indices_bgr, batch_dims=2)
    constraints_brg = tf.transpose(constraints_bgr, [0, 2, 1])
    return constraints_brg

  constraints_brg = get_constraints_matching_input_succes_ratios(
      constraints_bgt, success_ratios_bt)

  def squeeze(value: List[Any]) -> Any:
    """Extracts value from the single element list."""
    assert len(value) == 1, value
    return value[0]

  result = constraints_brg

  # Unstack batch sizes (b) and success ratios (r) dimensions to nested lists:
  result = tf.unstack(result, axis=0)
  result = tf.nest.map_structure(lambda t: tf.unstack(t, axis=0), result)

  # Unflatten size constraints to SizeConstraints.

  def get_size_constraints(
      flattened_constraints: tf.Tensor) -> SizeConstraints:
    size_constraints = tf.nest.pack_sequence_as(
        result_type_spec, tf.unstack(flattened_constraints))
    return _fine_tune_learned_constraints(
        size_constraints, graph_tensor_spec=graph_tensor_spec,
        min_nodes_per_component=min_nodes_per_component)

  result = tf.nest.map_structure(get_size_constraints, result)

  # Reshape result depending on the passed arguments:
  if not isinstance(success_ratio, Iterable):
    result = [squeeze(s) for s in result]
  if not isinstance(batch_size, Iterable):
    result = squeeze(result)

  return result


def _set_min_nodes_per_component(
    size_constraints: SizeConstraints,
    min_nodes_per_component: Optional[Mapping[const.NodeSetName, int]]
) -> SizeConstraints:
  assert not size_constraints.min_nodes_per_component
  if not min_nodes_per_component:
    return size_constraints

  return SizeConstraints(
      total_num_components=size_constraints.total_num_components,
      total_num_nodes=size_constraints.total_num_nodes.copy(),
      total_num_edges=size_constraints.total_num_edges.copy(),
      min_nodes_per_component=min_nodes_per_component.copy())


def _make_room_for_padding(
    size_constraints: SizeConstraints, graph_tensor_spec: gt.GraphTensorSpec,
    min_nodes_per_component: Optional[Mapping[const.NodeSetName, int]]):
  """Reserves space in `size_constraints` for padding."""
  # NOTE: there is currently no way to identify latent (w.o. features) node
  # sets with a static total number of nodes (`node_set_spec.total_size` is
  # always False). A side effect of this is that this function always reserves
  # space for a fake graph component (`total_num_components` increased by 1).
  # TODO(b/217538005): remove this comment after fixing.
  extra_num_components = int(
      graph_tensor_spec.total_num_components is None
      or any(n.total_size is None
             for n in graph_tensor_spec.node_sets_spec.values())
      or any(e.total_size is None
             for e in graph_tensor_spec.edge_sets_spec.values()))
  if extra_num_components == 0:
    return _set_min_nodes_per_component(size_constraints,
                                        min_nodes_per_component)

  extra_num_nodes = {}
  for node_set_name in graph_tensor_spec.node_sets_spec:
    extra_num_nodes[node_set_name] = (min_nodes_per_component or
                                      {}).get(node_set_name, 0)

  extra_num_edges = {}
  for edge_set_name, edge_set_spec in graph_tensor_spec.edge_sets_spec.items():
    extra_num_edges[edge_set_name] = 0
    if edge_set_spec.total_size is None:
      adj = graph_tensor_spec.edge_sets_spec[edge_set_name].adjacency_spec
      for _, (incident_node_set_name, _) in adj.get_index_specs_dict().items():
        extra_num_nodes[incident_node_set_name] = max(
            extra_num_nodes[incident_node_set_name], 1)

  result = tf.nest.map_structure(
      lambda size, extra_size: size + extra_size, size_constraints,
      SizeConstraints(
          total_num_components=extra_num_components,
          total_num_nodes=extra_num_nodes,
          total_num_edges=extra_num_edges))
  return _set_min_nodes_per_component(result, min_nodes_per_component)


def _fine_tune_learned_constraints(
    size_constraints: SizeConstraints, graph_tensor_spec: gt.GraphTensorSpec,
    min_nodes_per_component: Optional[Mapping[const.NodeSetName, int]]
) -> SizeConstraints:
  """Reserves space for padding and converts eager tensors to Python ints."""
  result = _make_room_for_padding(
      size_constraints,
      graph_tensor_spec,
      min_nodes_per_component=min_nodes_per_component)
  if tf.executing_eagerly():
    # If executed eagerly, convert integer scalar tensors to the python int.
    def cast_to_int(t):
      return int(t.numpy() if isinstance(t, tf.Tensor) else t)
    result = tf.nest.map_structure(cast_to_int, result)
  return result


def _convert_to_list(value: Union[Any, Iterable[Any]], value_dtype,
                     value_debug_name: str) -> List[Any]:
  """Converts single value or iterable to the python list."""
  if isinstance(value, value_dtype):
    return [value]
  if isinstance(value, Iterable):
    as_list = list(value)
    if not as_list or isinstance(as_list[0], value_dtype):
      return as_list
  dtype_debug_name = value_dtype.__name__
  raise ValueError((f'Expected `{value_debug_name}` of '
                    f'type {dtype_debug_name} or List[{dtype_debug_name}]'))


def _get_total_sizes(graph_tensor: gt.GraphTensor) -> SizeConstraints:
  """Returns the total number of items in the `graph_tensor`."""
  return SizeConstraints(
      total_num_components=graph_tensor.total_num_components,
      total_num_nodes={
          name: node_set.total_size
          for name, node_set in graph_tensor.node_sets.items()
      },
      total_num_edges={
          name: edge_set.total_size
          for name, edge_set in graph_tensor.edge_sets.items()
      })


def _get_total_sizes_int64(graph: gt.GraphTensor) -> SizeConstraints:
  """Same as `_get_total_sizes()` but with all sizes casted to tf.int64."""
  result = _get_total_sizes(graph)
  return tf.nest.map_structure(lambda s: tf.cast(s, tf.int64), result)


def _validate_and_prepare_constraints(
    constraints: SizeConstraints, graph_tensor_spec: gt.GraphTensorSpec
) -> Tuple[SizeConstraints, Mapping[const.NodeSetName, Union[int, tf.Tensor]]]:
  """Checks constraints and matches value types to the `graph_tensor_spec`."""
  def keys_to_debug_str(keys):
    return '[' + ', '.join(f'<{k}>' for k in sorted(keys)) + ']'

  keys_diff = set(constraints.total_num_nodes) - set(
      graph_tensor_spec.node_sets_spec)
  if keys_diff:
    raise ValueError(
        'The `constraints.total_num_nodes` keys must be existing node set'
        f' names. Invalid keys: {keys_to_debug_str(keys_diff)}.')
  keys_diff = set(constraints.total_num_edges) - set(
      graph_tensor_spec.edge_sets_spec)
  if keys_diff:
    raise ValueError(
        'The `constraints.total_num_edges` keys must be existing edge set'
        f' names. Invalid keys: {keys_to_debug_str(keys_diff)}.')
  keys_diff = set(constraints.min_nodes_per_component) - set(
      graph_tensor_spec.node_sets_spec)
  if keys_diff:
    raise ValueError(
        'The `constraints.min_nodes_per_component` keys must be existing node'
        f' set names. Invalid keys: {keys_to_debug_str(keys_diff)}.')

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

  max_total_sizes = SizeConstraints(
      total_num_components=tf.cast(
          constraints.total_num_components,
          graph_tensor_spec.context_spec.sizes_spec.dtype),
      total_num_nodes=total_num_nodes,
      total_num_edges=total_num_edges)
  return max_total_sizes, dict(constraints.min_nodes_per_component)


def dataset_from_generator(generator) -> tf.data.Dataset:
  """Creates dataset from generator of any nest of scalar graph pieces.

  Similar to `tf.data.Dataset.from_generator()`, but requires the generator
  to yield at least one element and sets the result's `.element_spec` from it.
  In subsequent elements, graph pieces must have the same features (incl. their
  shapes and dtypes), and graphs must have the same edge sets and node sets, but
  the numbers of nodes and edges may vary between elements.

  NOTE: Compared to `tf.data.from_generator()` the generator is first called
  during the dataset construction. If generator is shared between two datasets
  this could lead to some obscure behaviour, like:

  ```
  my_generator = [pieceA, pieceB, pieceC, pieceD]
  dataset1 = tfgnn.dataset_from_generator(my_generator).take(2)
  dataset2 = tfgnn.dataset_from_generator(my_generator).take(2)
  print([dataset2])  # prints: pieceB, pieceC, while expected pieceA, pieceB.
  print([dataset1])  # prints: pieceA, pieceD.
  ```

  Args:
    generator: a callable object that returns an object that supports the iter()
      protocol. Could consist of any nest of tensors and scalar graph pieces
      (e.g. `tfgnn.GraphTensor`, `tfgnn.Context`, `tfgnn.NodeSet`,
      `tfgnn.EdgeSet`, `tfgnn.Adjacency`, etc.)

  Returns:
    A `tf.data.Dataset`.

  Raises:
    ValueError: if any contained graph piece is not scalar or has not compatible
      number of graph components.
  """
  iterator0 = iter(generator())
  try:
    first = next(iterator0)
  except StopIteration as e:
    raise ValueError('The generator must produce at least one value.') from e

  def _get_spec(value):
    spec = tf.type_spec_from_value(value)
    if not isinstance(spec, gp.GraphPieceSpecBase):
      return spec

    gp.check_scalar_graph_piece(spec, 'dataset_from_generator()')
    if isinstance(spec, gt.GraphTensorSpec):
      return spec.relax(num_components=False, num_nodes=True, num_edges=True)
    if isinstance(spec, gt.ContextSpec):
      return spec
    if isinstance(spec, gt.NodeSetSpec):
      return spec.relax(num_components=False, num_nodes=True)
    if isinstance(spec, gt.EdgeSetSpec):
      return spec.relax(num_components=False, num_edges=True)
    if isinstance(spec,
                  (adjacency.AdjacencySpec, adjacency.HyperAdjacencySpec)):
      return spec.relax(num_edges=True)
    raise NotImplementedError(type(spec).__name__)

  def _validate(index: int, expected_spec, value) -> None:
    """Provides readable error messages for common mistakes."""
    # pylint: disable=protected-access

    if not isinstance(expected_spec,
                      (gt.GraphTensorSpec, gt._GraphPieceWithFeaturesSpec)):
      return
    actual_spec = _get_spec(value)

    if actual_spec != expected_spec:
      raise ValueError('Generated graph pieces are not compatible.'
                       f' piece0: {expected_spec},'
                       f' piece{index}: {actual_spec}.')

  relaxed_spec = tf.nest.map_structure(_get_spec, first)
  head_stack = [first]

  def restored_generator():
    # If dataset is re-iterated multiple times, we want to make sure that the
    # generator is called again for the first element.
    start_index = len(head_stack)
    if start_index == 0:
      iterator = iter(generator())
    else:
      iterator = iterator0
      while head_stack:
        yield head_stack.pop()

    for index, value in enumerate(iterator, start=start_index):
      tf.nest.map_structure(
          functools.partial(_validate, index), relaxed_spec, value)
      yield value

  return tf.data.Dataset.from_generator(
      restored_generator, output_signature=relaxed_spec)
