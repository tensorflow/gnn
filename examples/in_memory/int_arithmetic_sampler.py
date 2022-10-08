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
r"""Random tree walks to make GraphTensor of subgraphs rooted at seed nodes.

The entry point is method `make_sampled_subgraphs_dataset()`, which accepts as
input, an in-memory graph dataset (from dataset.py) and `SamplingSpec`, and
returns tf.data.Dataset that generates subgraphs according to `SamplingSpec`.

Specifically, `tf.data.Dataset` made by `make_sampled_subgraphs_dataset` wraps
a generator that yields `GraphTensor`, consisting of sub-graphs, rooted at
(randomly-sampeld train) seed nodes.


# Usage Example

```
# Load the dataset.
import datasets
dataset_name = 'ogbn-arxiv'  # or {ogbn-*, cora, citeseer, pubmed}
inmem_ds = datasets.get_dataset(dataset_name)

# Craft sampling specification.
sample_size = sample_size1 = 5
graph_schema = dataset_wrapper.export_graph_schema()
sampling_spec = (tfgnn.SamplingSpecBuilder(graph_schema)
                 .seed().sample([sample_size, sample_size1]).to_sampling_spec())

train_data = make_sampled_subgraphs_dataset(inmem_ds, sampling_spec)

for graph in train_data:
  print(graph)
  break

# Alternatively: `my_model.fit(train_data)`, where `my_model` is a `keras.Model`
# composed of `tfgnn.keras.layers`.
```

# Algorithm & Implementation

`make_sampled_subgraphs_dataset(ds)` returns a generator over object
`GraphSampler(ds)` over `inmem_ds.dataset` instance `ds` which
class exposes function `random_walk_tree`, which describe below.


## Pre-processing

When an instance of `GraphSampler` is constructed, two data structures are
initialized *per edge set*:

```
edges = [
    (n1, n514),
    (n1, n34),
    (n1, n13),
    ...,           # total of 305 n1 edges.
    (n1, n4),

    (n2, n50),
    (n2, n101),
    ...
]
```

and

```
node_degrees = [
    305,  # degree of n1.
    ...,  # degree of n2.
    ...
]
```


Both of which are stored as tf.Tensor.

After initialization, function `random_walk_tree` accepts seed nodes
`[n1, n2, n3, ..., nB]`, i.e. with batch size `B`.


NOTE: generator `make_sampled_subgraphs_dataset` yield `GraphTensor`
  instances, each instance contain subgraphs rooted at a batch of nodes, which
  cycle from `ds.node_split().train`.

The seed node vector initializes the `TypedWalkTree`, which can be depicted as:

```
                 sample(f1, 'cites')
    paper --------------------------> paper
     V1    \                            V2
            \ sample(f2, 'rev_writes')            sample(f3, 'affiliated_with')
             ---------------------------> author ------------------> institution
                                            V3                          V4
```

Instance nodes of `TypedWalkTree` (above) have attribute `nodes`, which is
`tf.Tensor`, depicted as V1, V2, V3, V4 with shapes, respectively (B), (B, f1),
(B, f2), (B, f2, f3). All are with dtype `tf.int{32, 64}`, matching the dtype of
input argument to function `random_walk_tree`. For some node position (i), then
node `V1[i]` has sampled edges pointing to nodes `V2[i, 0], V2[i, 1], ...`. The
(`int`) `B` corresponds to batch size and (`int`s) `f1, f2, ...` correspond to
`sample_size` that can be configured in `SamplingSpec` proto (below).

Further, if `sampling` strategy is one of `EdgeSampling.W_REPLACEMENT_W_ORPHANS`
or `EdgeSampling.WO_REPLACEMENT_WO_REPEAT`, then each `TypedWalkTree` node will
also contain attribute `valid` (tf.Tensor with dtype tf.bool) with same shape as
`nodes`, which marks positions in `nodes` that correspond to valid edges.


## Building SamplingSpec
Function `random_walk_tree` also requires argument `sampling_spec`, which
controls the subgraph size, sampled around seed nodes. For the above example,
`sampling_spec` instance can be built as, e.g.,:

```
f2 = f1 = 5
f3 = 3  # somewhat arbitrary.
builder = tfgnn.SamplingSpecBuilder(graph_schema).seed('papers')
builder.sample(f1, 'cites')
builder.sample(f2, 'rev_writes').sample(f3, 'affiliated_with')

sampling_spec = builder.to_sampling_spec()
```

Each walk tree node will contain graph-node indices in `walk_tree.nodes`.
Further, DAG traversal edges are stored in `walk_tree.next_steps`.
"""

import collections
import enum
import functools
from typing import Any, Tuple, Callable, Mapping, Optional, MutableMapping, List, Union

import numpy as np
import scipy.sparse as ssp
import tensorflow as tf
import tensorflow_gnn as tfgnn

import datasets
from tensorflow_gnn.sampler import sampling_spec_pb2


def process_sampling_spec_topologically(
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    *,
    init_callback: Optional[Callable[[sampling_spec_pb2.SeedOp], Any]] = None,
    process_callback: Optional[
        Callable[[sampling_spec_pb2.SamplingOp], Any]] = None):
  """Processes SamplingSpec topologically, while invoking given callbacks.

  The function is blocking. It will call `init_callback` once, then
  `process_callback` many times (once per sampling op), in topological order,
  then returns.

  Args:
    sampling_spec: with (one) seed_op populated and (possibly many) sampling_ops
      populated. You must verify that `input_op_names` represent a DAG.
    init_callback: will be called exactly once, passing the seed_op.
    process_callback: will be called once for every sampling op.
  """
  child_2_unprocessed_parents = {}  # str -> set of strings
  parent_2_children = collections.defaultdict(list)  # string -> list of ops.
  ready_nodes = [sampling_spec.seed_op]
  for sampling_op in sampling_spec.sampling_ops:
    if not sampling_op.input_op_names:
      ready_nodes.append(sampling_op)
    child_2_unprocessed_parents[sampling_op.op_name] = set(
        sampling_op.input_op_names)

    for parent_name in sampling_op.input_op_names:
      parent_2_children[parent_name].append(sampling_op)

  while ready_nodes:
    node = ready_nodes.pop(0)
    if node == sampling_spec.seed_op:  # Seed op.
      if init_callback is not None:
        init_callback(node)
    else:  # Sampling op.
      if process_callback is not None:
        process_callback(node)
    for child in parent_2_children[node.op_name]:
      child_2_unprocessed_parents[child.op_name].remove(node.op_name)
      if not child_2_unprocessed_parents[child.op_name]:
        # Processed all parents.
        ready_nodes.append(child)


class TypedWalkTree:
  """Data structure for holding tree-like traversals on heterogeneous graphs.

  The data structure is rooted on a batch of nodes [n_1, ..., n_B] as:
  seed_nodes = np.array([n_1, n_2, ..., n_B])  # B-sized vector.

  S = TypedWalkTree(seed_nodes)

  S_authors = S.add_step(
      "authors", np.array(edge_samples("written_by", seed_nodes)))

  where function `edge_samples()` should be implemented to return a matrix of
  shape `(B x f)` where `B` is input batch size (of `seed_nodes`) and `f` is
  "fanout" (number of replicas of random walker).

  More generally, edge_samples() should append a dimension on its input
  `seed_nodes`. Specifically, if input is shaped `[...]` then output must be
  shaped `[..., f]`.

  The intent of this class (`TypedWalkTree`), is to record the randomly-sampled
  subgraphs. This class represents each node as an int, and is not intended to
  store node features or labels. On the other hand, this class is populated by
  `GraphSampler`, by merging tree paths of node IDs (contained in
  `TypedWalkTree`) with node features & labels, into `GraphTensor` instances.
  """

  def __init__(self, nodes: tf.Tensor, owner: Optional['GraphSampler'] = None,
               valid_mask: Optional[tf.Tensor] = None):
    self._nodes = nodes
    self._next_steps = []
    self._owner = owner
    if valid_mask is None:
      self._valid_mask = tf.ones(shape=nodes.shape, dtype=tf.bool)
    else:
      self._valid_mask = valid_mask

  @property
  def nodes(self) -> tf.Tensor:
    """int tf.Tensor with shape `[b, s1, s2, ..., sH]` where `b` is batch size.

    `H` is number of hops (until this sampling step). Each int `si` indicates
    number of nodes sampled at step `i`.
    """
    return self._nodes

  @property
  def valid_mask(self) -> Optional[tf.Tensor]:
    """bool tf.Tensor with same shape of `nodes` marking "correct" samples.

    If entry `valid_mask[i, j, k]` is True, then `nodes[i, j, k]` corresponds to
    a node that is indeed a sampled neighbor of `previous_step.nodes[i, j]`.
    """
    return self._valid_mask

  @property
  def next_steps(self) -> List[Tuple[tfgnn.EdgeSetName, 'TypedWalkTree']]:
    return self._next_steps

  def add_step(self, edge_set_name: tfgnn.EdgeSetName, nodes: tf.Tensor,
               valid_mask: Optional[tf.Tensor] = None,
               propagate_validation: bool = True) -> 'TypedWalkTree':
    if propagate_validation and valid_mask is not None:
      valid_mask = tf.logical_and(tf.expand_dims(self.valid_mask, -1),
                                  valid_mask)
    child_tree = TypedWalkTree(nodes, owner=self._owner, valid_mask=valid_mask)
    self._next_steps.append((edge_set_name, child_tree))
    return child_tree

  def get_edge_lists(self) -> Mapping[tfgnn.EdgeSetName, tf.Tensor]:
    """Constructs sampled edge lists.

    Returns:
      dict with keys being `edge set name` and value as tf.Tensor matrix with
      shape (num_edges, 2). If an edge set was observed multiple times, then
      all sampled edges will be concatenated under the same key.
    """
    edge_lists = collections.defaultdict(list)
    self._get_edge_lists_recursive(edge_lists)
    for k in list(edge_lists.keys()):
      edge_lists[k] = tf.concat(edge_lists[k], axis=-1)
    return edge_lists

  def _get_edge_lists_recursive(
      self, edge_lists: MutableMapping[tfgnn.EdgeSetName, List[tf.Tensor]]):
    for edge_set_name, child_tree in self._next_steps:
      fanout = child_tree.nodes.shape[-1]
      stacked = tf.stack(
          [tf.stack([self.nodes] * fanout, -1), child_tree.nodes], 0)
      edge_lists[edge_set_name].append(tf.reshape(stacked, (2, -1)))
      child_tree._get_edge_lists_recursive(edge_lists)  # Same class. pylint: disable=protected-access

  def to_graph_tensor(
      self,
      node_features_fn: Callable[[str, tf.Tensor], Mapping[str, tf.Tensor]],
      static_sizes: bool = False,
      ) -> tfgnn.GraphTensor:
    """Converts the randomly traversed walk tree into a `GraphTensor`.

    GraphTensor can then be passed to TFGNN models (or readout functions).

    Args:
      node_features_fn: function accepts node set name and node IDs, then
        outputs dictionary of features.
      static_sizes: If set, then GraphTensor will always have same number
        of nodes. Specifically, nodes can be repeated. If not set, then even if
        random trees discover some node multiple times, then it would only
        appear once in node features.

    Returns:
      newly-constructed (not cached) tfgnn.GraphTensor.
    """
    if static_sizes:
      maybe_unique = lambda x: x
    else:
      maybe_unique = lambda x: tf.unique(x).y
    edge_lists = self.get_edge_lists()

    # Node set name -> unique node IDs.
    unique_node_ids = collections.defaultdict(list)

    for edge_set_name, edges in edge_lists.items():
      src_type = self._owner.edge_types[edge_set_name][0]
      dst_type = self._owner.edge_types[edge_set_name][1]
      # Uniqify nodes.
      unique_node_ids[src_type].append(maybe_unique(edges[0]))
      unique_node_ids[dst_type].append(maybe_unique(edges[1]))

    # Node set name -> node IDs used in batch.
    unique_node_ids = {name: tf.sort(maybe_unique(tf.concat(values, 0)))
                       for name, values in unique_node_ids.items()}

    node_sets = {}
    for node_set_name, node_ids in unique_node_ids.items():
      node_sets[node_set_name] = tfgnn.NodeSet.from_fields(
          sizes=as_tensor(node_ids.shape),
          features=node_features_fn(node_set_name, node_ids))

    edge_sets = {}
    for edge_set_name, edges in edge_lists.items():
      src_set_name, dst_set_name = self._owner.edge_types[edge_set_name]
      renumbered_src = tf.searchsorted(unique_node_ids[src_set_name], edges[0])
      renumbered_dst = tf.searchsorted(unique_node_ids[dst_set_name], edges[1])
      edge_sets[edge_set_name] = tfgnn.EdgeSet.from_fields(
          sizes=tf.shape(renumbered_src),
          adjacency=tfgnn.Adjacency.from_indices(
              source=(src_set_name, renumbered_src),
              target=(dst_set_name, renumbered_dst)))

    if static_sizes:
      newshape = np.prod(self.nodes.shape)
    else:
      newshape = -1

    context = tfgnn.Context.from_fields(features={
        'seed_nodes.' + self._owner.dataset.labeled_nodeset: tf.expand_dims(
            tf.searchsorted(
                unique_node_ids[self._owner.dataset.labeled_nodeset],
                tf.reshape(self.nodes, newshape)), 0)
    })

    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets, edge_sets=edge_sets, context=context)

    return graph_tensor


class EdgeSampling(enum.Enum):
  """Enum for randomized strategies for sampling neighbors."""
  # Samples each neighbor independently. It assumes that *every node* has at
  # least one outgoing neighbor, for all sampled edge-sets.
  W_REPLACEMENT = 'w_replacement'

  # Samples each neighbor independently. It assumes that some nodes might have
  # zero outgoing edges. This option causes `sample_one_hop()` to also return
  # `valid_mask` (boolean tf.Tensor) marking positions corresponding to an
  # actual edge, which will be False iff sampling from orphan nodes.
  W_REPLACEMENT_W_ORPHANS = 'w_replacement_w_orphans'

  # Samples neighbors without replacement. However, if (int) `S` neighbors were
  # requested, and there are only `s` neighbors (with `s < S`), then the samples
  # will be repeated. You *must* ensure that each node has at least one outgoing
  # neighbor. If your graph has orphan nodes, use `WO_REPLACEMENT_WO_REPEAT` or
  # `W_REPLACEMENT_W_ORPHANS`.
  WO_REPLACEMENT = 'wo_replacement'

  # Like the above. In cases if some nodes have very few neighbors (less than
  # `sample_size`), then nodes will only be sampled once. This option also works
  # when some nodes have zero outgoing edges.
  # This option causes `sample_one_hop()` to also return `valid_mask` (boolean
  # tf.Tensor) marking positions corresponding to an actual edge.
  WO_REPLACEMENT_WO_REPEAT = 'wo_replacement_wo_repeat'


class GraphSampler:
  """Yields random sub-graphs from TFGNN-wrapped datasets.

  Sub-graphs are encoded as `GraphTensor` or tf.data.Dataset. Random walks are
  performed using `TypedWalkTree`. Input data graph must be an instance of
  `NodeClassificationDatasetWrapper`
  """

  def __init__(self,
               dataset: datasets.NodeClassificationDatasetWrapper,
               make_undirected: bool = False,
               ensure_self_loops: bool = False,
               reduce_memory_footprint: bool = True,
               sampling: EdgeSampling = EdgeSampling.WO_REPLACEMENT):
    self.dataset = dataset
    self.sampling = sampling
    self.edge_types = {}  # edge set name -> (src node set name, dst *).
    self.adjacency = {}

    all_node_counts = dataset.node_counts()
    edge_lists = dataset.edge_lists()
    for edge_type, edges in edge_lists.items():
      src_node_set_name, edge_set_name, dst_node_set_name = edge_type
      self.edge_types[edge_set_name] = (src_node_set_name, dst_node_set_name)
      size_src = all_node_counts[src_node_set_name]
      size_dst = all_node_counts[dst_node_set_name]

      self.adjacency[edge_set_name] = ssp.csr_matrix(
          (np.ones([edges.shape[-1]], dtype='int8'), (edges[0], edges[1])),
          shape=(size_src, size_dst))

      if ensure_self_loops and src_node_set_name == dst_node_set_name:
        # Set diagonal entries.
        self.adjacency[edge_set_name] = (
            ssp.eye(size_src, dtype='int8').maximum(
                self.adjacency[edge_set_name]))

      if make_undirected:
        self.adjacency[edge_set_name] = self.adjacency[edge_set_name].maximum(
            self.adjacency[edge_set_name].T)
      else:
        self.adjacency['rev_' + edge_set_name] = ssp.csr_matrix(
            self.adjacency[edge_set_name].T)

    # Compute data structures required for sampling.
    self.edge_lists = {}      # Edge set name -> (optional src_ids, target_ids).
    self.degrees = {}         # Edge set name -> [deg_1, deg_2, ... deg_|V|].
    self.degrees_cumsum = {}  # Edge set name -> [0, deg_1, deg_1+deg_2. ...].
    for edge_set_name, csr_adj in self.adjacency.items():
      csr_adj = csr_adj > 0  # Binarize.
      nonzero_rows, nonzero_cols = csr_adj.nonzero()
      self.edge_lists[edge_set_name] = (
          None if reduce_memory_footprint else as_tensor(nonzero_rows),
          as_tensor(nonzero_cols))
      degree_vector = as_tensor(np.array(csr_adj.sum(1))[:, 0])
      self.degrees[edge_set_name] = degree_vector
      self.degrees_cumsum[edge_set_name] = tf.math.cumsum(
          degree_vector, exclusive=True)

    if reduce_memory_footprint:
      self.adjacency = None

  def make_sample_layer(self, edge_set_name, sample_size=3, sampling=None):
    # Function only accepts source_nodes.
    return functools.partial(
        self.sample_one_hop, edge_set_name=edge_set_name,
        sample_size=sample_size, sampling=sampling)

  def sample_one_hop(
      self, source_nodes: tf.Tensor, edge_set_name: tfgnn.EdgeSetName,
      sample_size: int, sampling: Optional[EdgeSampling] = None,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Samples one-hop from source-nodes using edge `edge_set_name`."""
    if sampling is None:
      sampling = EdgeSampling.WO_REPLACEMENT

    all_degrees = self.degrees[edge_set_name]
    node_degrees = tf.gather(all_degrees, source_nodes)

    offsets = self.degrees_cumsum[edge_set_name]

    next_nodes = valid_mask = None  # Answer, to be populated, below.

    if sampling in (EdgeSampling.W_REPLACEMENT,
                    EdgeSampling.W_REPLACEMENT_W_ORPHANS):
      sample_indices = tf.random.uniform(
          shape=source_nodes.shape + [sample_size], minval=0, maxval=1,
          dtype=tf.float32)

      node_degrees_expanded = tf.expand_dims(node_degrees, -1)
      sample_indices = sample_indices * tf.cast(node_degrees_expanded,
                                                tf.float32)

      # According to https://www.pcg-random.org/posts/bounded-rands.html, this
      # sample is biased. NOTE: we plan to adopt one of the linked alternatives.
      sample_indices = tf.cast(tf.math.floor(sample_indices), tf.int64)

      if sampling == EdgeSampling.W_REPLACEMENT_W_ORPHANS:
        valid_mask = sample_indices < node_degrees_expanded

      # Shape: (sample_size, nodes_reshaped.shape[0])
      sample_indices += tf.expand_dims(tf.gather(offsets, source_nodes), -1)
      nonzero_cols = self.edge_lists[edge_set_name][1]
      if sampling == EdgeSampling.W_REPLACEMENT_W_ORPHANS:
        sample_indices = tf.where(
            valid_mask, sample_indices, tf.zeros_like(sample_indices))
      next_nodes = tf.gather(nonzero_cols, sample_indices)
    elif sampling in (EdgeSampling.WO_REPLACEMENT,
                      EdgeSampling.WO_REPLACEMENT_WO_REPEAT):
      # shape=(total_input_nodes).
      nodes_reshaped = tf.reshape(source_nodes, [-1])
      # shape=(total_input_nodes).
      reshaped_node_degrees = tf.reshape(node_degrees, [-1])
      reshaped_node_degrees_or_1 = tf.maximum(
          reshaped_node_degrees, tf.ones_like(reshaped_node_degrees))
      # shape=(sample_size, total_input_nodes).
      sample_upto = tf.stack([reshaped_node_degrees] * sample_size, axis=0)

      # [[0, 1, 2, ..., f], <repeated>].T
      subtract_mod = tf.stack(
          [tf.range(sample_size, dtype=tf.int64)] * nodes_reshaped.shape[0],
          axis=-1)
      if sampling == EdgeSampling.WO_REPLACEMENT_WO_REPEAT:
        valid_mask = subtract_mod < reshaped_node_degrees
        valid_mask = tf.reshape(
            tf.transpose(valid_mask), source_nodes.shape + [sample_size])

      subtract_mod = subtract_mod % tf.maximum(
          sample_upto, tf.ones_like(sample_upto))

      # [[d, d-1, d-2, ... 1, d, d-1, ...]].T
      # where 'd' is degree of node in row corresponding to nodes_reshaped.
      sample_upto -= subtract_mod

      max_degree = tf.reduce_max(node_degrees)

      sample_indices = tf.random.uniform(
          shape=[sample_size, nodes_reshaped.shape[0]], minval=0, maxval=1,
          dtype=tf.float32)
      # (sample_size, num_sampled_nodes)
      sample_indices = sample_indices * tf.cast(sample_upto, tf.float32)
      # According to https://www.pcg-random.org/posts/bounded-rands.html, this
      # sample is biased. NOTE: we plan to adopt one of the linked alternatives.
      sample_indices = tf.cast(tf.math.floor(sample_indices), tf.int64)

      adjusted_sample_indices = [sample_indices[0]]
      already_sampled = sample_indices[:1]  # (1, total_input_nodes)

      for i in range(1, sample_size):
        already_sampled = tf.where(
            i % reshaped_node_degrees_or_1 == 0,
            tf.ones_like(already_sampled) * max_degree, already_sampled)
        next_sample = sample_indices[i]
        for j in range(i):
          next_sample += tf.cast(next_sample >= already_sampled[j], tf.int64)
        adjusted_sample_indices.append(next_sample)
        # Register as already-sampled.
        already_sampled = tf.concat(
            [already_sampled, tf.expand_dims(next_sample, 0)], axis=0)
        already_sampled = tf.sort(already_sampled, axis=0)

      # num nodes, sample_size
      adjusted_sample_indices = tf.stack(adjusted_sample_indices, axis=0)
      # Shape: (sample_size, total_input_nodes)
      sample_indices = adjusted_sample_indices

      sample_indices += tf.expand_dims(tf.gather(offsets, nodes_reshaped), 0)
      sample_indices = tf.reshape(tf.transpose(sample_indices),
                                  source_nodes.shape + [sample_size])
      nonzero_cols = self.edge_lists[edge_set_name][1]
      if sampling == EdgeSampling.WO_REPLACEMENT_WO_REPEAT:
        sample_indices = tf.where(
            valid_mask, sample_indices, tf.zeros_like(sample_indices))

      next_nodes = tf.gather(nonzero_cols, sample_indices)
    else:
      raise ValueError('Unknown sampling ' + str(sampling))

    if next_nodes.dtype != source_nodes.dtype:
      # It could happen, e.g., if edge-list is int32 and input seed is int64.
      next_nodes = tf.cast(next_nodes, source_nodes.dtype)

    if valid_mask is None:
      return next_nodes
    else:
      return next_nodes, valid_mask

  def generate_subgraphs(
      self, batch_size: int,
      sampling_spec: sampling_spec_pb2.SamplingSpec,
      split: str = 'train',
      sampling=EdgeSampling.WO_REPLACEMENT):
    """Infinitely yields random subgraphs each rooted on node in train set."""
    if isinstance(split, bytes):
      split = split.decode()
    if not isinstance(split, (tuple, list)):
      split = (split,)

    partitions = self.dataset.node_split()

    node_ids = tf.concat([getattr(partitions, s) for s in split], 0)
    queue = tf.random.shuffle(node_ids)

    while True:
      while queue.shape[0] < batch_size:
        queue = tf.concat([queue, tf.random.shuffle(node_ids)], axis=0)
      batch = queue[:batch_size]
      queue = queue[batch_size:]
      yield self.sample_sub_graph_tensor(
          batch, sampling_spec=sampling_spec, sampling=sampling)

  def random_walk_tree(
      self, node_idx: tf.Tensor, sampling_spec: sampling_spec_pb2.SamplingSpec,
      sampling: EdgeSampling = EdgeSampling.WO_REPLACEMENT) -> TypedWalkTree:
    """Returns `TypedWalkTree` where `nodes` are seed root-nodes.

    Args:
      node_idx: int tf.Tensor containing node IDs to seed the random walk trees.
        From each seed node in `nodes`, a random walk tree will be constructed.
      sampling_spec: to guide sampling (number of hops & number of nodes per
        hop). It can be built using `sampling_spec_builder`.
      sampling: to spcify with or without replacement.

    Returns:
      `TypedWalkTree` where each edge is sampled uniformly.
    """
    op_name_to_tree: MutableMapping[str, TypedWalkTree] = {}
    seed_op_names = []

    def process_seed_op(sampling_op: sampling_spec_pb2.SeedOp):
      seed_op_names.append(sampling_op.op_name)
      op_name_to_tree[sampling_op.op_name] = TypedWalkTree(node_idx, owner=self)

    def process_sampling_op(sampling_op: sampling_spec_pb2.SamplingOp):
      parent_trees = [op_name_to_tree[op_name]
                      for op_name in sampling_op.input_op_names]
      if len(parent_trees) > 1:
        raise ValueError(
            'Multiple paths for sampling is not yet supported. To support, you '
            'can extend TypedWalkTree into WalkDAG.')
      parent_trees = parent_trees[:1]
      parent_nodes = [tree.nodes for tree in parent_trees]
      parent_nodes = tf.concat(parent_nodes, axis=1)

      next_nodes = self.sample_one_hop(
          parent_nodes, sampling_op.edge_set_name,
          sample_size=sampling_op.sample_size, sampling=sampling)
      if isinstance(next_nodes, tuple):
        next_nodes, valid_mask = next_nodes
        child_tree = parent_trees[0].add_step(
            sampling_op.edge_set_name, next_nodes, valid_mask=valid_mask)
      else:
        child_tree = parent_trees[0].add_step(
            sampling_op.edge_set_name, next_nodes)

      op_name_to_tree[sampling_op.op_name] = child_tree

    process_sampling_spec_topologically(
        sampling_spec, process_callback=process_sampling_op,
        init_callback=process_seed_op)
    if len(seed_op_names) != 1:
      raise ValueError('Expecting exactly one seed.')

    return op_name_to_tree[seed_op_names[0]]

  def sample_sub_graph_tensor(
      self, node_idx: tf.Tensor, sampling_spec: sampling_spec_pb2.SamplingSpec,
      sampling: EdgeSampling = EdgeSampling.WO_REPLACEMENT
      ) -> tfgnn.GraphTensor:
    """Samples GraphTensor starting from seed nodes `node_idx`.

    Args:
      node_idx: (int) tf.Tensor of node indices to seed random-walk trees.
      sampling_spec: Specifies the hops (edge set names) to be sampled, and the
        number of sampled edges per hop.
      sampling: If `== EdgeSampling.W_REPLACEMENT`, then neighbors for a node
        will be sampled uniformly and indepedently. If
        `== EdgeSampling.WO_REPLACEMENT`, then a node's neighbors will be
        chosen in (random) round-robin order. If more samples are requested are
        larger than neighbors, then the samples will be repeated (each time, in
        a different random order), such that, all neighbors appears exactly the
        same number of times (+/- 1, if sample_size % neighbors != 0).

    Returns:
      `tfgnn.GraphTensor` containing subgraphs traversed as random trees rooted
      on input `node_idx`.
    """
    walk_tree = self.random_walk_tree(
        node_idx, sampling_spec=sampling_spec, sampling=sampling)
    return walk_tree.to_graph_tensor(self.gather_node_features_dict)

  def gather_node_features_dict(self, node_set_name, node_idx):
    features = self.dataset.node_features_dicts(add_id=True)[node_set_name]
    features = {feature_name: tf.gather(feature_value, node_idx)
                for feature_name, feature_value in features.items()}
    if node_set_name == self.dataset.labeled_nodeset:
      features['label'] = tf.gather(self.dataset.labels(), node_idx)

    return features


def make_sampled_subgraphs_dataset(
    dataset: datasets.NodeClassificationDatasetWrapper,
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    batch_size: int = 64,
    split='train',
    make_undirected: bool = False,
    sampling=EdgeSampling.WO_REPLACEMENT
    ) -> Tuple[tf.TensorSpec, tf.data.Dataset]:
  """Infinite tf.data.Dataset wrapping generate_subgraphs."""
  subgraph_generator = GraphSampler(dataset, make_undirected=make_undirected)
  relaxed_spec = None
  for graph_tensor in subgraph_generator.generate_subgraphs(
      batch_size, split=split, sampling_spec=sampling_spec, sampling=sampling):
    # relaxed_spec = _get_relaxed_spec_from_graph_tensor(graph_tensor)
    relaxed_spec = graph_tensor.spec.relax(num_nodes=True, num_edges=True)
    break

  assert relaxed_spec is not None
  bound_generate_fn = functools.partial(
      subgraph_generator.generate_subgraphs, sampling_spec=sampling_spec,
      sampling=sampling, split=split, batch_size=batch_size)

  tf_dataset = tf.data.Dataset.from_generator(
      bound_generate_fn,
      output_signature=relaxed_spec)

  return relaxed_spec, tf_dataset


# Can be replaced with: `_t = tf.convert_to_tensor`.
def as_tensor(obj: Any) -> tf.Tensor:
  """short-hand for tf.convert_to_tensor."""
  return tf.convert_to_tensor(obj)
