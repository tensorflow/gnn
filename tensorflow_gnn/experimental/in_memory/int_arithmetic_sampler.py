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
r"""Samples one-hop edges and multi-hop subgraphs from `InMemoryGraphData`.

Class `GraphSampler` provide sampling logic. It provides various APIs:

1. One-hop sampling. Method `sample_one_hop()` accepts `int tf.Tensor` input
   node IDs, and output IDs of neighbors to input. Keras-compatible layer can be
   constructed as `make_sampling_layer()`.

2. Sampling for Runner. Multi-hop sampling. Method `sample_walk_tree` returns a
   data structure that have methods `as_graph_tensor` and `as_tensor_dict`,
   respectively which store the multi-hop sampled graph as `tfgnn.GraphTensor`
   and as python dict (where keys denote sampling path and `tf.Tensor` values).

For node classification tasks, the following are also provided:

1. Class `NodeClassificationGraphSampler` extends `GraphSampler` for
   node-classification graph data. Specifically, it offers methods that can
   populate `context` of `GraphTensor` to contain seed node positions, in
   addition to exporting node labels as part of the features.

2. Method `NodeClassificationGraphSampler.as_dataset` show-cases how to create
   `tf.data.Dataset` of sampled subgraphs.


# Usage Examples

```
# Load the graph data.
from tensorflow_gnn.examples.in_memory import datasets
graph_dataset_name = 'ogbn-arxiv'  # or {ogbn-*, cora, citeseer, pubmed}
inmem_ds = datasets.get_in_memory_graph_data(dataset_name)

# Craft sampling specification.
graph_schema = inmem_ds.export_graph_schema()
```

For multi-hop sampling, instance of `SamplingSpec` is required. However, one-hop
sampling does not use `SamplingSpec`.


## Usage example of `GraphSampler()`

```
graph_data = in_memory.datasets.get_in_memory_graph_data('ogbn-arxiv')
# or graph_data = datasets.UnigraphData(unigraph.read_schema("schema.pbtxt"))
sampler = GraphSampler(graph_data)
# By default, sampler uses `WITHOUT_REPLACEMENT`. It can be override with, e.g,
# `sampling_mode=EdgeSampling.WITH_REPLACEMENT`.


src_node_ids = tf.constant([0, 1, 2, 3, 4])

# Sample 20 `tgt` nodes per `src` node, from edge set "edges".

tgt_node_ids = sampler.sample_one_hop(src_node_ids, "edges", 20)

# You may also:

tgt_node_ids, valid_mask = sampler.sample_one_hop_with_valid_mask(
    src_node_ids, "edges", 20)
```

Shapes of `tgt_node_ids` and `valid_mask` are both `[5, 20]`. For node with ID
`src_node_ids[0]` contains 20 sampled neighbors in tgt_node_ids[0], etc.
`valid_mask` contain binary entries, marking positions of `tgt_node_ids` that
correspond to valid sampling.

Given `SamplingSpec` (below), instance of `GraphSampler` can sample multi-hops:

```
walk_tree = sampler.sample_walk_tree(src_node_ids, sampling_spec)
tensor_dict = walk_tree.as_tensor_dict()

print(tensor_dict["seed"],  # `tf.Tensor` with shape `B == src_node_ids.shape`.
      tensor_dict["seed.edges"],
      tensor_dict["seed.edges.edges"])
```

where `sampling_spec` (type `SamplingSpec`) configures sampled subgraph size.
For homogeneous graph, you can collect up-to `Bx16x16` edges with:

```
sampling_spec = (tfgnn.SamplingSpecBuilder(graph_schema)
                 .seed().sample([16, 16]).build())
```

where `B == src_node_ids.shape`.

Further, `TypedWalkTree` instances have attributes `nodes` and `next_steps`,
respectively, storing node IDs at the traversal step, and next-step samples
(pair of (edge name, `TypedWalkTree`)).

## Usage Example of `NodeClassificationGraphSampler.as_dataset`.

```
dataset = NodeClassificationGraphSampler(graph_data).as_dataset(sampling_spec)

for graph in dataset:
  print(graph)
  break

# Alternatively: `my_model.fit(dataset)`, where `my_model` is a `keras.Model`,
# e.g., composed of `tfgnn.keras.layers`.
# You may refer to examples/keras_minibatch_trainer.py
```

# Algorithm & Implementation

## Pre-processing

When an instance of `GraphSampler` is constructed, two data structures are
initialized *per edge set*:

```
edges = [
    (0, 45),        # edge 0 -> 45
    (0, 1),         # edge 0 -> 1
    (0, 13), ...,   # Total of 305 n1 edges.

    (1, 51),
    (1, 893), ...,

    (2, 0),
    ...
]
```

and

```
node_degrees = [
    305,  # out-degree of node 0.
    ...,  # out-degree of node 1.
    ...
]
```

Both of which are stored as tf.Tensor. `edges` are sorted by first column.
(note: first column is redudntant: it can be reconstructed from `node_degrees`).


## Invoking Sampling

Once `GraphSampler()` is constructed, sampling can be invoked as:

  * `sample_one_hop`, described above.
  * `sample_walk_tree`, returning a `TypedWalkTree` instance, storing multi-hop
    sampling as a sampling tree and `.as_tensor_dict` gives dict of `tf.Tensor`.

All with io of `tf.Tensor`s, i.e., can be in scope of `@tf.function`.


The seed node Tensor initializes the `TypedWalkTree`, which can be depicted as:

```
                 sample(f1, 'cites')
    paper --------------------------> paper
         \
          \ sample(f2, 'rev_writes')            sample(f3, 'affiliated_with')
           ---------------------------> author ------------------> institution
```

Instance nodes of `TypedWalkTree` (above) have attribute `nodes` with shapes:
(B), (B, f1), (B, f2), (B, f2, f3) -- (left-to-right). All are `tf.Tensor`s
with dtype `tf.int{32, 64}`, matching the dtype of its input argument.

Function `sample_walk_tree` also requires argument `sampling_spec`, which
controls the subgraph size, sampled around seed nodes. For the above example,
`sampling_spec` instance can be built as, e.g.,:


```
f2 = f1 = 5
f3 = 3  # somewhat arbitrary.
builder = (
    tfgnn.SamplingSpecBuilder(
        graph_schema,
        sampling_spec_builder.SamplingStrategy.RANDOM_UNIFORM)
    .seed('papers'))
builder.sample(f1, 'cites')
builder.sample(f2, 'rev_writes').sample(f3, 'affiliated_with')

sampling_spec = builder.build()
```
"""

import collections
import enum
import functools
from typing import Any, Tuple, Callable, Mapping, Optional, MutableMapping, List, Dict, Union

import numpy as np
import scipy.sparse as ssp
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.in_memory import datasets
from tensorflow_gnn.experimental.in_memory import reader_utils
from tensorflow_gnn.experimental.sampler import interfaces
from tensorflow_gnn.graph import tensor_utils as utils
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
    self._nodes: tf.Tensor = nodes
    self._next_steps: List[Tuple[tfgnn.EdgeSetName, TypedWalkTree]] = []
    self._owner: Optional[GraphSampler] = owner
    if valid_mask is None:
      shape = nodes.shape if nodes.shape[0] is not None else tf.shape(nodes)
      self._valid_mask = tf.ones(shape=shape, dtype=tf.bool)
    else:
      self._valid_mask = valid_mask

  def as_tensor_dict(self) -> Mapping[str, tf.Tensor]:
    """Flattens walk-tree making dict, path (e.g.,"seed.edge.edge") to node IDs.

    For instance, this could return:

    ```
    {
        "seed": tf.Tensor  #  with shape  `B`, where `B == self.nodes.shape`
        "seed.edge": tf.Tensor,  # w shape `[B, step_1_num_samples]`
        "seed.edge.edge": ... ,  # `[B, step_1_num_samples, step_2_num_samples]`
        ...
    }
    ```

    With `len() == ` number of sampling hops (+1, for storing `"seed"`).

    Returns:
      `dict(str: tf.Tensor)`, as decribed.
    """
    return self._as_tensor_dict_recursive({}, self, ['seed'])

  def _as_tensor_dict_recursive(  # Private helper for as_tensor_dict.
      self, result_dict: Dict[str, tf.Tensor],
      root: 'TypedWalkTree', path: List[str]) -> Mapping[str, tf.Tensor]:
    """Recursively populates result_dict with to tensors."""
    path_str = '/'.join(path)
    i = 0
    while path_str in result_dict:
      i += 1
      path_str = '/'.join(path) + '.%i' % i
    result_dict[path_str] = root.nodes
    for edge_set_name, typed_walk_tree in root.next_steps:
      self._as_tensor_dict_recursive(
          result_dict,
          typed_walk_tree, path + [edge_set_name])
    return result_dict

  @property
  def nodes(self) -> tf.Tensor:
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
               inherit_validity: bool = True) -> 'TypedWalkTree':
    """Adds one step on onto the walk-tree as a new `TypedWalkTree`."""
    if valid_mask is None:
      valid_mask = tf.ones(shape=nodes.shape, dtype=tf.bool)

    if inherit_validity:
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
    """Recursively accumulates into `edge_lists` traversed edges."""
    for edge_set_name, child_tree in self._next_steps:
      fanout = child_tree.nodes.shape[-1]
      stacked = tf.stack(
          [tf.stack([self.nodes] * fanout, -1), child_tree.nodes], 0)

      reshaped = tf.reshape(stacked, [2, -1])
      valid_mask = tf.reshape(child_tree.valid_mask, [-1])
      reshaped = tf.transpose(
          tf.boolean_mask(tf.transpose(reshaped), valid_mask))

      edge_lists[edge_set_name].append(reshaped)
      child_tree._get_edge_lists_recursive(edge_lists)  # Same class. pylint: disable=protected-access

  def as_graph_tensor(
      self,
      node_features_fn: Callable[
          [tfgnn.NodeSetName, tf.Tensor], Mapping[tfgnn.FieldName, tf.Tensor]],
      static_sizes: bool = False
      ) -> tfgnn.GraphTensor:
    """Converts the randomly traversed walk tree into a `GraphTensor`.

    GraphTensor can then be passed to TFGNN models (or readout functions).

    Args:
      node_features_fn: function accepts (node set name, node IDs). It should
        output dict of features: feature name -> feature matrix, where leading
        dimensions of feature matrix must equal to shape of input node IDs.
      static_sizes: If set, then GraphTensor will always have same number of
        nodes and edges. Specifically, nodes can be repeated. If not set, then
        even if random trees discover some node multiple times, then it would
        only appear once in node features.

    Returns:
      newly-constructed tfgnn.GraphTensor.
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
      if node_ids.shape[0]:
        sizes = as_tensor(node_ids.shape)
      else:
        sizes = tf.shape(node_ids)

      node_sets[node_set_name] = tfgnn.NodeSet.from_fields(
          sizes=sizes,
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

    context = self._owner.create_context(unique_node_ids, self.nodes)

    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets, edge_sets=edge_sets, context=context)

    return graph_tensor


class EdgeSampling(enum.Enum):
  WITH_REPLACEMENT = 'with_replacement'
  WITHOUT_REPLACEMENT = 'without_replacement'


class EdgeSampler(tf.keras.layers.Layer, interfaces.OutgoingEdgesSampler):
  """Samples neighbors given nodes. Follows Edge-sampling API.

  To an instance, EdgeSampler you must first create `sampler = GraphSampler()`.
  Then:

  ```python
  edge_sampler = sampler.make_edge_sampler(
      sampling_spec_pb2.SamplingOp(
          edge_set_name="nameOfEdgeSet", sample_size=10))
  ```

  Finally, you may invoke as:

  ```python
  nodes = tf.ragged.constant([[0, 1], [2]])
  edges = edge_sampler(nodes)  # Must be dict with keys "#source" and "#target".

  print(edges['#source'])  # Should print ragged tensor with source node IDs,
                           # e.g., [[0, 0, 0, 1, 1], [2]], if node 0 has 3
                           # connections, node 1 has 2 connections, and node 2
                           # has only one connection.
  print(edges['#target'])  # Must be same shape as above, e.g.,
                           # [[5, 6, 7, 8, 9], [20]], implying sampled edges
                           # 0-5, 0-6, 0-7, 1-8, 1-9, and 2-20.
  ```
  """

  def __init__(
      self, sampler: 'GraphSampler', sample_size: int,
      edge_set_name: tfgnn.EdgeSetName,
      sampling_mode: Optional[EdgeSampling] = None):
    super().__init__()
    self._sampler = sampler
    self._edge_set_name = edge_set_name
    self._sample_size = sample_size
    self._sampling_mode = sampling_mode

  def call(self, source_node_ids: Union[tf.Tensor, tf.RaggedTensor]) -> Mapping[
      str, tf.RaggedTensor]:
    endpoint_spec = tf.RaggedTensorSpec(
        shape=[None], dtype=source_node_ids.dtype, ragged_rank=0)
    edges_src, edges_tgt = tf.map_fn(
        self._sample_from_tensor_node_ids, source_node_ids,
        fn_output_signature=(endpoint_spec, endpoint_spec))
    return {tfgnn.SOURCE_NAME: edges_src, tfgnn.TARGET_NAME: edges_tgt}

  def _sample_from_tensor_node_ids(self, nodes: tf.Tensor) -> Tuple[
      tf.Tensor, tf.Tensor]:
    """Given dense `nodes` (of any shape), returns (num_edges, 2) Tensor."""
    src = tf.expand_dims(nodes, -1)  # In case `nodes` is scalar.
    tgt, valid_mask = self._sampler.sample_one_hop_with_valid_mask(
        src, edge_set_name=self._edge_set_name, sample_size=self._sample_size,
        sampling_mode=self._sampling_mode)
    # Now, `tgt` has an additional dimension over `src`. The size of this
    # (last) dimension must be equal to `self._sample_size`. Let's repeat `src`
    # along that axis:
    src = tf.expand_dims(src, -1) + tf.zeros_like(tgt)

    # Filter only to valid tgt (and associated valid src) and return pair
    valid_reshaped = tf.reshape(valid_mask, [-1])
    valid_src = tf.boolean_mask(tf.reshape(src, [-1]), valid_reshaped)
    valid_tgt = tf.boolean_mask(tf.reshape(tgt, [-1]), valid_reshaped)
    return valid_src, valid_tgt


class GraphSampler:
  """Yields random sub-graphs from `InMemoryGraphData`.

  Sub-graphs are encoded as `GraphTensor` or tf.data.Dataset. Random walks are
  performed using `TypedWalkTree`. Input data graph must be an instance of
  `Dataset`.
  """

  def __init__(self,
               graph_data: datasets.InMemoryGraphData,
               reduce_memory_footprint: bool = True,
               sampling_mode: EdgeSampling = EdgeSampling.WITHOUT_REPLACEMENT):
    self.graph_data = graph_data
    self.sampling_mode = sampling_mode
    self.edge_types = {}  # edge set name -> (src node set name, dst *).
    self.adjacency = {}

    all_node_counts = graph_data.node_counts()
    edge_sets = graph_data.edge_sets()
    for edge_set_name, edge_set in edge_sets.items():
      self.edge_types[edge_set_name] = (edge_set.adjacency.source_name,
                                        edge_set.adjacency.target_name)
      size_src = all_node_counts[edge_set.adjacency.source_name]
      size_tgt = all_node_counts[edge_set.adjacency.target_name]
      edges_src = edge_set.adjacency.source.numpy()  # Assumption: in-memory.
      edges_tgt = edge_set.adjacency.target.numpy()

      self.adjacency[edge_set_name] = ssp.csr_matrix(
          (np.ones(edges_src.shape, dtype='int8'), (edges_src, edges_tgt)),
          shape=(size_src, size_tgt))

    if not edge_sets:
      raise ValueError('graph_data has no edge-sets.')

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

  def make_edge_sampler(
      self, sampling_op: sampling_spec_pb2.SamplingOp) -> EdgeSampler:
    """Makes layer out of `sample_one_hop`."""
    available_edge_set_names = self.graph_data.edge_sets().keys()
    # Validation.
    edge_set_name = sampling_op.edge_set_name

    if (sampling_op.strategy
        != sampling_spec_pb2.SamplingStrategy.RANDOM_UNIFORM):
      raise ValueError(
          'int-arithmetic sampler currently only supports strategy '
          'RANDOM_UNIFORM')

    if edge_set_name is None:
      if len(available_edge_set_names) > 1:
        raise ValueError(
            'You must provide `edge_set_name` as your graph has multiple edge '
            'sets: ' + ', '.join(available_edge_set_names))
      edge_set_name = list(available_edge_set_names)[0]
    else:
      if edge_set_name not in available_edge_set_names:
        raise ValueError('Edge-set "%s" is not one of: %s' % (
            edge_set_name, ', '.join(available_edge_set_names)))

    return EdgeSampler(
        self, sampling_op.sample_size, edge_set_name, self.sampling_mode)

  def sample_one_hop(
      self, source_nodes: tf.Tensor, edge_set_name: tfgnn.EdgeSetName,
      sample_size: int,
      **kwargs) -> tf.Tensor:
    """Samples one-hop from source-nodes using edge `edge_set_name`.

    Args:
      source_nodes: forwarded to `self.sample_one_hop_with_valid_mask()`.
      edge_set_name: forwarded to `self.sample_one_hop_with_valid_mask()`.
      sample_size: forwarded to `self.sample_one_hop_with_valid_mask()`.
      **kwargs: forwarded to `self.sample_one_hop_with_valid_mask()`.

    Returns:
      sample_one_hop_with_valid_mask(**kwargs)[0]
    """
    tgt_node_ids, unused_valid_mask = self.sample_one_hop_with_valid_mask(
        source_nodes, edge_set_name, sample_size, **kwargs)
    return tgt_node_ids

  def sample_one_hop_with_valid_mask(
      self, source_nodes: tf.Tensor, edge_set_name: tfgnn.EdgeSetName,
      sample_size: int,
      sampling_mode: Optional[EdgeSampling] = None,
      validate=True) -> Tuple[tf.Tensor, tf.Tensor]:
    """Like sample_one_hop(), but returns also `valid_mask`, per header doc."""
    if sampling_mode is None:
      sampling_mode = self.sampling_mode

    all_degrees = self.degrees[edge_set_name]
    node_degrees = tf.gather(all_degrees, source_nodes)

    offsets = self.degrees_cumsum[edge_set_name]

    if source_nodes.shape[0] is None:
      newshape = tf.shape(source_nodes)
      newshape = tf.concat([newshape, [sample_size]], axis=0)
    else:
      newshape = source_nodes.shape + [sample_size]

    if sampling_mode == EdgeSampling.WITH_REPLACEMENT:
      sample_indices = tf.random.uniform(
          shape=newshape, minval=0, maxval=1,
          dtype=tf.float32)
      node_degrees_expanded = tf.expand_dims(node_degrees, -1)

      # TODO(b/256045133): This line does not work if node_degrees has large
      # values (e.g., >billions). Currently, this code is designed for in-memory
      # graphs, i.e., <100M edges.
      sample_indices = sample_indices * tf.cast(
          node_degrees_expanded, tf.float32)

      # According to https://www.pcg-random.org/posts/bounded-rands.html, this
      # sample is biased. NOTE: we plan to adopt one of the linked alternatives.
      sample_indices = tf.cast(tf.math.floor(sample_indices), tf.int64)
      valid_mask = sample_indices < node_degrees_expanded

      # Shape: (sample_size, nodes_reshaped.shape[0])
      sample_indices += tf.expand_dims(tf.gather(offsets, source_nodes), -1)
      nonzero_cols = self.edge_lists[edge_set_name][1]
    elif sampling_mode == EdgeSampling.WITHOUT_REPLACEMENT:
      # shape=(total_input_nodes).
      nodes_reshaped = tf.reshape(source_nodes, [-1])

      # shape=(total_input_nodes).
      reshaped_node_degrees = tf.reshape(node_degrees, [-1])
      reshaped_node_degrees_or_1 = tf.maximum(
          reshaped_node_degrees, tf.ones_like(reshaped_node_degrees))
      # shape=(sample_size, total_input_nodes).
      sample_upto = tf.stack([reshaped_node_degrees] * sample_size, axis=0)

      # [[0, 1, 2, ..., f], <repeated>].T
      if nodes_reshaped.shape[0]:
        subtract_mod = tf.stack(
            [tf.range(sample_size, dtype=tf.int64)] * nodes_reshaped.shape[0],
            axis=-1)
        sample_size_x_num_input_nodes = subtract_mod.shape
      else:
        subtract_mod = tf.transpose(utils.repeat(
            tf.expand_dims(tf.range(sample_size, dtype=tf.int64), 0),
            tf.shape(nodes_reshaped)[:1]))
        sample_size_x_num_input_nodes = tf.shape(subtract_mod)

      valid_mask = subtract_mod < reshaped_node_degrees
      valid_mask = tf.reshape(
          tf.transpose(valid_mask), newshape)

      subtract_mod = subtract_mod % tf.maximum(
          sample_upto, tf.ones_like(sample_upto))
      # [[d, d-1, d-2, ... 1, d, d-1, ...]].T
      # where 'd' is degree of node in row corresponding to nodes_reshaped.
      sample_upto -= subtract_mod

      max_degree = tf.reduce_max(node_degrees)

      sample_indices = tf.random.uniform(
          shape=sample_size_x_num_input_nodes, minval=0, maxval=1,
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
                                  newshape)

      if valid_mask.shape != sample_indices.shape:
        valid_mask = tf.reshape(valid_mask, sample_indices.shape)

      nonzero_cols = self.edge_lists[edge_set_name][1]
    else:
      raise ValueError('Unknown sampling ' + str(sampling_mode))

    if validate:
      sample_indices = tf.where(
          valid_mask, sample_indices, tf.zeros_like(sample_indices))

    next_nodes = tf.gather(nonzero_cols, sample_indices)

    if next_nodes.dtype != source_nodes.dtype:
      # It could happen, e.g., if edge-list is int32 and input seed is int64.
      next_nodes = tf.cast(next_nodes, source_nodes.dtype)

    return next_nodes, valid_mask

  def sample_walk_tree(
      self, node_idx: tf.Tensor, sampling_spec: sampling_spec_pb2.SamplingSpec,
      sampling_mode: Optional[EdgeSampling] = None) -> TypedWalkTree:
    """Returns `TypedWalkTree` where `nodes` are seed root-nodes.

    Args:
      node_idx: int tf.Tensor containing node IDs to seed the random walk trees.
        From each seed node in `nodes`, a random walk tree will be constructed.
      sampling_spec: to guide sampling (number of hops & number of nodes per
        hop). It can be built using `sampling_spec_builder`.
      sampling_mode: to spcify with or without replacement.

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
      if (sampling_op.strategy !=
          sampling_spec_pb2.SamplingStrategy.RANDOM_UNIFORM):
        raise ValueError('sampling_op.strategy must be "RANDOM_UNIFORM".')
      parent_trees = parent_trees[:1]
      parent_nodes = [tree.nodes for tree in parent_trees]
      parent_nodes = tf.concat(parent_nodes, axis=1)

      next_nodes, valid_mask = self.sample_one_hop_with_valid_mask(
          parent_nodes, sampling_op.edge_set_name,
          sample_size=sampling_op.sample_size, sampling_mode=sampling_mode)
      child_tree = parent_trees[0].add_step(
          sampling_op.edge_set_name, next_nodes, valid_mask=valid_mask)

      op_name_to_tree[sampling_op.op_name] = child_tree

    process_sampling_spec_topologically(
        sampling_spec, process_callback=process_sampling_op,
        init_callback=process_seed_op)
    if len(seed_op_names) != 1:
      raise ValueError('Expecting exactly one seed.')

    return op_name_to_tree[seed_op_names[0]]

  def sample_sub_graph(
      self, node_idx: tf.Tensor, sampling_spec: sampling_spec_pb2.SamplingSpec,
      sampling_mode: Optional[EdgeSampling] = None,
      node_feature_gather_fn: Optional[
          Callable[[str, tf.Tensor], Mapping[str, tf.Tensor]]] = None,
      static_sizes: bool = False,
      ) -> tfgnn.GraphTensor:
    """Samples GraphTensor starting from seed nodes `node_idx`.

    Args:
      node_idx: (int) tf.Tensor of node indices to seed random-walk trees.
      sampling_spec: Specifies the hops (edge set names) to be sampled, and the
        number of sampled edges per hop.
      sampling_mode: If `== EdgeSampling.WITH_REPLACEMENT`, then neighbors for a
        node will be sampled uniformly and indepedently. If
        `== EdgeSampling.WITHOUT_REPLACEMENT`, then a node's neighbors will be
        chosen in (random) round-robin order. If more samples are requested are
        larger than neighbors, then the samples will be repeated (each time, in
        a different random order), such that, all neighbors appears exactly the
        same number of times (+/- 1, if sample_size % neighbors != 0).
      node_feature_gather_fn: Forwarded to as_graph_tensor.
      static_sizes: Forwarded to as_graph_tensor.

    Returns:
      `tfgnn.GraphTensor` containing subgraphs traversed as random trees rooted
      on input `node_idx`.
    """
    walk_tree = self.sample_walk_tree(
        node_idx, sampling_spec=sampling_spec, sampling_mode=sampling_mode)
    return walk_tree.as_graph_tensor(
        node_feature_gather_fn or self.gather_node_features_dict,
        static_sizes=static_sizes)

  def gather_node_features_dict(self, node_set_name, node_idx):
    features = self.graph_data.node_features_dicts().get(node_set_name, {})
    features = {feature_name: tf.gather(feature_value, node_idx)
                for feature_name, feature_value in features.items()}
    return features

  def create_context(
      self, sampled_node_ids: Mapping[str, tf.Tensor], seed_nodes: tf.Tensor):
    """Create `tfgnn.Context` for `GraphTensor` seeded at `seed_nodes`.

    Args:
      sampled_node_ids: From node-set name to (tf.Tensor) int vector containing
        sorted node indices, that are sampled under each node set.
      seed_nodes: Seed nodes that seeded the sampler.
    """
    del sampled_node_ids, seed_nodes
    return None


class NodeClassificationGraphSampler(GraphSampler):
  """Samples subgraphs for node-classification in-memory graph data."""

  def __init__(self,
               graph_data: datasets.NodeClassificationGraphData,
               **sampler_kwargs):
    super().__init__(graph_data, **sampler_kwargs)
    self.graph_data = graph_data

  def gather_node_features_dict(self, node_set_name, node_idx):
    features = super().gather_node_features_dict(node_set_name, node_idx)
    if node_set_name == self.graph_data.labeled_nodeset:
      features['label'] = tf.gather(self.graph_data.labels(), node_idx)

    return features

  def create_context(
      self, sampled_node_ids: Mapping[str, tf.Tensor], seed_nodes: tf.Tensor):
    """Create `tfgnn.Context` for `GraphTensor` seeded at `seed_nodes`.

    Args:
      sampled_node_ids: From node-set name to (tf.Tensor) int vector containing
        sorted node indices, that are sampled under each node set.
      seed_nodes: Seed nodes that seeded the sampler.

    Returns:
      `tfgnn.Context` with feature "seed_nodes.<labeledNodeSetName>" and value
      of tf.Tensor containing int vector with positions of seed nodes, within
      `sampled_node_ids["<labeledNodeSetName>"]`.
    """
    newshape = [-1]
    seed_node_positions = tf.expand_dims(
        tf.searchsorted(sampled_node_ids[self.graph_data.labeled_nodeset],
                        tf.reshape(seed_nodes, newshape)),
        0)
    return tfgnn.Context.from_fields(features={
        'seed_nodes.' + self.graph_data.labeled_nodeset: seed_node_positions
    })

  def _get_seed_nodes(self) -> tf.Tensor:
    partitions = self.graph_data.node_split()
    splits = self.graph_data.splits
    return tf.concat([getattr(partitions, s) for s in splits], 0)

  def as_dataset(
      self,
      sampling_spec: sampling_spec_pb2.SamplingSpec,
      pop_labels_from_graph: bool = True,
      num_seed_nodes: int = 1,
      sampling_mode=EdgeSampling.WITH_REPLACEMENT,
      repeat: Union[bool, int] = True, shuffle=True,
      static_sizes: bool = False,
      ) -> tf.data.Dataset:
    """Returns dataset with elements (`GraphTensor`, labels), seeded at `split`.

    If `pop_labels_from_graph == True` (default), then dataset yields:
      (`GraphTensor`, labels), and no labels will be present in GraphTensor.

    If `pop_labels_from_graph == False`, then dataset yields:
      `GraphTensor` with labels being part of it. Passing
      `pop_labels_from_graph=False` then `.map(pop_labels_from_graph)`, is
      equivalent to calling with `pop_labels_from_graph=True`.

    File examples/keras_minibatch_trainer.py shows a usage example

    Args:
      sampling_spec: SamplingSpec proto to indicate number of hops and number of
        samples per hop.
      pop_labels_from_graph: If set (default), records in the datasets are a
        tuple `(GraphTensor, tf.Tensor)` where first contains *no* label feature
        on nodes, and the second is the label matrix of seed nodes. If unset,
        then records are `GraphTensor` with NodeSet `graph_data.labeled_nodeset`
        having additional feature named "labels" (see `graph_data.labels()`).
      num_seed_nodes: int to instruct the seed root nodes per example.
      sampling_mode: to indicate sampling with VS without replacement.
      repeat: If True, then the dataset will be infinitely repeated. If an int,
        then dataset will be repeated this many times. If False, dataset will
        not be repeated.
      shuffle: If set, the nodes will be shuffled.
      static_sizes: Forwarded to sample_sub_graph.
    """
    graph_data = self.graph_data.with_labels_as_features(True)
    seed_nodes = self._get_seed_nodes()
    total_nodes = seed_nodes.shape[0]
    if total_nodes is None:
      total_nodes = tf.shape(seed_nodes)[0]
    dataset = tf.data.Dataset.range(total_nodes)
    dataset = dataset.map(lambda indices: tf.gather(seed_nodes, indices))

    if repeat:
      if isinstance(repeat, bool) and repeat:
        dataset = dataset.repeat()
      else:
        dataset = dataset.repeat(repeat)
    if shuffle:
      num_nodes = graph_data.node_counts()[graph_data.labeled_nodeset]
      dataset = dataset.shuffle(num_nodes)

    dataset = dataset.batch(num_seed_nodes)

    dataset = dataset.map(functools.partial(
        self.sample_sub_graph, sampling_mode=sampling_mode,
        sampling_spec=sampling_spec, static_sizes=static_sizes))

    if pop_labels_from_graph:
      num_classes = graph_data.num_classes()
      dataset = dataset.map(
          functools.partial(reader_utils.pop_labels_from_graph, num_classes))

    return dataset


# Can be replaced with: `_t = tf.convert_to_tensor`.
def as_tensor(obj: Any) -> tf.Tensor:
  """short-hand for tf.convert_to_tensor."""
  return tf.convert_to_tensor(obj)
