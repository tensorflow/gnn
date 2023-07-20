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
"""Provides builder pattern that eases creation of `tfgnn.SamplingSpec`.

Output `SamplingSpec` will contain topologically-sorted `SamplingOp`s.



Example: Homogeneous Graphs.

If your graph is *homogeneous* and your node set is named "nodes" and edge set
is named "edges", then you can create the sampling spec proto as:


```python
# Assume homogeneous schema with edge-set name "edges" connecting "nodes".
schema = schema_pb2.GraphSchema()
schema.edge_sets['edges'].source = s.edge_sets['edges'].target = 'nodes'

proto = (SamplingSpecBuilder(schema).seed('nodes').sample(10, 'edges')
         .sample(5, 'edges').build())

# Since graph homogeneous (i.e., only one edge type and node type), then you may
# skip the edge-set and node-set names, and call as:
proto = SamplingSpecBuilder(schema).seed().sample(10).sample(5).build()

# Since above example samples from same edge type, you can pass a list as the
# second argument.
proto = (SamplingSpecBuilder(schema)
         .seed('nodes').sample('edges', [10, 5]).build())

# Or alternatively,
proto = tfgnn.SamplingSpecBuilder(schema).seed().sample([10, 5]).build()
```

The above spec is instructing to start at:
  - Nodes of type set name "nodes", then,
  - for each seed node, sample (up to) 10 of its neighbors (from edge set
    "edges").
  - for each of those neighbors, sample (up to) 5 neighbors (from same edgeset).


Example: Heterogeneous Graphs.

E.g., if you consider citation datasets, you can make a SamplingSpec proto as:

```python
proto = (SamplingSpecBuilder(schema)
          .seed('author').sample('writes', 10).sample('cited_by', 5)
          .build())
```

This samples, starting from author node, 10 papers written by author, and for
each paper, 10 papers citing it.


Example: DAG Sampling.

Finally, your sampling might consist of a DAG. For this, you need to cache
some returns of `.sample()` calls. For example:

```python
# Store builder at level of "author written papers":
builder = tfgnn.SamplingSpecBuilder(schema).seed('author').sample('writes', 10)
path1 = builder.sample('cited_by', 5)
path2 = builder.sample('written_by', 3).sample('writes')

proto = (tfgnn.SamplingSpecBuilder.Join([path1, path2]).sample('cited_by', 10)
         .build())

# The above `Join()` can be made less verbose with:
proto = path1.Join([path2]).sample('cited_by', 10).build()
```

This merges together the papers written by author, and written by co-authors,
and for each of those papers, sample 10 papers citing it.
"""
import collections
from typing import Optional, Iterable, Union, List, Any, Mapping

from tensorflow_gnn.graph import graph_constants
from tensorflow_gnn.graph import graph_tensor
from tensorflow_gnn.graph import graph_tensor_ops
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from tensorflow_gnn.sampler import sampling_spec_pb2


NodeSetName = graph_constants.NodeSetName
EdgeSetName = graph_constants.EdgeSetName
GraphTensor = graph_tensor.GraphTensor

SamplingStrategy = sampling_spec_pb2.SamplingStrategy


def _topological_sort(all_steps):
  """Uses `children` & `parents` to topologically-sort `all_steps`."""
  sorted_steps = []
  all_steps_set = set(all_steps)
  for step in all_steps:  # Verify no dangling edges to outside all_steps.
    children_set = set(step.children)
    parent_set = set(step.parents)
    assert all_steps_set.intersection(children_set) == children_set
    assert all_steps_set.intersection(parent_set) == parent_set
  #
  # Top sort algorithm.
  ready_nodes = [step for step in all_steps if not step.parents]
  unprocessed_parents = {step: set(step.parents)
                         for step in all_steps if step.parents}
  while ready_nodes:
    next_ready_node = ready_nodes.pop()
    sorted_steps.append(next_ready_node)
    for child in next_ready_node.children:
      unprocessed_parents[child].remove(next_ready_node)
      if not unprocessed_parents[child]:  # No more unprocessed parents.
        ready_nodes.append(child)
        unprocessed_parents.pop(child)
  #
  if unprocessed_parents:
    raise ValueError('Topological sort does not terminate. Likely, the '
                     'sampling steps do not form a DAG.')
  return sorted_steps


def _op_name_from_parents(parents):
  if len(parents) == 1:
    return parents[0].node_set_name

  return '(%s)' % '|'.join([s.op_name for s in parents])


def make_sampling_spec_tree(
    graph_schema: schema_pb2.GraphSchema,
    seed_nodeset: NodeSetName,
    *,
    sample_sizes: List[int],
    sampling_strategy=SamplingStrategy.RANDOM_UNIFORM
) -> sampling_spec_pb2.SamplingSpec:
  """Automatically creates `SamplingSpec` by starting from seed node set.

  From seed node set, `sample_sizes[0]` are sampled from *every* edge set `E`
  that originates from seed node set. Subsequently, from sampled edge `e` in `E`
  the created `SamplingSpec` instructs sampling up to `sample_sizes[1]` edges
  for `e`'s target node, and so on, until depth of `len(sample_sizes)`.

  Args:
    graph_schema: contains node-sets & edge-sets.
    seed_nodeset: name of node-set that the sampler will be instructed to use as
      seed nodes.
    sample_sizes: list of number of nodes to sample. E.g. if `sample_sizes` are
      `[5, 2, 2]`, then for every sampled node, up-to `5` of its neighbors will
      be sampled, and for each, up to `2` of its neighbors will be sampled, etc,
      totalling sampled nodes up to `5 * 2 * 2 = 20` for each seed node.
    sampling_strategy: one of the supported sampling strategies, the same for
      each depth.

  Returns:
    `SamplingSpec` that instructs the sampler to sample according to the
    `sampling_strategy` and `sample_sizes`.
  """
  edge_sets_by_src_node_set = _edge_set_names_by_source(graph_schema)
  spec_builder = SamplingSpecBuilder(
      graph_schema,
      default_strategy=sampling_strategy)
  spec_builder = spec_builder.seed(seed_nodeset)

  def _recursively_sample_all_edge_sets(
      cur_node_set_name, sampling_step, remaining_sample_sizes):
    if not remaining_sample_sizes:
      return
    for edge_set_name in sorted(edge_sets_by_src_node_set[cur_node_set_name]):
      if graph_tensor.get_aux_type_prefix(edge_set_name):
        continue  # Skip private edges (e.g., _readout).

      edge_set_schema = graph_schema.edge_sets[edge_set_name]
      _recursively_sample_all_edge_sets(
          edge_set_schema.target,
          sampling_step.sample(remaining_sample_sizes[0], edge_set_name),
          remaining_sample_sizes[1:])

  _recursively_sample_all_edge_sets(seed_nodeset, spec_builder, sample_sizes)

  return spec_builder.build()


class SamplingSpecBuilder(object):
  """Mimics builder pattern that eases creation of `tfgnn.SamplingSpec`.


  Example: Homogeneous Graphs.

  If your graph is *homogeneous* and your node set is named "nodes" and edge set
  is named "edges", then you can create the sampling spec proto as:

  NOTE: This should come from the outside, e.g., `graph_tensor.schema`.
  
  ```python
  schema = schema_pb2.GraphSchema()
  schema.edge_sets['edges'].source = s.edge_sets['edges'].target = 'nodes'

  proto = (SamplingSpecBuilder(schema)
           .seed('nodes').sample('edges', 10).sample('edges', 5)
           .build())
  ```

  The above spec is instructing to start at:
    - Nodes of type set name "nodes", then,
    - for each seed node, sample 10 of its neighbors (from edge set "edges").
    - for each of those neighbors, sample 5 neighbors (from same edge set).

  Example: Heterogeneous Graphs.

  E.g., if you consider citation datasets, you can make a SamplingSpec proto as:

  ```python
  proto = (SamplingSpecBuilder(schema)
           .seed('author').sample('writes', 10).sample('cited_by', 5)
           .build())
  ```

  This samples, starting from author node, 10 papers written by author, and for
  each paper, 10 papers citing it.


  Example: DAG Sampling.

  Finally, your sampling might consist of a DAG. For this, you need to cache
  some returns of `.sample()` calls.
  """

  def __init__(
      self, graph_schema: schema_pb2.GraphSchema,
      default_strategy: SamplingStrategy = SamplingStrategy.TOP_K):
    self.graph_schema = graph_schema
    self.seeds = []
    self.default_strategy = default_strategy

  @staticmethod
  def join(steps):
    return Join(steps)

  def seed(self, node_set_name: Optional[str] = None):
    """Initializes sampling by seeding on node with `node_set_name`.

    Args:
      node_set_name: Becomes the `node_set_name` of built `spec.sampling_op`. If
        not given, the graph schema must be homogeneous (with one `node_set`).
        If given, it must correspond to some node set name in `graph_schema`
        given to constructor.

    Returns:
      Object which support builder pattern, upon which, you may repeatedly call
      `.sample()`, per header comments.
    """
    if node_set_name is None:
      all_node_set_names = list(self.graph_schema.node_sets.keys())
      if len(all_node_set_names) != 1:
        raise ValueError(
            'node_set_name is not set. Expecting graph with exactly one node '
            'set. However, found node set names: ' +
            ', '.join(all_node_set_names))
      node_set_name = all_node_set_names[0]
    if node_set_name not in self.graph_schema.node_sets:
      raise ValueError(
          'GraphSchema does not have node_set_name: ' + node_set_name)
    step = _SamplingStep(node_set_name, None, parent=None, builder=self)
    self.seeds.append(step)
    return step

  def to_sampling_spec(self) -> sampling_spec_pb2.SamplingSpec:
    """DEPRECATED: use `build` instead."""
    return self.build()

  def build(self) -> sampling_spec_pb2.SamplingSpec:
    """Creates new SamplingSpec that is built at this moment."""
    spec = sampling_spec_pb2.SamplingSpec()

    def add_step_and_children(step, all_steps):
      all_steps.append(step)
      for child in step.children:
        add_step_and_children(child, all_steps)

    all_steps = []  # To be recursively expanded.
    for step in self.seeds:
      add_step_and_children(step, all_steps)

    all_steps = _topological_sort(all_steps)
    utilized_names = collections.Counter()
    for step in all_steps:
      if step.parents:  # sampling_op.
        if not step.op_name:
          step.op_name = '%s->%s' % (
              _op_name_from_parents(step.parents),
              step.node_set_name)
          utilized_names[step.op_name] += 1
          if utilized_names[step.op_name] > 1:
            step.op_name += '.%i' % utilized_names[step.op_name]
        sampling_op = spec.sampling_ops.add(
            op_name=step.op_name, edge_set_name=step.edge_set_name,
            strategy=step.strategy, sample_size=step.sample_size)
        sampling_op.input_op_names.extend([s.op_name for s in step.parents])
      else:  # seed op.
        if spec.seed_op.node_set_name:
          raise ValueError('SamplingSpec proto can contain only one seed.')
        if not step.op_name:
          step.op_name = 'SEED->' + step.node_set_name
          utilized_names[step.op_name] += 1
          if utilized_names[step.op_name] > 1:
            step.op_name += '.%i' % utilized_names[step.op_name]
        spec.seed_op.op_name = step.op_name
        spec.seed_op.node_set_name = step.node_set_name

    return spec


class _SamplingStep(object):
  """Node on the sampling graph, owned by SamplingSpecBuilder."""

  def __init__(self, node_set_name: str, edge_set_name: Optional[str],
               parent: Optional['_SamplingStep'],
               *,
               sample_size: Optional[int] = None,
               strategy: Optional['sampling_spec_pb2.SamplingStrategy'] = None,
               op_name: Optional[str] = None,
               builder: SamplingSpecBuilder):
    """Constructs _SamplingStep.

    Args:
      node_set_name: Node set name that this sampling step reaches.
      edge_set_name: Edge set name which lead to this `node_set_name`.
      parent: Should be the parent of this sampling step, or None if this is the
        seed step (corresponding to seed).
      sample_size: Number of edges of name `edge_set_name` that will be sampled.
      strategy: Sampling strategy. Member of enum `SamplingStrategy`. If your
        edges are weighted, you probably want SamplingStrategy.TOP_K.
      op_name: Op name for this sampling. If not given, it will be auto-assigned
        "[source node set name]->[target node set name]".
      builder: `SamplingSpecBuilder` that created the seed of this step.
    """
    self.builder = builder
    self.node_set_name = node_set_name
    self.edge_set_name = edge_set_name
    self.sample_size = sample_size
    self.children = []
    self.parents = []
    self.op_name = op_name
    self.strategy = strategy if strategy else builder.default_strategy
    if parent: self.parents.append(parent)

  def build(self) -> sampling_spec_pb2.SamplingSpec:
    return self.builder.build()

  def to_sampling_spec(self) -> sampling_spec_pb2.SamplingSpec:
    """DEPRECATED: use `build` instead."""
    return self.build()

  def sample(self, sample_size: Union[int, List[int]],
             edge_set_name: Optional[str] = None,
             strategy: Optional[int] = None,
             op_name: Optional[str] = None) -> '_SamplingStep':
    """Instructs builder to choose `sample_size` from `edge_set_name`.

    This must follow from previous sampling step. Specifically, `edge_set_name`
    must connect node set (self.node_set_name).

    Returns:
      _SamplingStep instance `step` with `step.parents = self` and
      step.node_set_name determined by following edge `edge_set_name`.

    Args:
      sample_size: Can be an int (for one step) or list of ints (for multiple
        steps). The int is the number of edges of `edge_set_name` to be sampled.
      edge_set_name: Must be edge set name available in init `GraphSchema`. In
        addition, the edge set must connect node set `self.node_set_name`. If
        graph has only one edge set name, this can be left unset.
      strategy: Sampling strategy. Member of enum `SamplingStrategy`. If your
        edges are weighted, you probably want SamplingStrategy.TOP_K.
      op_name: Optional name of sampling op. It must be unique. If not given, it
        will be deduced.
    """
    if edge_set_name is None:
      all_edge_set_names = list(self.builder.graph_schema.edge_sets.keys())
      if len(all_edge_set_names) != 1:
        raise ValueError(
            'edge_set_name is not set. Expecting graph with exactly one edge '
            'set. However, found edge set names: ' +
            ', '.join(all_edge_set_names))
      edge_set_name = all_edge_set_names[0]
    if edge_set_name not in self.builder.graph_schema.edge_sets:
      raise ValueError(
          'GraphSchema does not have edge_set_name: ' + edge_set_name)
    edge_set = self.builder.graph_schema.edge_sets[edge_set_name]
    if edge_set.source != self.node_set_name:
      raise ValueError(
          'Sampling from node_set_name %s edge_set with name %s but the edge '
          'set has source node_set_name %s.' % (
              self.node_set_name, edge_set_name, edge_set.source))
    to_node_set_name = edge_set.target

    if isinstance(sample_size, List):
      if len(sample_size) > 1:
        next_step = _SamplingStep(
            to_node_set_name, edge_set_name, parent=self,
            sample_size=sample_size[0], strategy=strategy,
            op_name=op_name, builder=self.builder)
        self.children.append(next_step)
        return next_step.sample(
            sample_size[1:], edge_set_name, strategy=strategy, op_name=op_name)
      elif len(sample_size) == 1:
        sample_size = sample_size[0]
      else:
        raise ValueError('Empty list of sizes.')

    step = _SamplingStep(
        to_node_set_name, edge_set_name, parent=self, sample_size=sample_size,
        strategy=strategy, op_name=op_name, builder=self.builder)

    self.children.append(step)
    return step

  def merge_then_sample(
      self, other_steps: Iterable['_SamplingStep'],
      *sample_arg, **sample_kwargs) -> '_SamplingStep':
    """Like `sample` but accepts another branch of the builder to build DAG."""
    for step in other_steps:
      if step.node_set_name != self.node_set_name:
        raise ValueError('merge_then_sample expects sample steps pointing to '
                         'node set with same name. Found %s and %s.' % (
                             self.node_set_name, step.node_set_name))

    new_step = self.sample(*sample_arg, **sample_kwargs)
    for other_step in other_steps:
      other_step.children.append(new_step)
      new_step.parents.append(other_step)

    return new_step

  def join(self, other_steps):
    return Join([self] + list(other_steps))


class Join:

  def __init__(self, steps: List[_SamplingStep]):
    if not steps:
      raise ValueError('Expecting at least one step.')
    self._steps = steps

  def sample(self, *sample_args, **sample_kwargs) -> '_SamplingStep':
    return self._steps[0].merge_then_sample(
        self._steps[1:], *sample_args, **sample_kwargs)


def _edge_set_names_by_source(
    graph: Union[schema_pb2.GraphSchema, GraphTensor, Any]
    ) -> Mapping[NodeSetName, List[EdgeSetName]]:
  """Returns map: node set name -> list of edge names outgoing from node set."""
  results = collections.defaultdict(list)
  if isinstance(graph, schema_pb2.GraphSchema):
    for edge_set_name, edge_set_schema in graph.edge_sets.items():
      results[edge_set_schema.source].append(edge_set_name)
  elif graph_tensor_ops.is_graph_tensor(graph):
    for edge_set_name, edge_set_tensor in graph.edge_sets.items():
      results[edge_set_tensor.adjacency.source_name].append(edge_set_name)
  else:
    raise TypeError('Not yet supported type %s' % str(graph.__class__))
  return results
