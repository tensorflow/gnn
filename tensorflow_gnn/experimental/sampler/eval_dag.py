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
"""Converts Keras sampling models into EvalDag."""

import collections
import contextlib
import dataclasses
import functools

from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import networkx as nx
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.experimental.sampler import core
from tensorflow_gnn.experimental.sampler import eval_dag_pb2
from tensorflow_gnn.experimental.sampler import interfaces

try:
  from keras.engine import input_layer  # pylint:disable=g-import-not-at-top # pytype: disable=import-error
except ImportError:
  from keras.src.engine import input_layer  # pylint:disable=g-import-not-at-top # pytype: disable=import-error


@dataclasses.dataclass
class Artifacts:
  """Collection of sampling `Program` artifacts."""

  # Keras models for `TFModel` layers keyed by layers' ids.
  models: Dict[str, tf.keras.Model]


def create_program(
    model: tf.keras.Model,
) -> Tuple[eval_dag_pb2.Program, Artifacts]:
  """Converts Keras functional model into `Program` message plus artifacts.

  The `Program` contains directed acyclic graph of computation stages. Each
  stage takes results from upstream stages as its inputs and returns new set of
  values as its outputs. The logic how stage computes its results is controlled
  by the *layer* referenced by its global unique identifier. Some layers may
  contain their own computational DAGs (whose stages refer to layers..). The
  `Program` contains global mapping for all layers keyed by their ids.

  The returned `Artifacts` contain additional data that requires special
  handling for serialization and could not be represented as proto messages.
  E.g. collection of `tf.keras.Model`s for 'TFModel' layer types.

  The input model must take as its input(s) any dense or ragged tensors and
  produce any nest of tensors, ragged tensors or graph tensors as its output.
  All input or output values must have the same leading batch dimension of the
  unknown size (`[None, *example_dims]` shape). Examples in the batch must be
  independent: running model on a batch of examples must be equivalent to
  running model individually on any partition of examples from the same batch.

  The model conversion happens by traversing its underlying Keras Functional
  graph. Subsets of nodes from this graph are converted into EvalDAG stages.

  The conversion rules are:


    1. Node of input, output or sampling primive layer are always converted into
      a single EvalDAG stage.
    2. Nodes that are not 1. could be grouped into a single TFModel stage.
    3. Eval dags of `CompositeLayer` nodes are created from `wrapped_model` and
      stored in the `Layer.eval_dag` message field.
    4. The model output is converted into a flat dictionary of tensors or ragged
      tensors keyed by their names. See `flatten_to_dict` for details.

  Args:
    model: The sampling model to convert.

  Returns:
    Pair of eval dag proto with its artifacts.
  """
  artifacts = Artifacts(models={})
  layers = {}
  eval_dag = _create_eval_dag(model, layers, artifacts, root_dag=True)
  result = eval_dag_pb2.Program()
  result.eval_dag.CopyFrom(eval_dag)
  for layer_id, layer in layers.items():
    result.layers[layer_id].CopyFrom(layer)
  return result, artifacts


def create_stages_dag(nodes_dag: nx.DiGraph) -> nx.DiGraph:
  """Converts DAG of Keras nodes into DAG of stages.

  Rules:
    1. each node belong to some unique stage.
    2. each sampling primitive is a single node in its stage (1:1 mapping).
    3. edges between two distinct stages do not create loops.
    4. two regular nodes belong to the same stage if all their inputs are
      from the same single stage, not necessary the same as they belong to.

  The output stages graph has an edge between two stages if and only if nodes
  in those stages are connected. Because of 3. the output graph is a DAG. Each
  edge has an 'inputs' attribute which is a set of keras tensor references (as
  `KerasTensor.ref()`) between nodes in two stages connected by that edge.

  Args:
    nodes_dag: directed acyclic dag of Keras nodes.

  Returns:
    The DAG of stages.
  """
  stages = _partition_in_stages(nodes_dag)
  result = nx.DiGraph()
  node_to_stage = {}
  for stage in stages:
    result.add_node(stage, index=stage.index)
    node_to_stage.update({node: stage for node in stage.nodes})

  for tgt_stage in stages:
    all_inputs = collections.defaultdict(set)
    for tgt_node in tgt_stage.nodes:
      for src_node, _ in nodes_dag.in_edges(tgt_node):
        src_stage = node_to_stage[src_node]
        if src_stage == tgt_stage:
          # ignore inner-edge between nodes within the same stage.
          continue
        all_inputs[src_stage].update(
            _filter_keras_tensor_refs(src_node, tgt_node)
        )

    for src_stage, inputs in all_inputs.items():
      result.add_edge(src_stage, tgt_stage, inputs=inputs)
  return result


Node = Any  #  Node in the Keras functional Graph.
Edge = Any  #  Edge in the Keras functional Graph (KerasTensor instance).
EdgeRef = Any  # Reference to the Edge (as `KerasTensor.ref()`).


def build_ordered_dag(sink: Node) -> List[Node]:
  """Constructs computation dag with nodes sorted in their execution order.

  The Keras functional model defines computational as a directed DAG of `Node`s
  connected by Keras tensors as its edges. Each node is an act of calling some
  Keras layer. It contains information which layer was called, with what
  arguments (nest of input `KerasTensor`s or python constants) and what was the
  output (nest of `KerasTensor`s).

  Args:
    sink: A special single node in the Keras functional model graph to which all
      model outputs are connected to.

  Returns:
    NetworkX directed dag with nodes from model dag Nodes and directed edges
    that connect any two coonected nodes of the original dag. Nodes have special
    `index` attribute which is the node topological sort order from the model
    inputs to the model outputs.
  """
  # First extract all edges as pairs or source and target nodes.
  edges = set()
  nodes = {sink}
  frontier = nodes.copy()
  while frontier:
    next_frontier = set()
    for n in frontier:
      for t_in in tf.nest.flatten(n.keras_inputs):
        edges.add((t_in.node, n))
        next_frontier.add(t_in.node)
    nodes = set.union(nodes, frontier)
    frontier = set.difference(next_frontier, nodes)

  # Sort nodes in topological order.
  unsorted_graph = nx.DiGraph()
  for node in nodes:
    unsorted_graph.add_node(node)

  for src, tgt in edges:
    unsorted_graph.add_edge(src, tgt)

  nodes = nx.lexicographical_topological_sort(
      unsorted_graph, key=_get_node_sort_key
  )
  result = nx.DiGraph()
  for index, node in enumerate(nodes):
    result.add_node(node, index=index)
    for edge in unsorted_graph.out_edges(node):
      result.add_edge(*edge)

  return result


_FLATTEN_LAYER = tf.keras.layers.Lambda(
    lambda t: tf.nest.flatten(t, expand_composites=True)
)


class Sink(tf.keras.layers.Layer, interfaces.SamplingPrimitive):
  """Auxilary class to connect Keras Model outputs to a single node."""

  def __init__(self, *, io_config: Optional[Mapping[str, int]] = None):
    super().__init__()
    self._io_config = io_config.copy() if io_config else {}

  @property
  def io_config(self) -> Mapping[str, int]:
    return self._io_config

  def call(self, inputs):
    # Layer need to provide some output. We are not going to use it anyways.
    del inputs
    return tf.constant(0)


@dataclasses.dataclass
class _Stage:
  """Helper class to group multiple Nodes into single eval dag stage."""

  nodes: Set[Node]
  index: int
  specialized: bool

  def get_single_node(self) -> Node:
    assert len(self.nodes) == 1
    return next(iter(self.nodes))

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    if isinstance(other, int):
      return self.index == other
    assert isinstance(other, _Stage)
    return self.index == other.index


def _create_eval_dag(
    model: tf.keras.Model,
    layers: Dict[str, eval_dag_pb2.Layer],
    artifacts: Artifacts,
    *,
    root_dag: bool,
) -> eval_dag_pb2.EvalDAG:
  """Helper to build `EvalDAG`s recursively."""
  output = model.output
  if not root_dag:
    io_config = {}
  else:
    # Convert output(s) of the root eval DAG into plain dictionary of ragged or
    # dense tensors keyed feature names.
    if _requires_io_adapter_layer(output):
      # Some output tensors must be created using TF ops.
      adapter = tf.keras.layers.Lambda(flatten_to_dict, name='io_adapter')
      output = adapter(output)
    else:
      # All output tensors could be trivially extracted from the model output.
      output = flatten_to_dict(output)
    io_config = _create_io_config(output)

  sink = (Sink(io_config=io_config)(output)).node

  nodes_dag: nx.DiGraph = build_ordered_dag(sink)
  stages_dag: nx.DiGraph = create_stages_dag(nodes_dag)
  return _convert_stages_dag_to_eval_dag(stages_dag, layers, artifacts)


def _convert_stages_dag_to_eval_dag(
    stages_dag: nx.DiGraph,
    layers: Dict[str, eval_dag_pb2.Layer],
    artifacts: Artifacts,
) -> eval_dag_pb2.EvalDAG:
  """Converts stages DAG to eval dag proto updating artifacts."""

  result = eval_dag_pb2.EvalDAG()
  out_edges_spec = {}
  for stage in ordered_nodes(stages_dag):
    stage_pb = result.stages.add()
    stage_pb.id = f'stage{stage.index}'
    in_edges = _get_stage_in_edges(stages_dag, stage)
    out_edges = _get_stage_out_edges(stages_dag, stage)

    for edge_idx, edge in enumerate(out_edges):
      out_edges_spec[edge.ref()] = (stage_pb.id, edge_idx)

    for edge in in_edges:
      stage_id, output = out_edges_spec[edge.ref()]
      stage_pb.input_matchers.add(stage_id=stage_id, output_index=output)

    layer_pb = eval_dag_pb2.Layer()
    layer_pb.inputs.extend(tf.nest.map_structure(_get_spec_pb, in_edges))
    layer_pb.outputs.extend(tf.nest.map_structure(_get_spec_pb, out_edges))

    if not stage.specialized:
      layer_pb.id = f'model{len(artifacts.models)}'
      layer_pb.type = 'TFModel'
      artifacts.models[layer_pb.id] = create_tf_stage_model(
          in_edges, out_edges
      )
    else:
      layer_pb.id, layer_pb.type = _get_layer_pb_id_and_type(
          stage.get_single_node().layer
      )

      config = get_layer_config_pb(stage.get_single_node().layer)
      if config:
        layer_pb.config.Pack(config)

      if isinstance(stage.get_single_node().layer, core.CompositeLayer):
        for arg_layer in tf.nest.flatten(
            stage.get_single_node().layer.wrapped_model.input
        ):
          layer_pb.input_names.feature_names.append(arg_layer.node.layer.name)
        layer_pb.eval_dag.CopyFrom(
            _create_eval_dag(
                stage.get_single_node().layer.wrapped_model,
                layers,
                artifacts,
                root_dag=False,
            )
        )

    layers[layer_pb.id] = layer_pb
    stage_pb.layer_id = layer_pb.id

  return result


def _has_specialized_stage(node: Node) -> bool:
  """`True` if `node` has a specialized stage."""
  return isinstance(
      node.layer,
      (
          input_layer.InputLayer,
          interfaces.SamplingPrimitive,
          core.CompositeLayer,
      ),
  )


def _create_stage(nodes_dag: nx.DiGraph, seed: Node, stage_idx: int) -> _Stage:
  """Groups nodes into a single stage starting from the seed node.

  The nodes are added to the stage starting from the `seed` node and follow
  directions of `nodes_dag` edges. The `seed` could be regular node or sampling
  primitive. For the latter case, the `seed` is used to start the traversal but
  it is not included in the stage itself.

  A node is added to the stage if and only if it is not a sampling primitive and
  all its predecessor nodes are from the same stage.

  This algorithm allows the Sampler to group a sequence of ordinary
  Keras layers (each Keras layer is a node) into one stage.

  This stage will be saved as a single TF Model so a bulk runner can compute
  stage results in a single TF model call.

  Args:
    nodes_dag: directed acyclic dag of Keras nodes.
    seed: initial node to start expansion from.
    stage_idx: unique integer stage identifier.

  Returns:
    One or multiple nodes grouped into a newly created stage.
  """
  done = {seed}
  frontier = {seed}
  while frontier:
    next_frontier = set()
    for src in frontier:
      for _, tgt in nodes_dag.out_edges(src):
        if tgt in done or _has_specialized_stage(tgt):
          continue

        # Add `tgt` node to the stage if all its predecessor nodes belong
        # to the the same stage.
        if set(nodes_dag.predecessors(tgt)) - done:
          continue

        next_frontier.add(tgt)
        done.add(tgt)
    frontier = next_frontier

  result_nodes = done
  if _has_specialized_stage(seed):
    result_nodes.remove(seed)
  return _Stage(
      nodes=result_nodes,
      index=stage_idx,
      specialized=False,
  )


def _partition_in_stages(nodes_dag: nx.DiGraph) -> List[_Stage]:
  """Partition nodes of input DAG into non-overlapping stages.

  This is a helper function for `create_stages_dag` that partition nodes into
  the list of stages.

  NOTE: currently each `Input` layer has its own stage. This could be
  inefficient if multiple inputs are read from the same file.

  TODO(aferludin): add support for grouping of multiple inputs into the same
    stage.

  Args:
    nodes_dag: directed acyclic dag of Keras nodes.

  Returns:
    List of stages sorted in the same topological order as an input dag.
  """
  result = []
  done = set()
  has_next_stage = True
  stage_idx = 0
  nodes = ordered_nodes(nodes_dag)
  while has_next_stage:
    has_next_stage = False
    for node in nodes:
      if node in done:
        continue

      # Only explore node if all its inputs are part of some stages.
      if not all(e.node in done for e in node.keras_inputs):
        continue

      has_next_stage = True
      if _has_specialized_stage(node):
        # Sampling primitive has 1:1 mapping with their stages.
        stage = _Stage(
            nodes={node},
            index=stage_idx,
            specialized=True,
        )
      else:
        # Check if node has a single predecessor. If so, we better start
        # stage expansion from that node (without including it). This allows to
        # merge all nodes that have the same single source into a single stage.
        predecessors = {e.node for e in node.keras_inputs}
        seed = predecessors.pop() if len(predecessors) == 1 else node
        stage = _create_stage(nodes_dag, seed, stage_idx)

      assert node in stage.nodes
      result.append(stage)
      done.update(stage.nodes)

      stage_idx += 1
      break
  return result


def _filter_keras_tensor_refs(src: Node, tgt: Node) -> Set[EdgeRef]:
  return {t.ref() for t in tf.nest.flatten(tgt.keras_inputs) if t.node == src}


def _get_node_sort_key(n: Node) -> Tuple[str, int]:
  """Returns unique node name within the same graph (as in layer summary)."""
  layer = n.layer
  index = layer.inbound_nodes.index(n)
  return layer.name, index


def _get_stage_out_edges(graph: nx.DiGraph, stage: _Stage) -> List[Edge]:
  """Get stage output as an ordered list of edges."""

  def get_node_outputs(node: Node) -> List[Edge]:
    if isinstance(node.layer, Sink):
      return []
    return tf.nest.flatten(node.outputs)

  return _get_stage_edges(
      stage,
      get_node_outputs,
      lambda stage: graph.out_edges(stage, data=True),
  )


def _get_stage_in_edges(graph: nx.DiGraph, stage: _Stage) -> List[Edge]:
  """Get stage input as an ordered list of edges."""
  return _get_stage_edges(
      stage,
      lambda node: tf.nest.flatten(node.keras_inputs),
      lambda stage: graph.in_edges(stage, data=True),
  )


def _get_stage_edges(
    stage: _Stage,
    node_edges_fn: Callable[[Node], Any],
    stage_edges_fn: Callable[[_Stage], Any],
) -> List[Edge]:
  """Helper to get select edges from a stage."""

  if stage.specialized:
    return node_edges_fn(stage.get_single_node())

  edges_in_use = set()
  for _, _, data in stage_edges_fn(stage):
    edges_in_use.update(data['inputs'])

  result = set()
  sorted_nodes = sorted(stage.nodes, key=_get_node_sort_key)
  for node in sorted_nodes:
    for edge in node_edges_fn(node):
      if edge.ref() not in edges_in_use:
        continue
      result.add(edge.ref())

  return [e.deref() for e in result]


def ordered_nodes(nodes_dag: nx.DiGraph) -> List[Node]:
  node_index_pairs = sorted(nodes_dag.nodes(data='index'), key=lambda ki: ki[1])
  return [node for node, _ in node_index_pairs]


_SERVING_ATTR = 'gnn_serving'


def create_tf_stage_model(
    inputs: List[Edge], outputs: List[Edge]
) -> tf.keras.Model:
  """Creates keras model from input and output tensors with serving signature.

  Args:
    inputs: list of model input tensors.
    outputs: list of model output tensors.

  Returns:
    Keras Model object with `gnn_serving` callable attribute that takes list of
    flattened input tensors  and returns list of flattened output tensors.
    Flattening is equivalent to `tf.nest.flatten(..., expand_composites=True)`.
  """

  def get_spec(nested_struct):
    def fn(t):
      result = getattr(t, 'type_spec', None)
      if result is None:
        raise ValueError(f'Expected keras tensor values, got {t}')

      return result

    return tf.nest.map_structure(fn, nested_struct)

  with _input_layer_fix():
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

  # NOTE: we must call flatten on real tensors, so within Keras call.
  model_args_struct = get_spec(inputs)
  inputs_spec = get_spec(tf.keras.layers.Lambda(_to_executor_values)(inputs))
  flattened_inputs_spec = tf.nest.flatten(inputs_spec)

  @tf.function(input_signature=([flattened_inputs_spec]))
  def serving(argw):
    inputs = tf.nest.pack_sequence_as(inputs_spec, argw)
    inputs = _from_executor_values(model_args_struct, inputs)
    outputs = model(inputs)
    outputs = _to_executor_values(outputs)
    return tf.nest.flatten(outputs)

  setattr(model, _SERVING_ATTR, serving)
  return model


def save_model(
    model: tf.keras.Model,
    filepath: str,
    signature_name: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
) -> None:
  # TODO(b/277116619): save with traces when/if fixed.
  model.save(
      filepath,
      signatures={
          signature_name: getattr(model, _SERVING_ATTR),
      },
      save_format='tf',
      save_traces=False,
  )


@contextlib.contextmanager
def _input_layer_fix():
  """Fix for unsupported composite `tensor` arguments in `tf.keras.Input`.

  The current keras `Input` layer implementaion does not support composite
  `tensor` arguments. This results in an exception on attempt to create
  `tf.keras.Model` using composite tensors as its inputs.

  Yields:
    Context with the fix.
  """
  # pylint: disable=g-doc-return-or-yield
  original_input = getattr(input_layer, 'Input')

  def _fixed_input(
      shape=None,
      batch_size=None,
      name=None,
      dtype=None,
      sparse=None,
      tensor=None,
      ragged=None,
      type_spec=None,
      **kwargs,
  ):
    """Fix when `tensor` argument is composite tensor."""
    if tensor is not None and hasattr(tensor, 'type_spec'):
      return original_input(type_spec=tensor.type_spec, **kwargs)
    return original_input(
        shape=shape,
        batch_size=batch_size,
        name=name,
        dtype=dtype,
        sparse=sparse,
        ragged=ragged,
        type_spec=type_spec,
        **kwargs,
    )

  setattr(input_layer, 'Input', _fixed_input)
  try:
    yield
  finally:
    setattr(input_layer, 'Input', original_input)


def flatten_to_dict(nested_values: Any) -> tfgnn.Fields:
  """Converts nest of features to the flat dictionary of its components.

  The input `nested_values` could be any nested structure of tuples, lists or
  dictionaries with `Tensor`, `RaggedTensor` or `GraphTensor` values. The name
  of the output feature is formed as a '/' separated path to the feature,
  starting from the outermost structure, through the inner structures to the
  value.

  The path elements are:
    * stringized keys of any mapping collections;
    * stringized indices of tuples or lists;
    * feature names in the `GraphTensor` IO format.

  If the `nested_values` consists of a single `tf.Tensor` or `tf.RaggedTensor`,
  '__output__' is used as an output key.


  Example 1: nested dictionaries.

  ```python
  dt = tf.convert_to_tensor
  flatten_to_dict({'x': {'a': dt([1])}})
  # {'x/a': [1]}
  ```

  Example 2: tuple of dictionaries.

  ```python
  dt = tf.convert_to_tensor
  flatten_to_dict([{'a': dt([1])}, {'a': dt([1]), 'b': dt([2])}])
  # {'0/a': [1], '1/a': [1], '1/b': [2]}
  ```

  Example 3: single value.

  ```python
  flatten_to_dict(tf.ragged.constant([[1], [2, 3]]))
  # {'__output__': [[1], [2, 3]]}


  Args:
    nested_values: values.

  Returns:
    a `tfgnn.Fields`, which is a dictionary of `tf.Tensor` or `tf.RaggedTensor`
    values keyed by feature names.
  """
  return _flatten_to_dict('', nested_values)


def _flatten_to_dict(prefix, nested_struct):
  """Recursive implementation of flatten_to_dict."""
  if tfgnn.is_ragged_tensor(nested_struct) or tfgnn.is_dense_tensor(
      nested_struct
  ):
    prefix = prefix or '__output__'
    return {prefix: nested_struct}

  if tfgnn.is_graph_tensor(nested_struct):
    return _flatten_to_dict(
        prefix, _flatten_graph_tensor_to_dict(nested_struct)
    )

  if isinstance(nested_struct, Mapping):
    return _flatten_kv_to_dict(prefix, nested_struct.items())
  if isinstance(nested_struct, (List, Tuple)):
    return _flatten_kv_to_dict(prefix, enumerate(nested_struct))

  raise ValueError(
      'Model outputs must be nested structure of dict, list or tuple with'
      ' `tf.Tensor` `tf.RaggedTensor` or `tfgnn.GraphTensor` as its'
      ' values'
      f', got {nested_struct}'
  )


def _flatten_kv_to_dict(prefix: str, kv_iter: Any):
  prefix = f'{prefix}/' if prefix else ''
  result = {}
  for k, v in kv_iter:
    result.update(_flatten_to_dict(f'{prefix}{k}', v))
  return result


def _flatten_graph_tensor_to_dict(graph: tfgnn.GraphTensor) -> tfgnn.Fields:
  """Converts input graph tensor to the flat dictionary of features."""

  result = {}

  def _add_features(prefix: str, values: tfgnn.Fields) -> None:
    for fname, value in values.items():
      result[f'{prefix}{fname}'] = value

  _add_features(f'{tfgnn.CONTEXT}/', graph.context.features)
  for set_name, node_set in graph.node_sets.items():
    _add_features(f'{tfgnn.NODES}/{set_name}.', node_set.features)
    result[f'{tfgnn.NODES}/{set_name}.{tfgnn.SIZE_NAME}'] = node_set.sizes

  for set_name, edge_set in graph.edge_sets.items():
    adjacency = edge_set.adjacency
    if not isinstance(adjacency.spec, tfgnn.AdjacencySpec):
      raise ValueError(
          f'Expected `tfgnn.Adjacency` adjacency for {set_name} edge set.'
      )
    _add_features(f'{tfgnn.EDGES}/{set_name}.', edge_set.features)
    result[f'{tfgnn.EDGES}/{set_name}.{tfgnn.SIZE_NAME}'] = edge_set.sizes
    result[f'{tfgnn.EDGES}/{set_name}.{tfgnn.SOURCE_NAME}'] = adjacency.source
    result[f'{tfgnn.EDGES}/{set_name}.{tfgnn.TARGET_NAME}'] = adjacency.target
  return result


@functools.singledispatch
def get_layer_config_pb(layer: tf.keras.layers.Layer) -> Any:
  raise NotImplementedError(
      f'Config dispatching is not defined for {type(layer).__name__}'
  )


@get_layer_config_pb.register(core.CompositeLayer)
def _(layer: core.CompositeLayer):
  # TODO(aferludin): add configs for composite layers to allow specialization.
  return None


@get_layer_config_pb.register(interfaces.KeyToBytesAccessor)
def _(layer: interfaces.KeyToBytesAccessor):
  return None


@get_layer_config_pb.register(input_layer.InputLayer)
def _(layer: input_layer.InputLayer):
  return None


@get_layer_config_pb.register(core.UniformEdgesSampler)
def _(layer: core.UniformEdgesSampler):
  return eval_dag_pb2.EdgeSamplingConfig(
      edge_set_name=layer.resource_name, sample_size=layer.sample_size
  )


@get_layer_config_pb.register(Sink)
def _(layer: Sink):
  if not layer.io_config:
    return None
  result = eval_dag_pb2.IOFeatures()
  io_config = sorted(layer.io_config.items(), key=lambda kv: kv[1])
  for name, _ in io_config:
    result.feature_names.append(name)
  return result


def _get_spec_pb(edge: Edge) -> eval_dag_pb2.ValueSpec:
  """Maps edge on the value spec proto."""
  result = eval_dag_pb2.ValueSpec()
  spec = edge.type_spec
  if isinstance(spec, tf.TensorSpec):
    result.tensor.CopyFrom(
        eval_dag_pb2.TensorSpec(
            dtype=spec.dtype.as_datatype_enum, shape=spec.shape.as_proto()
        )
    )
  elif isinstance(spec, tf.RaggedTensorSpec):
    result.ragged_tensor.CopyFrom(
        eval_dag_pb2.RaggedTensorSpec(
            dtype=spec.dtype.as_datatype_enum,
            shape=spec.shape.as_proto(),
            ragged_rank=spec.ragged_rank,
            row_splits_dtype=spec.row_splits_dtype.as_datatype_enum,
        )
    )
  else:
    result.flattened.components.extend(
        [_get_spec_pb(piece).tensor for piece in _FLATTEN_LAYER(edge)]
    )

  return result


def _requires_io_adapter_layer(nested_struct) -> bool:
  """Returns `True` if nested structure requires IO Adapter layer."""
  return not all(
      tfgnn.is_ragged_tensor(t) or tfgnn.is_dense_tensor(t)
      for t in tf.nest.flatten(nested_struct)
  )


def _create_io_config(feature: tfgnn.Fields) -> Mapping[str, int]:
  """Map feature names on thier indices in the flattened array."""
  assert isinstance(feature, Mapping)
  indices = [i for i, _ in enumerate(tf.nest.flatten(feature))]
  return tf.nest.pack_sequence_as(feature, indices)


def _to_executor_values(nested_struct) -> List[List[tf.Tensor]]:
  """Converts any nest of tensors to bulk executor format.

  Args:
    nested_struct: any nest of tensors or composite tensors.

  Returns:
    Structure first flattened to its components (tensors or composite tensors)
    and then each component being converted to its tensor pieces according to
    the `_flatten` rules.
  """
  return [_flatten(t) for t in tf.nest.flatten(nested_struct)]


def _from_executor_values(nested_struct, values: List[List[tf.Tensor]]) -> Any:
  """The inverse operation to `_to_executor_values`."""

  specs = tf.nest.flatten(nested_struct)
  pieces = [_unflatten(s, c) for s, c in zip(specs, values)]
  return tf.nest.pack_sequence_as(nested_struct, pieces)


def _flatten(value) -> List[tf.Tensor]:
  """Converts tensor or composite tensor value to the bulk executor format.

  NOTE: this specialization is needed because the default `tf.nest.flatten`
  outputs components which sometimes hard to manipulate. In particular, the
  flattening of `tf.RaggedTensor` uses row splits, and those don't stack well.

  Rules of conversion:
    * tf.Tensor: [value]
    * tf.RaggedTensor: [value.flat_values, *value.nested_row_lengths()]
    * other: tf.nest.flatten(value, expand_composites=True).

  Args:
    value: any tensor or composite tensor.

  Returns:
    List of tensors.
  """
  if isinstance(value, tf.Tensor):
    return [value]

  if isinstance(value, tf.RaggedTensor):
    return [value.flat_values, *value.nested_row_lengths()]

  return tf.nest.flatten(value, expand_composites=True)


def _unflatten(spec, components: List[tf.Tensor]) -> Any:
  """The inverse operation to `_flatten`."""
  if isinstance(spec, tf.RaggedTensorSpec):
    values, nested_row_lengths = components[0], components[1:]
    return tf.RaggedTensor.from_nested_row_lengths(
        values, nested_row_lengths, validate=False
    )

  return tf.nest.pack_sequence_as(spec, components, expand_composites=True)


def _get_layer_pb_id_and_type(layer: tf.keras.layers.Layer) -> Tuple[str, str]:
  if isinstance(layer, interfaces.KeyToBytesAccessor):
    return layer.name, 'KeyToBytesAccessor'

  return layer.name, type(layer).__name__
