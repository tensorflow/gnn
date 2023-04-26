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
from typing import Dict, List, Set

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.experimental.sampler import core
from tensorflow_gnn.experimental.sampler import eval_dag as lib
from tensorflow_gnn.experimental.sampler import eval_dag_pb2 as pb
from tensorflow_gnn.experimental.sampler import interfaces


rt = tf.ragged.constant  # ragged tensor
dt = tf.convert_to_tensor  # dense tensor

context_graph = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        sizes=dt([1, 1]), features={'s': dt([[1], [2]])}
    ),
)
aa_graph = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'s': dt([['1', '2'], ['3', '4']])}
    ),
    node_sets={
        'a': tfgnn.NodeSet.from_fields(
            features={'f': rt([[1.0, 2.0], [3.0, 4.0]])},
            sizes=dt([[1, 1], [1, 1]]),
        ),
    },
    edge_sets={
        'a->a': tfgnn.EdgeSet.from_fields(
            features={
                'w': rt([[1.0, 2.0], [3.0]]),
            },
            sizes=dt([[2, 0], [0, 1]]),
            adjacency=tfgnn.Adjacency.from_indices(
                ('a', rt([[0, 1], [0]])),
                ('a', rt([[1, 2], [0]])),
            ),
        ),
    },
)


class EvalDagSimpleTest(tf.test.TestCase):

  def testPassthrough1(self):
    seeds = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        ),
        name='input',
    )
    model = tf.keras.Model(seeds, seeds)
    program, artifacts = lib.create_program(model)
    self.assertSetEqual(set(program.layers.keys()), {'input', 'sink'})
    self.assertEmpty(set(artifacts.models.keys()))
    self.assertProtoEquals(
        """
        stages { id: "stage0" layer_id: "input" }
        stages {
          id: "stage1"
          layer_id: "sink"
          input_matchers {
            stage_id: "stage0"
          }
        }
        """,
        program.eval_dag,
    )
    io_config = pb.IOFeatures()
    self.assertTrue(program.layers['sink'].config.Unpack(io_config))
    self.assertProtoEquals('feature_names: "__output__"', io_config)

  def testPassthrough2(self):
    inputs = {
        'i1': tf.keras.Input([], dtype=tf.string, name='input1'),
        'i2': tf.keras.Input([], dtype=tf.string, name='input2'),
    }
    model = tf.keras.Model(inputs, inputs)
    program, artifacts = lib.create_program(model)
    self.assertSetEqual(
        set(program.layers.keys()), {'input1', 'input2', 'sink'}
    )
    self.assertEmpty(set(artifacts.models.keys()))
    self.assertProtoEquals(
        """
        stages { id: "stage0" layer_id: "input1" }
        stages { id: "stage1" layer_id: "input2" }
        stages {
          id: "stage2"
          layer_id: "sink"
          input_matchers {
            stage_id: "stage0"
          }
          input_matchers {
            stage_id: "stage1"
          }
        }
        """,
        program.eval_dag,
    )
    io_config = pb.IOFeatures()
    self.assertTrue(program.layers['sink'].config.Unpack(io_config))
    self.assertProtoEquals('feature_names: ["i1", "i2"]', io_config)

  def testIOAdapter(self):
    seeds = tf.keras.Input(
        type_spec=tf.type_spec_from_value(context_graph),
        name='input',
    )
    model = tf.keras.Model(seeds, seeds)
    program, artifacts = lib.create_program(model)
    self.assertSetEqual(set(program.layers.keys()), {'input', 'model0', 'sink'})
    self.assertSetEqual(set(artifacts.models.keys()), {'model0'})
    io_config = pb.IOFeatures()
    self.assertTrue(program.layers['sink'].config.Unpack(io_config))
    self.assertProtoEquals('feature_names : "context/s"', io_config)

  def testNoSamplingPrimitives(self):
    i = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        ),
        name='input',
    )
    o1 = tf.strings.join(['x', i], separator='-')
    o2 = tf.strings.join(['y', i], separator='-')
    o = tf.concat([o1, o2], axis=-1)
    model = tf.keras.Model(i, o)
    program, artifacts = lib.create_program(model)
    self.assertSetEqual(set(program.layers.keys()), {'input', 'model0', 'sink'})
    self.assertProtoEquals(
        """
        stages { id: "stage0" layer_id: "input" }
        stages {
          id: "stage1"
          layer_id: "model0"
          input_matchers { stage_id: "stage0" }
        }
        stages {
          id: "stage2"
          layer_id: "sink"
          input_matchers { stage_id: "stage1" }
        }
        """,
        program.eval_dag,
    )

    self.assertSetEqual(set(artifacts.models.keys()), {'model0'})
    model = artifacts.models['model0']
    actual = model(tf.ragged.constant([[], [b'a'], [b'b', b'c']]))
    expected = tf.ragged.constant(
        [[], [b'x-a', b'y-a'], [b'x-b', b'x-c', b'y-b', b'y-c']]
    )
    self.assertAllEqual(expected.flat_values, actual.flat_values)
    self.assertAllEqual(expected.row_lengths(), actual.row_lengths())


class SpecializableLambda(tf.keras.layers.Lambda, interfaces.SamplingPrimitive):
  pass


class Add2(core.CompositeLayer):

  def symbolic_call(self, inputs):
    a, b = inputs
    return a + b


class Sub2(core.CompositeLayer):

  def symbolic_call(self, inputs):
    a, b = inputs
    return a - b


class Add3(core.CompositeLayer):

  def symbolic_call(self, inputs):
    a, b, c = inputs
    add2 = Add2()
    return add2([add2([a, b]), c])


class EvalDagExecutionTest(tf.test.TestCase):
  """Checks if running eval dag matches its sampling model outputs."""

  def _run(
      self,
      program: pb.Program,
      artifacts: lib.Artifacts,
      inputs: Dict[str, tfgnn.Field],
  ) -> Dict[str, tfgnn.Field]:
    def get_layer(layer_id: str) -> pb.Layer:
      self.assertIn(layer_id, program.layers)
      return program.layers[layer_id]

    def check_specs(expected_specs, actual_specs, msg):
      # pylint:disable=private-access
      self.assertEqual(len(expected_specs), len(actual_specs), msg=msg)
      for expected, actual in zip(expected_specs, actual_specs):
        actual_spec = tf.type_spec_from_value(actual)
        # Explicitly set batch dimension to None.
        actual_spec = actual_spec._unbatch()._batch(None)
        # Wrap as a Keras tensor to re-use library functions.
        actual = tf.keras.Input(type_spec=actual_spec)
        self.assertProtoEquals(expected, lib._get_spec_pb(actual), msg={msg})

    def execute(
        eval_dag: pb.EvalDAG, inputs: Dict[str, tfgnn.Field], scope: str
    ) -> List[tfgnn.Field]:
      self.assertGreaterEqual(len(eval_dag.stages), 2, msg=scope)
      self.assertEqual(
          get_layer(eval_dag.stages[0].layer_id).type, 'InputLayer', msg=scope
      )
      self.assertEqual(
          get_layer(eval_dag.stages[-1].layer_id).type, 'Sink', msg=scope
      )
      stage_id_to_outputs = {}
      for stage in eval_dag.stages:
        stage_inputs = []
        for matcher in stage.input_matchers:
          self.assertIn(
              matcher.stage_id,
              stage_id_to_outputs,
              msg=f'graph is disconnected, {stage.id}',
          )
          output = stage_id_to_outputs[matcher.stage_id]
          self.assertLess(
              matcher.output_index,
              len(output),
              msg=f'invalid stage input, {matcher.stage_id}->{stage.id}',
          )
          stage_inputs.append(output[matcher.output_index])

        layer = get_layer(stage.layer_id)
        check_specs(layer.inputs, stage_inputs, msg=layer.id)
        if layer.type == 'TFModel':
          self.assertIn(layer.id, artifacts.models)
          model = artifacts.models[layer.id]
          stage_outputs = tf.nest.flatten(model(stage_inputs))
        elif layer.type == 'InputLayer':
          stage_outputs = [inputs[layer.id]]
        elif layer.type == 'Sink':
          stage_outputs = stage_inputs
        else:
          self.assertTrue(hasattr(layer, 'eval_dag'), msg=layer.id)
          self.assertTrue(hasattr(layer, 'input_names'), msg=layer.id)
          substage_inputs = {}
          for index, name in enumerate(layer.input_names.feature_names):
            substage_inputs[name] = stage_inputs[index]
          substage_outputs = execute(
              layer.eval_dag, substage_inputs, scope=layer.id
          )
          stage_outputs = tf.nest.flatten(substage_outputs)
        stage_id_to_outputs[stage.id] = stage_outputs
        if layer.type == 'Sink':
          self.assertEmpty(layer.outputs, msg=str(program))
        else:
          check_specs(layer.outputs, stage_outputs, msg=layer.id)

      return stage_id_to_outputs[eval_dag.stages[-1].id]

    flat_result = execute(program.eval_dag, inputs, scope='root')
    io_config = pb.IOFeatures()
    self.assertTrue(program.layers['sink'].config.Unpack(io_config))
    result = {}
    for index, name in enumerate(io_config.feature_names):
      self.assertLess(index, len(flat_result))
      result[name] = flat_result[index]
    return result

  def testSimpleModel(self):
    i = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        ),
        name='input',
    )
    o1 = tf.strings.join(['x', i], separator='-')
    o2 = tf.strings.join(['y', i], separator='-')
    o = tf.concat([o1, o2], axis=-1)
    model = tf.keras.Model(i, o)
    program, artifacts = lib.create_program(model)

    inputs = tf.ragged.constant([[], [b'a'], [b'b', b'c']])
    actual = self._run(
        program,
        artifacts,
        {'input': inputs},
    )
    expected = {'__output__': model(inputs)}
    tf.nest.map_structure(self.assertAllEqual, expected, actual)

  def testWithGraphTensorAdapter(self):
    # pylint: disable=g-complex-comprehension
    edges = {
        name: tf.keras.Input(
            type_spec=tf.RaggedTensorSpec(
                [None, None], dtype=tf.int64, ragged_rank=1
            ),
            name=name,
        )
        for name in ('#source', '#target')
    }
    sampled_edges = tf.nest.map_structure(lambda t: t[:, :2], edges)
    graph = core.build_graph_tensor(edge_sets={'node,edge,node': sampled_edges})
    model = tf.keras.Model(edges, graph)

    program, artifacts = lib.create_program(model)
    input_edges = {
        '#source': tf.ragged.constant([[0, 0, 1], [1]], dtype=tf.int64),
        '#target': tf.ragged.constant([[1, 2, 0], [0]], dtype=tf.int64),
    }
    actual = self._run(
        program,
        artifacts,
        input_edges,
    )
    expected = lib.flatten_to_dict(model(input_edges))
    self.assertSetEqual(
        set(expected.keys()),
        {
            'nodes/node.#id',
            'nodes/node.#size',
            'edges/edge.#size',
            'edges/edge.#source',
            'edges/edge.#target',
        },
    )
    tf.nest.map_structure(self.assertAllEqual, expected, actual)

  def testComposite1(self):
    i = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.float32, ragged_rank=1
        ),
        name='input',
    )
    x = i * 1.0
    y = i * 10.0
    z = Add2()([x, y])
    o = Add3()([x, y, z])
    model = tf.keras.Model(i, o)
    program, artifacts = lib.create_program(model)

    inputs = tf.ragged.constant([[], [1.0], [2.0, 3.0]])
    actual = self._run(
        program,
        artifacts,
        {'input': inputs},
    )
    expected = {'__output__': model(inputs)}
    tf.nest.map_structure(self.assertAllEqual, expected, actual)

  def testComposite2(self):
    inputs = []
    for i in range(32):
      inputs.append(tf.keras.Input([], name=f'i{i}', dtype=tf.int64))
    frontier = inputs
    for r in range(5):
      next_frontier = []
      for c, (x, y) in enumerate(zip(frontier[::2], frontier[1::2])):
        next_frontier.append(Sub2(name=f'add/{r}/{c}')([x, y]))
      frontier = next_frontier
    self.assertLen(frontier, 1)
    model = tf.keras.Model(inputs, frontier[0])
    program, artifacts = lib.create_program(model)

    input_values = tf.random.uniform([32, 16], 0, 100, dtype=tf.int64)
    input_values = tf.unstack(input_values, axis=0)
    actual = self._run(
        program,
        artifacts,
        {f'i{i}': v for i, v in enumerate(input_values)},
    )
    expected = {'__output__': model(input_values)}
    tf.nest.map_structure(self.assertAllEqual, expected, actual)


class StagesCreationTest(tf.test.TestCase):

  def _get_layer_names(self, stage) -> Set[str]:
    return {n.layer.name for n in stage.nodes}

  def _build_stages(self, inputs, outputs):
    model = tf.keras.Model(inputs, outputs)
    node = model.output.node
    nodes_dag = lib.build_ordered_dag(node)
    return lib.create_stages_dag(nodes_dag)

  def testSingleNode(self):
    i = tf.keras.Input([], name='input')
    stages_dag = self._build_stages(i, i)
    self.assertEqual(stages_dag.number_of_nodes(), 1)
    self.assertEqual(stages_dag.number_of_edges(), 0)
    nodes = lib.ordered_nodes(stages_dag)
    self.assertSetEqual(self._get_layer_names(nodes[0]), {'input'})

  def testIdentityTransform(self):
    i = tf.keras.Input([], name='input')
    o = tf.keras.layers.Layer(name='layer')(i)
    stages_dag = self._build_stages(i, o)
    self.assertEqual(stages_dag.number_of_nodes(), 2)
    self.assertEqual(stages_dag.number_of_edges(), 1)
    nodes = lib.ordered_nodes(stages_dag)
    self.assertEqual(nodes[0].get_single_node().layer.name, 'input')
    self.assertEqual(nodes[1].get_single_node().layer.name, 'layer')

  def testTrivialGrouping(self):
    i = tf.keras.Input([], name='input')
    o = i
    o = tf.keras.layers.Layer(name='1')(o)
    o = tf.keras.layers.Layer(name='2')(o)
    o = tf.keras.layers.Layer(name='3')(o)
    stages_dag = self._build_stages(i, o)
    self.assertEqual(stages_dag.number_of_nodes(), 2)
    self.assertEqual(stages_dag.number_of_edges(), 1)
    nodes = lib.ordered_nodes(stages_dag)
    self.assertEqual(nodes[0].get_single_node().layer.name, 'input')
    self.assertSetEqual(self._get_layer_names(nodes[1]), {'1', '2', '3'})

  def testTwoInputsSingleOutput(self):
    xi = tf.keras.Input([], name='xi')
    x = tf.keras.layers.Layer(name='x1')(xi)
    x = tf.keras.layers.Layer(name='x2')(x)

    yi = tf.keras.Input([], name='yi')
    y = tf.keras.layers.Layer(name='y1')(yi)
    y = tf.keras.layers.Layer(name='y2')(y)
    y = tf.keras.layers.Layer(name='y3')(y)
    z = tf.keras.layers.Lambda(tf.add_n, name='add1')([x, y])
    z1 = tf.keras.layers.Layer(name='z1')(z)
    z2 = tf.keras.layers.Layer(name='z2')(z)
    z = tf.keras.layers.Lambda(tf.add_n, name='add2')([z1, z2])
    stages_dag = self._build_stages([xi, yi], z)
    self.assertEqual(stages_dag.number_of_nodes(), 2 * (1 + 1) + 1)
    self.assertEqual(stages_dag.number_of_edges(), 2 * (1 + 1))

    nodes = lib.ordered_nodes(stages_dag)
    self.assertSetEqual(self._get_layer_names(nodes[0]), {'xi'})
    self.assertSetEqual(self._get_layer_names(nodes[1]), {'x1', 'x2'})
    self.assertSetEqual(self._get_layer_names(nodes[2]), {'yi'})
    self.assertSetEqual(self._get_layer_names(nodes[3]), {'y1', 'y2', 'y3'})
    self.assertSetEqual(
        self._get_layer_names(nodes[4]), {'add1', 'add2', 'z1', 'z2'}
    )

    self.assertSetEqual(
        set(stages_dag.edges()),
        {
            (nodes[0], nodes[1]),
            (nodes[2], nodes[3]),
            (nodes[1], nodes[4]),
            (nodes[3], nodes[4]),
        },
    )
    self.assertSetEqual(stages_dag[0][1]['inputs'], {xi.ref()})
    self.assertSetEqual(stages_dag[1][4]['inputs'], {x.ref()})
    self.assertSetEqual(stages_dag[2][3]['inputs'], {yi.ref()})
    self.assertSetEqual(stages_dag[3][4]['inputs'], {y.ref()})

  def testSingleInputSingleOutput(self):
    i = tf.keras.Input([], name='i')
    x = tf.keras.layers.Layer(name='x1')(i)
    x = tf.keras.layers.Layer(name='x2')(x)
    x = tf.keras.layers.Layer(name='x3')(x)

    y = tf.keras.layers.Layer(name='y1')(i)
    y = tf.keras.layers.Layer(name='y2')(y)

    z = tf.keras.layers.Layer(name='z1')(i)

    o = tf.keras.layers.Lambda(tf.add_n, name='add1')([x, y])
    o = tf.keras.layers.Lambda(tf.add_n, name='add2')([o, z])
    stages_dag = self._build_stages(i, o)
    self.assertEqual(stages_dag.number_of_nodes(), 1 + 1)
    self.assertEqual(stages_dag.number_of_edges(), 1)

    nodes = lib.ordered_nodes(stages_dag)
    self.assertSetEqual(self._get_layer_names(nodes[0]), {'i'})
    self.assertSetEqual(
        self._get_layer_names(nodes[1]),
        {'x1', 'x2', 'x3', 'y1', 'y2', 'z1', 'add1', 'add2'},
    )

    self.assertSetEqual(
        set(stages_dag.edges()),
        {
            (nodes[0], nodes[1]),
        },
    )

  def testSpecialization(self):
    xi = tf.keras.Input([], name='xi')
    x1 = tf.keras.layers.Layer(name='x1')(xi)
    x2 = tf.keras.layers.Layer(name='x2')(x1)

    yi = tf.keras.Input([], name='yi')
    y1 = tf.keras.layers.Layer(name='y1')(yi)
    y2 = tf.keras.layers.Layer(name='y2')(yi)
    add_swap = SpecializableLambda(
        lambda xyy: (xyy[1] + xyy[2], xyy[0]), name='add_swap'
    )
    zi = add_swap([x2, y1, y2])

    z1 = tf.keras.layers.Layer(name='z1')(zi[0])
    z2 = tf.keras.layers.Layer(name='z2')(zi[1])
    z = tf.keras.layers.Lambda(tf.add_n, name='add')([z1, z2])
    stages_dag = self._build_stages([xi, yi], z)
    self.assertEqual(stages_dag.number_of_nodes(), 2 + 2 + 1 + 1)
    self.assertEqual(stages_dag.number_of_edges(), 2 + 2 + 1)

    nodes = lib.ordered_nodes(stages_dag)
    self.assertSetEqual(self._get_layer_names(nodes[0]), {'xi'})
    self.assertSetEqual(self._get_layer_names(nodes[1]), {'x1', 'x2'})
    self.assertSetEqual(self._get_layer_names(nodes[2]), {'yi'})
    self.assertSetEqual(self._get_layer_names(nodes[3]), {'y1', 'y2'})
    self.assertSetEqual(self._get_layer_names(nodes[4]), {'add_swap'})
    self.assertSetEqual(self._get_layer_names(nodes[5]), {'z1', 'z2', 'add'})
    self.assertSetEqual(stages_dag[1][4]['inputs'], {x2.ref()})
    self.assertSetEqual(stages_dag[3][4]['inputs'], {y1.ref(), y2.ref()})
    self.assertSetEqual(stages_dag[4][5]['inputs'], {zi[0].ref(), zi[1].ref()})

  def testSpecialization2(self):
    x = tf.keras.Input([], name='x')
    y = tf.keras.Input([], name='y')
    z = tf.keras.Input([], name='z')
    special1 = SpecializableLambda(lambda xyz: xyz, name='special1')
    k, l, m = special1([x, y, z])

    special2 = SpecializableLambda(lambda kl: kl[0] + kl[1], name='special2')
    o1 = special2([k, l])
    o3 = m
    o2 = tf.keras.layers.Layer(name='unit1')(m)
    o2 = tf.keras.layers.Layer(name='unit2')(o2)

    o = tf.keras.layers.Lambda(tf.add_n, name='add')([o1, o2, o3])
    stages_dag = self._build_stages([x, y, z], o)
    self.assertEqual(stages_dag.number_of_nodes(), 3 + 1 + 2 + 1)
    self.assertEqual(stages_dag.number_of_edges(), 3 + 2 + 2 + 1)
    nodes = lib.ordered_nodes(stages_dag)
    self.assertSetEqual(self._get_layer_names(nodes[0]), {'x'})
    self.assertSetEqual(self._get_layer_names(nodes[1]), {'y'})
    self.assertSetEqual(self._get_layer_names(nodes[2]), {'z'})
    self.assertSetEqual(self._get_layer_names(nodes[3]), {'special1'})
    self.assertSetEqual(self._get_layer_names(nodes[4]), {'special2'})
    self.assertSetEqual(self._get_layer_names(nodes[5]), {'unit1', 'unit2'})
    self.assertSetEqual(self._get_layer_names(nodes[6]), {'add'})
    self.assertSetEqual(stages_dag[3][6]['inputs'], {m.ref()})
    self.assertSetEqual(stages_dag[3][4]['inputs'], {k.ref(), l.ref()})

  def testSpecialization3(self):
    add = tf.keras.layers.Lambda(tf.add_n, name='add')
    identity = tf.keras.layers.Layer(name='identity')
    i = tf.keras.Input([], name='i')
    x1 = identity(i)
    x2 = identity(i)
    x = add([x1, x2])
    special1 = SpecializableLambda(lambda x: x, name='special1')
    y = special1(x)
    y1 = identity(y)
    y2 = identity(y)
    y4 = identity(y)
    y4 = identity(y4)
    y = add([y1, y2, y4])
    special2 = SpecializableLambda(lambda x: x, name='special2')
    z1, z2 = special2([i, y])
    z = add([z1, z2])
    o = add([z, i])
    stages_dag = self._build_stages(i, o)
    self.assertEqual(stages_dag.number_of_nodes(), 7)
    self.assertEqual(stages_dag.number_of_edges(), 6 + 2)
    nodes = lib.ordered_nodes(stages_dag)
    self.assertTrue(nodes[0].specialized)
    self.assertLen(nodes[1].nodes, 3)
    self.assertTrue(nodes[2].specialized)
    self.assertLen(nodes[3].nodes, 5)
    self.assertTrue(nodes[4].specialized)
    self.assertLen(nodes[5].nodes, 1)
    self.assertFalse(nodes[5].specialized)
    self.assertLen(nodes[6].nodes, 1)
    self.assertFalse(nodes[6].specialized)
    self.assertSetEqual(stages_dag[0][4]['inputs'], {i.ref()})
    self.assertSetEqual(stages_dag[4][5]['inputs'], {z1.ref(), z2.ref()})
    self.assertSetEqual(stages_dag[5][6]['inputs'], {z.ref()})
    self.assertSetEqual(stages_dag[0][6]['inputs'], {i.ref()})


class NodesOrderingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([0, 5, 10, 20])
  def testChain(self, depth: int):
    i = tf.keras.Input([], name='layer.0')
    o = i
    for index in range(1, depth + 1):
      o = tf.keras.layers.Layer(name=f'layer.{index}')(o)
    model = tf.keras.Model(i, o)
    dag = lib.build_ordered_dag(model.output.node)
    sorted_nodes = lib.ordered_nodes(dag)
    for index, node in enumerate(sorted_nodes):
      self.assertEqual(f'layer.{index}', node.layer.name)

  @parameterized.product(
      depth=[5, 10], mixing_strategy=['aabb', 'bbaa', 'abab']
  )
  def testTwoBranches(self, depth: int, mixing_strategy: str):
    i = tf.keras.Input([], name='input')
    o_a, o_b = i, i
    if mixing_strategy == 'aabb':
      for index in range(0, depth):
        o_a = tf.keras.layers.Layer(name=f'a_layer.{index}')(o_a)
      for index in range(0, depth):
        o_b = tf.keras.layers.Layer(name=f'b_layer.{index}')(o_b)
    elif mixing_strategy == 'bbaa':
      for index in range(0, depth):
        o_b = tf.keras.layers.Layer(name=f'b_layer.{index}')(o_b)
      for index in range(0, depth):
        o_a = tf.keras.layers.Layer(name=f'a_layer.{index}')(o_a)
    elif mixing_strategy == 'abab':
      for index in range(0, depth):
        o_a = tf.keras.layers.Layer(name=f'a_layer.{index}')(o_a)
        o_b = tf.keras.layers.Layer(name=f'b_layer.{index}')(o_b)
    else:
      assert False
    o = lib.Sink()([o_a, o_b])
    model = tf.keras.Model(i, o)
    dag = lib.build_ordered_dag(model.output.node)
    sorted_nodes = lib.ordered_nodes(dag)
    self.assertLen(sorted_nodes, 1 + depth * 2 + 1)
    for i, node in enumerate(sorted_nodes[1 : depth + 1]):
      self.assertEqual(f'a_layer.{i}', node.layer.name)
    for i, node in enumerate(sorted_nodes[depth + 1 : -1]):
      self.assertEqual(f'b_layer.{i}', node.layer.name)

  @parameterized.product(
      depth=[0, 5, 10], mixing_strategy=['aabb', 'bbaa', 'abab']
  )
  def testSharedLayers(self, depth: int, mixing_strategy: str):
    i = tf.keras.Input([], name='input')
    o_a, o_b = i, i
    layer_a = tf.keras.layers.Layer(name='a_layer')
    layer_b = tf.keras.layers.Layer(name='b_layer')
    if mixing_strategy == 'aabb':
      for _ in range(0, depth):
        o_a = layer_a(o_a)
      for _ in range(0, depth):
        o_b = layer_b(o_b)
    elif mixing_strategy == 'bbaa':
      for _ in range(0, depth):
        o_b = layer_b(o_b)
      for _ in range(0, depth):
        o_a = layer_a(o_a)
    elif mixing_strategy == 'abab':
      for _ in range(0, depth):
        o_a = layer_a(o_a)
        o_b = layer_b(o_b)
    else:
      assert False
    o = lib.Sink()([o_a, o_b])
    model = tf.keras.Model(i, o)
    dag = lib.build_ordered_dag(model.output.node)
    sorted_nodes = lib.ordered_nodes(dag)
    self.assertLen(sorted_nodes, 1 + depth * 2 + 1)
    for node in sorted_nodes[1 : depth + 1]:
      self.assertEqual('a_layer', node.layer.name)
    for node in sorted_nodes[depth + 1 : -1]:
      self.assertEqual('b_layer', node.layer.name)


class HelpersTest(tf.test.TestCase, parameterized.TestCase):
  # pylint:disable=private-access

  def testNodesSortKey(self):
    layer1 = tf.keras.layers.Layer(name='layer1')
    layer2 = tf.keras.layers.Layer(name='layer2')
    i = tf.keras.Input([])
    o1 = layer1(i)
    o2 = layer2(o1)
    o3 = layer1(o2)
    o4 = layer1(o3)
    o5 = layer2(o4)
    self.assertEqual(('layer1', 2), lib._get_node_sort_key(o4.node))
    self.assertEqual(('layer1', 1), lib._get_node_sort_key(o3.node))
    self.assertEqual(('layer1', 0), lib._get_node_sort_key(o1.node))

    self.assertEqual(('layer2', 1), lib._get_node_sort_key(o5.node))
    self.assertEqual(('layer2', 0), lib._get_node_sort_key(o2.node))


class ModelServingTest(tf.test.TestCase):

  def _save_and_load(self, model: tf.keras.Model) -> tf.keras.Model:
    temp_dir = self.create_tempdir().full_path
    lib.save_model(model, temp_dir)
    return tf.saved_model.load(temp_dir)

  def testSaveAndLoad(self):
    i1 = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.int64, ragged_rank=1
        ),
        name='i1',
    )
    i2 = tf.keras.Input([1], dtype=tf.int64, name='i2')

    o1 = i1 + i2
    o2 = i1 * i2
    model = tf.keras.Model([i1, i2], [o1, o2])
    model = lib.create_tf_stage_model(model.inputs, model.outputs)

    serving_model = self._save_and_load(model)
    fn = serving_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    v1 = rt([[1], [2, 3]], dtype=tf.int64)
    v2 = dt([[10], [20]], dtype=tf.int64)
    results = fn(argw=v1.flat_values, argw_1=v1.row_lengths(), argw_2=v2)
    expected = {
        'output_0': dt([11, 22, 23], dtype=tf.int64),  # flat_values 1
        'output_1': dt([1, 2], dtype=tf.int64),  # row_splits 1
        'output_2': dt([10, 40, 60], dtype=tf.int64),  # flat_values 2
        'output_3': dt([1, 2], dtype=tf.int64),  # row_splits 2
    }
    tf.nest.map_structure(self.assertAllEqual, expected, results)

  def testInputCompositeTensorBug(self):
    i0 = tf.keras.Input(
        type_spec=tf.type_spec_from_value(context_graph),
        name='i',
    )
    # It is important that model is NOT constructed from the Input layer.
    i = tf.keras.layers.Layer(name='test')(i0)
    o = tf.keras.layers.Layer(name='test')(i)
    with self.assertRaisesRegex(
        AttributeError,
        (
            'KerasTensor wraps TypeSpec GraphTensorSpec, which does not have a'
            ' dtype'
        ),
    ):
      tf.keras.Model(i, o)

  def testInputGraphTensor(self):
    i0 = tf.keras.Input(
        type_spec=tf.type_spec_from_value(context_graph), name='i'
    )
    # It is important that model is NOT constructed from the Input layer.
    i = tf.keras.layers.Layer(name='test')(i0)
    o = tf.keras.layers.Layer(name='test')(i)
    model = lib.create_tf_stage_model(i, o)

    inputs = tf.nest.flatten(context_graph, expand_composites=True)
    outputs = model.gnn_serving(inputs)
    tf.nest.map_structure(self.assertAllEqual, inputs, outputs)
    serving_model = self._save_and_load(model)
    fn = serving_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    kwargs = {
        ('argw' + (f'_{index}' if index > 0 else '')): value
        for index, value in enumerate(inputs)
    }
    flat_outputs = tf.nest.flatten(fn(**kwargs))
    tf.nest.map_structure(self.assertAllEqual, inputs, flat_outputs)


class AdaptForIOTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'single_dense',
          dt([[1.0, 2.0], [3.0, 4.0]]),
          {'__output__': dt([[1.0, 2.0], [3.0, 4.0]])},
      ),
      (
          'single_ragged',
          rt([[1, 2], [3]], tf.int32),
          {'__output__': rt([[1, 2], [3]], tf.int32)},
      ),
      (
          'tuple',
          (dt([1.0, 2.0]), rt([['a'], ['b', 'c']])),
          {'0': dt([1.0, 2.0]), '1': rt([['a'], ['b', 'c']])},
      ),
      (
          'list',
          (dt([[1], [2]]), rt([['a'], ['b', 'c']])),
          {'0': dt([[1], [2]]), '1': rt([['a'], ['b', 'c']])},
      ),
      (
          'dict',
          {'s1': dt([1]), 's2': dt([2]), 'rt': rt([[1.0]])},
          {'s1': dt([1]), 's2': dt([2]), 'rt': rt([[1.0]])},
      ),
      (
          'tuple_dict',
          ({'a': dt([1])}, {'a': dt([1]), 'b': dt([2])}),
          {'0/a': dt([1]), '1/a': dt([1]), '1/b': dt([2])},
      ),
      (
          'dict_dict',
          {'x': {'a': dt([1])}, 'a': dt([1]), 'b': dt([2])},
          {'x/a': dt([1]), 'a': dt([1]), 'b': dt([2])},
      ),
      (
          'graph_tensor',
          aa_graph,
          {
              'context/s': dt([['1', '2'], ['3', '4']]),
              'nodes/a.#size': dt([[1, 1], [1, 1]]),
              'nodes/a.f': rt([[1.0, 2.0], [3.0, 4.0]]),
              'edges/a->a.#size': dt([[2, 0], [0, 1]]),
              'edges/a->a.#source': rt([[0, 1], [0]]),
              'edges/a->a.#target': rt([[1, 2], [0]]),
              'edges/a->a.w': rt([[1.0, 2.0], [3.0]]),
          },
      ),
      (
          'dual_encoder',
          {'query': context_graph, 'document': aa_graph},
          {
              'query/context/s': dt([[1], [2]]),
              'document/context/s': dt([['1', '2'], ['3', '4']]),
              'document/nodes/a.#size': dt([[1, 1], [1, 1]]),
              'document/nodes/a.f': rt([[1.0, 2.0], [3.0, 4.0]]),
              'document/edges/a->a.#size': dt([[2, 0], [0, 1]]),
              'document/edges/a->a.#source': rt([[0, 1], [0]]),
              'document/edges/a->a.#target': rt([[1, 2], [0]]),
              'document/edges/a->a.w': rt([[1.0, 2.0], [3.0]]),
          },
      ),
  ])
  def testConversion(self, nest, expected):
    tf.nest.map_structure(
        self.assertAllEqual, expected, lib.flatten_to_dict(nest)
    )


if __name__ == '__main__':
  tf.test.main()
