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
"""Tests for padding_ops.py."""
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_test_utils as tu
from tensorflow_gnn.graph import padding_ops as ops
from tensorflow_gnn.graph import preprocessing_common as preprocessing

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


class PaddingToTotalSizesTest(tu.GraphTensorTestBase):
  """Tests for context, node sets and edge sets creation."""
  test_2_a2b4_ab3_graph = gt.GraphTensor.from_pieces(
      context=gt.Context.from_fields(features={'f': as_tensor(['X', 'Y'])}),
      node_sets={
          'a':
              gt.NodeSet.from_fields(
                  features={'f': as_tensor([1., 2.])}, sizes=as_tensor([1, 1])),
          'b':
              gt.NodeSet.from_fields(features={}, sizes=as_tensor([2, 2])),
      },
      edge_sets={
          'a->b':
              gt.EdgeSet.from_fields(
                  features={'weight': as_tensor([1., 2., 3.])},
                  sizes=as_tensor([2, 1]),
                  adjacency=adj.Adjacency.from_indices(
                      ('a', as_tensor([0, 1, 0])),
                      ('b', as_tensor([1, 2, 0])),
                  )),
      },
  )

  def asserHasStaticNRows(self, graph_piece):
    # pylint: disable=protected-access

    self.assertTrue(graph_piece.shape.is_fully_defined())
    components = graph_piece.spec._to_components(graph_piece)
    components_spec = graph_piece.spec._component_specs

    def map_fn(component_spec, component):
      msg = f'spec={component_spec}'
      if isinstance(component_spec, tf.TensorSpec):
        self.assertTrue(component.shape.is_fully_defined(), msg=msg)
        self.assertTrue(component_spec.shape.is_fully_defined(), msg=msg)
      elif isinstance(component_spec, tf.RaggedTensorSpec):
        self.assertTrue(
            component.row_lengths().shape.is_fully_defined(), msg=msg)
      else:
        self.asserHasStaticNRows(component)

    tf.nest.map_structure(map_fn, components_spec, components)

  def testEmpty(self):
    source = gt.GraphTensor.from_pieces()
    padded0, mask0 = ops.pad_to_total_sizes(
        source,
        preprocessing.SizeConstraints(
            total_num_components=0, total_num_nodes={}, total_num_edges={}))
    self.assertAllEqual(mask0, as_tensor([], tf.bool))
    self.asserHasStaticNRows(padded0)
    self.assertEqual(padded0.spec, source.spec)
    self.assertAllEqual(padded0.context.total_size, 0)
    self.assertAllEqual(padded0.context.sizes, as_tensor([], tf.int32))

    padded1, mask1 = ops.pad_to_total_sizes(
        source,
        preprocessing.SizeConstraints(
            total_num_components=1, total_num_nodes={}, total_num_edges={}))
    self.assertAllEqual(mask1, as_tensor([False], tf.bool))
    self.asserHasStaticNRows(padded1)
    self.assertAllEqual(padded1.context.total_size, 1)
    self.assertAllEqual(padded1.context.sizes, [1])

    padded2, mask2 = ops.pad_to_total_sizes(
        source,
        preprocessing.SizeConstraints(
            total_num_components=2, total_num_nodes={}, total_num_edges={}))
    self.assertAllEqual(mask2, as_tensor([False, False], tf.bool))
    self.asserHasStaticNRows(padded2)
    self.assertAllEqual(padded2.context.total_size, 2)
    self.assertAllEqual(padded2.context.sizes, [1, 1])

  @parameterized.parameters([
      dict(
          padding_size=3,
          padding_values=preprocessing.FeatureDefaultValues(context={
              'id': -1,
              'f2': .5,
          }),
          features={
              'id': [1],
              'f1': [1.],
              'f2': [[1., 2.]],
              'i3': [[[1, 2], [3, 4]]]
          },
          expected_features={
              'id': [1, -1, -1],
              'f1': [1., 0., 0.],
              'f2': [[1., 2.], [.5, .5], [.5, .5]],
              'i3': [[[1, 2], [3, 4]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
          }),
      dict(
          padding_size=3,
          padding_values=preprocessing.FeatureDefaultValues(context={
              'label': '?',
          }),
          features={
              'label': ['A', 'B'],
              'words': as_ragged([['1', '2'], ['3']])
          },
          expected_features={
              'label': ['A', 'B', '?'],
              'words': as_ragged([['1', '2'], ['3'], []])
          }),
  ])
  def testContextPadding(self, padding_size: int, features: gt.Fields,
                         expected_features: gt.Fields,
                         padding_values: preprocessing.FeatureDefaultValues):
    source = gt.GraphTensor.from_pieces(
        gt.Context.from_fields(shape=[], features=features))
    padded, mask = ops.pad_to_total_sizes(
        source,
        preprocessing.SizeConstraints(
            total_num_components=padding_size,
            total_num_nodes={},
            total_num_edges={}),
        padding_values=padding_values)
    self.assertAllEqual(
        mask, [i < source.total_num_components for i in range(padding_size)])
    self.asserHasStaticNRows(padded)
    self.assertFieldsEqual(padded.context.features, expected_features)

  @parameterized.parameters([
      dict(num_components=3, num_nodes=2, sizes=as_tensor([], tf.int64)),
      dict(num_components=3, num_nodes=2, sizes=as_tensor([1], tf.int64)),
      dict(num_components=3, num_nodes=5, sizes=as_tensor([1, 3], tf.int32)),
      dict(num_components=3, num_nodes=4, sizes=as_tensor([1, 2, 1], tf.int32)),
  ])
  def testNodeSetsPadding(self, num_components: int, num_nodes: int,
                          sizes: gt.Field):
    source = gt.GraphTensor.from_pieces(
        node_sets={'a': gt.NodeSet.from_fields(features={}, sizes=sizes)})
    padded, mask = ops.pad_to_total_sizes(
        source,
        preprocessing.SizeConstraints(
            total_num_components=num_components,
            total_num_nodes={'a': num_nodes},
            total_num_edges={}))
    self.assertAllEqual(
        mask, [i < source.total_num_components for i in range(num_components)])
    self.asserHasStaticNRows(padded)

    source_num_nodes = source.node_sets['a'].total_size
    padded_node_set = padded.node_sets['a']
    self.assertAllEqual(padded_node_set.sizes[:source.total_num_components],
                        sizes)
    self.assertAllEqual(
        tf.reduce_sum(padded_node_set.sizes[source.total_num_components:]),
        tf.cast(num_nodes, source_num_nodes.dtype) - source_num_nodes)

  @parameterized.parameters([
      dict(
          num_components=3,
          num_nodes=2,
          num_edges=2,
          sizes=as_tensor([], tf.int64)),
      dict(
          num_components=3,
          num_nodes=2,
          num_edges=4,
          sizes=as_tensor([1], tf.int64)),
      dict(
          num_components=3,
          num_nodes=4,
          num_edges=4,
          sizes=as_tensor([1, 3], tf.int32)),
      dict(
          num_components=4,
          num_nodes=5,
          num_edges=8,
          sizes=as_tensor([1, 2, 1], tf.int32)),
  ])
  def testEdgeSetsPadding(self, num_components: int, num_nodes: int,
                          num_edges: int, sizes: gt.Field):
    source_num_nodes = tf.reduce_sum(sizes)
    source_num_edges = source_num_nodes
    source = gt.GraphTensor.from_pieces(
        node_sets={'a': gt.NodeSet.from_fields(features={}, sizes=sizes)},
        edge_sets={
            'a->a':
                gt.EdgeSet.from_fields(
                    sizes=sizes,
                    adjacency=adj.Adjacency.from_indices(
                        ('a', tf.range(source_num_edges, dtype=sizes.dtype)),
                        ('a', tf.zeros([source_num_edges], dtype=sizes.dtype)),
                    )),
        })
    padded, _ = ops.pad_to_total_sizes(
        source,
        preprocessing.SizeConstraints(
            total_num_components=num_components,
            total_num_nodes={'a': num_nodes},
            total_num_edges={'a->a': num_edges},
        ))
    self.asserHasStaticNRows(padded)

    source_edge_set = source.edge_sets['a->a']
    padded_edge_set = padded.edge_sets['a->a']

    self.assertAllEqual(padded_edge_set.sizes[:source.total_num_components],
                        sizes)
    self.assertAllEqual(
        tf.reduce_sum(padded_edge_set.sizes[source.total_num_components:]),
        tf.cast(num_edges, source_num_edges.dtype) - source_num_edges)
    self.assertAllEqual(tf.size(padded_edge_set.adjacency.source), num_edges)
    self.assertAllEqual(padded_edge_set.adjacency.source[:source_num_edges],
                        source_edge_set.adjacency.source)
    self.assertAllInRange(padded_edge_set.adjacency.source[source_num_edges:],
                          source_num_nodes, num_nodes - 1)

    self.assertAllEqual(tf.size(padded_edge_set.adjacency.target), num_edges)
    self.assertAllInRange(padded_edge_set.adjacency.target[source_num_edges:],
                          source_num_nodes, num_nodes - 1)
    self.assertAllEqual(padded_edge_set.adjacency.target[:source_num_edges],
                        source_edge_set.adjacency.target)

  def testAdjacencyPaddingWithLinspace(self):
    padded, _ = ops.pad_to_total_sizes(
        self.test_2_a2b4_ab3_graph,
        preprocessing.SizeConstraints(
            total_num_components=4,
            total_num_nodes={
                'a': 100,
                'b': 200
            },
            total_num_edges={'a->b': 1000},
        ))
    padded_adjacency = padded.edge_sets['a->b'].adjacency

    # Check that edges cover all fake nodes.
    self.assertAllEqual(
        tf.unique(padded_adjacency.source[3:]).y, tf.range(2, 100))
    self.assertAllEqual(
        tf.unique(padded_adjacency.target[3:]).y, tf.range(4, 200))

  def testHyperAdjacencyPaddingWithLinspace(self):
    source = gt.GraphTensor.from_pieces(
        node_sets={
            'a': gt.NodeSet.from_fields(sizes=as_tensor([1])),
            'b': gt.NodeSet.from_fields(sizes=as_tensor([2])),
            'c': gt.NodeSet.from_fields(sizes=as_tensor([3])),
        },
        edge_sets={
            'a.b.c':
                gt.EdgeSet.from_fields(
                    sizes=as_tensor([3]),
                    adjacency=adj.HyperAdjacency.from_indices({
                        0: ('a', as_tensor([0, 0, 0])),
                        1: ('b', as_tensor([0, 1, 1])),
                        2: ('c', as_tensor([0, 1, 2])),
                    })),
        },
    )
    padded, _ = ops.pad_to_total_sizes(
        source,
        preprocessing.SizeConstraints(
            total_num_components=4,
            total_num_nodes={
                'a': 100,
                'b': 200,
                'c': 300
            },
            total_num_edges={'a.b.c': 1000},
        ))
    for tag, index_begin, index_end in [(0, 1, 100), (1, 2, 200), (2, 3, 300)]:
      indices = padded.edge_sets['a.b.c'].adjacency[tag][3:]
      self.assertAllEqual(tf.size(indices), 1000 - 3, msg=f'tag={tag}')
      self.assertAllEqual(
          tf.unique(indices[3:]).y,
          tf.range(index_begin, index_end),
          msg=f'tag={tag}')

  def testPaddingFromTfFunction(self):

    @tf.function(input_signature=[
        gt.GraphTensorSpec.from_piece_specs(
            gt.ContextSpec.from_field_specs(
                shape=[],
                features_spec={
                    'f': tf.TensorSpec(shape=[None], dtype=tf.float32)
                }))
    ])
    def pad_to_3(graph):
      return ops.pad_to_total_sizes(
          graph,
          preprocessing.SizeConstraints(
              total_num_components=3, total_num_nodes={}, total_num_edges={}))

    source = gt.GraphTensor.from_pieces(
        gt.Context.from_fields(shape=[], features={'f': [1., 2., 3.]}))
    padded, _ = pad_to_3(source)
    self.asserHasStaticNRows(padded)
    self.assertAllEqual(padded.context['f'], [1., 2., 3.])

    source = gt.GraphTensor.from_pieces(
        gt.Context.from_fields(shape=[], features={'f': [2.]}))
    padded, _ = pad_to_3(source)
    self.asserHasStaticNRows(padded)
    self.assertAllEqual(padded.context['f'], [2., 0., 0.])

    self.assertEqual(pad_to_3.experimental_get_tracing_count(), 1)

  @parameterized.parameters([0, 1, 2])
  def testOnlyContextPadding(self, min_nodes_per_component: int):

    target_b_nodes_count = 4 + (16 - 2) * min_nodes_per_component
    padded, mask = ops.pad_to_total_sizes(
        self.test_2_a2b4_ab3_graph,
        preprocessing.SizeConstraints(
            total_num_components=16,
            total_num_nodes={
                'a': 2,
                'b': target_b_nodes_count,
            },
            total_num_edges={'a->b': 3},
            min_nodes_per_component={'b': min_nodes_per_component}))
    self.asserHasStaticNRows(padded)
    self.assertAllEqual(mask, [True, True] + [False] * 14)
    self.assertAllEqual(padded.context.total_size, 16)
    self.assertAllEqual(padded.node_sets['a'].total_size, 2)
    self.assertAllEqual(padded.node_sets['b'].total_size, target_b_nodes_count)
    self.assertAllEqual(padded.edge_sets['a->b'].total_size, 3)

  def testMinNodesPerComponent1(self):
    padded, _ = ops.pad_to_total_sizes(
        self.test_2_a2b4_ab3_graph,
        preprocessing.SizeConstraints(
            total_num_components=5,
            total_num_nodes={
                'a': 2,
                'b': 9,
            },
            total_num_edges={'a->b': 3},
            min_nodes_per_component={'b': 1}))
    self.asserHasStaticNRows(padded)
    self.assertAllEqual(padded.context.total_size, 5)
    self.assertAllEqual(padded.node_sets['a'].sizes, [1, 1, 0, 0, 0])
    self.assertAllEqual(padded.node_sets['b'].sizes, [2, 2, 3, 1, 1])

  def testMinNodesPerComponent2(self):
    padded, _ = ops.pad_to_total_sizes(
        self.test_2_a2b4_ab3_graph,
        preprocessing.SizeConstraints(
            total_num_components=3,
            total_num_nodes={
                'a': 3,
                'b': 9,
            },
            total_num_edges={'a->b': 3},
            min_nodes_per_component={'a': 1, 'b': 1}))
    self.asserHasStaticNRows(padded)
    self.assertAllEqual(padded.context.total_size, 3)
    self.assertAllEqual(padded.node_sets['a'].sizes, [1, 1, 1])
    self.assertAllEqual(padded.node_sets['b'].sizes, [2, 2, 5])

  def testMinNodesPerComponent3(self):
    padded, _ = ops.pad_to_total_sizes(
        self.test_2_a2b4_ab3_graph,
        preprocessing.SizeConstraints(
            total_num_components=4,
            total_num_nodes={
                'a': 3,
                'b': 9,
            },
            total_num_edges={'a->b': 4},
            min_nodes_per_component={'b': 2}))
    self.asserHasStaticNRows(padded)
    self.assertAllEqual(padded.context.total_size, 4)
    self.assertAllEqual(padded.node_sets['a'].sizes, [1, 1, 1, 0])
    self.assertAllEqual(padded.node_sets['b'].sizes, [2, 2, 3, 2])

  def testMinNodesPerComponent4(self):
    padded, _ = ops.pad_to_total_sizes(
        self.test_2_a2b4_ab3_graph,
        preprocessing.SizeConstraints(
            total_num_components=5,
            total_num_nodes={
                'a': 2,
                'b': 4 + (5 - 2),
            },
            total_num_edges={'a->b': 3},
            min_nodes_per_component={'b': 1}))
    self.asserHasStaticNRows(padded)
    self.assertAllEqual(padded.context.total_size, 5)
    self.assertAllEqual(padded.node_sets['a'].sizes, [1, 1, 0, 0, 0])
    self.assertAllEqual(padded.node_sets['b'].sizes, [2, 2, 1, 1, 1])

  def testGraphTensorPadding(self):

    padded, mask = ops.pad_to_total_sizes(
        self.test_2_a2b4_ab3_graph,
        preprocessing.SizeConstraints(
            total_num_components=4,
            total_num_nodes={
                'a': 5,
                'b': 6
            },
            total_num_edges={'a->b': 6},
        ),
        padding_values=preprocessing.FeatureDefaultValues(
            context={'f': '?'},
            node_sets={'a': {
                'f': -1.
            }},
            edge_sets={'a->b': {
                'weight': -1.
            }},
        ))
    self.asserHasStaticNRows(padded)
    self.assertAllEqual(mask, [True, True, False, False])

    self.assertAllEqual(padded.context.sizes, [1, 1, 1, 1])
    self.assertAllEqual(padded.context['f'], ['X', 'Y', '?', '?'])

    self.assertAllEqual(padded.node_sets['a'].sizes, [1, 1, 3, 0])
    self.assertAllEqual(padded.node_sets['a']['f'], [1., 2., -1., -1., -1.])

    self.assertAllEqual(padded.node_sets['b'].sizes, [2, 2, 2, 0])

    self.assertAllEqual(padded.edge_sets['a->b'].sizes, [2, 1, 3, 0])
    self.assertAllEqual(padded.edge_sets['a->b']['weight'],
                        [1., 2., 3., -1., -1., -1.])

    source = padded.edge_sets['a->b'].adjacency.source
    self.assertLen(source, 6)
    self.assertAllEqual(
        source[:3],
        self.test_2_a2b4_ab3_graph.edge_sets['a->b'].adjacency.source)
    self.assertAllInRange(source[3:],
                          self.test_2_a2b4_ab3_graph.node_sets['a'].total_size,
                          5 - 1)

    target = padded.edge_sets['a->b'].adjacency.target
    self.assertLen(target, 6)
    self.assertAllEqual(
        target[:3],
        self.test_2_a2b4_ab3_graph.edge_sets['a->b'].adjacency.target)
    self.assertAllInRange(target[3:],
                          self.test_2_a2b4_ab3_graph.node_sets['b'].total_size,
                          6 - 1)

  @parameterized.parameters([1, 2, 3, 10])
  def testGraphDoublePaddingHasNoEffect(self, scale):
    size_constraints = preprocessing.SizeConstraints(
        total_num_components=4 * scale,
        total_num_nodes={
            'a': 5 * scale,
            'b': 6 * scale
        },
        total_num_edges={'a->b': 6 * scale})
    padded1, _ = ops.pad_to_total_sizes(self.test_2_a2b4_ab3_graph,
                                        size_constraints)
    padded2, mask2 = ops.pad_to_total_sizes(padded1, size_constraints)
    self.assertAllEqual(mask2, [True] * 4 * scale)
    tf.nest.assert_same_structure(
        padded1.spec, padded2.spec, expand_composites=True)
    for index, (a, b) in enumerate(
        zip(
            tf.nest.flatten(padded1, expand_composites=True),
            tf.nest.flatten(padded2, expand_composites=True))):
      self.assertAllEqual(a, b, msg=f'index={index}')

  def testRaisesOnIncompleteTotalSizes(self):

    no_node = preprocessing.SizeConstraints(
        total_num_components=4,
        total_num_nodes={'a': 5},
        total_num_edges={'a->b': 6},
    )
    self.assertRaisesRegex(
        ValueError, 'number of <b> nodes must be specified',
        lambda: ops.pad_to_total_sizes(self.test_2_a2b4_ab3_graph, no_node))

    no_edge = preprocessing.SizeConstraints(
        total_num_components=4,
        total_num_nodes={
            'a': 5,
            'b': 6
        },
        total_num_edges={},
    )
    self.assertRaisesRegex(
        ValueError, 'number of <a->b> edges must be specified',
        lambda: ops.pad_to_total_sizes(self.test_2_a2b4_ab3_graph, no_edge))

  @parameterized.parameters(['eager', 'tf.function', 'dataset1', 'dataset2'])
  def testRaisesOnImpossiblePadding(self, mode):

    def create_for_spec(size_constraints):
      graph = self.test_2_a2b4_ab3_graph
      self.assertFalse(ops.satisfies_size_constraints(graph, size_constraints))

      def pad(graph):
        tf.debugging.assert_equal(
            ops.satisfies_size_constraints(graph, size_constraints), False)
        return ops.pad_to_total_sizes(graph, size_constraints)

      if mode == 'eager':
        return lambda: pad(graph)

      if mode == 'tf.function':

        @tf.function
        def fn(graph):
          return pad(graph)

        return lambda: fn(graph)

      if mode == 'dataset1':

        def dataset1_case():
          ds = tf.data.Dataset.from_tensors(graph).map(pad)
          return ds.get_single_element()

        return dataset1_case

      if mode == 'dataset2':

        def dataset2_case():
          ds = tf.data.Dataset.from_tensors(graph)
          # sets the components dimension to None in spec.
          ds = ds.batch(1).map(lambda g: g.merge_batch_to_components())
          ds = ds.map(pad)
          return ds.get_single_element()

        return dataset2_case
      raise ValueError(mode)

    components_overflow = preprocessing.SizeConstraints(
        total_num_components=1,
        total_num_nodes={
            'a': 100,
            'b': 100
        },
        total_num_edges={'a->b': 100},
    )
    self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                           ('Could not pad graph as it already has more graph'
                            ' components then it is allowed'),
                           create_for_spec(components_overflow))

    nodes_overflow = preprocessing.SizeConstraints(
        total_num_components=4,
        total_num_nodes={
            'a': 1,
            'b': 100
        },
        total_num_edges={'a->b': 100},
    )
    self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                           ('Could not pad <a> as it already has more nodes'
                            ' then it is allowed by the'
                            r' `total_sizes\.total_num_nodes\[<a>\]`'),
                           create_for_spec(nodes_overflow))

    edges_overflow = preprocessing.SizeConstraints(
        total_num_components=4,
        total_num_nodes={
            'a': 100,
            'b': 100
        },
        total_num_edges={'a->b': 1},
    )
    self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                           ('Could not pad <a->b> as it already has more'
                            ' edges then it is allowed by the'
                            r' `total_sizes\.total_num_edges\[<a->b>\]`'),
                           create_for_spec(edges_overflow))

    no_b_node_to_use_for_fake_edge = preprocessing.SizeConstraints(
        total_num_components=4,
        total_num_nodes={
            'a': 2,
            'b': 6
        },
        total_num_edges={'a->b': 100},
    )
    self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'Could not create fake incident edges for the node set',
        create_for_spec(no_b_node_to_use_for_fake_edge))

  @parameterized.parameters([
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=2,
              total_num_nodes={
                  'a': 2,
                  'b': 4
              },
              total_num_edges={'a->b': 3},
          ),
          expected_result=True),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=4,
              total_num_nodes={
                  'a': 4,
                  'b': 6
              },
              total_num_edges={'a->b': 3},
          ),
          expected_result=True),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=4,
              total_num_nodes={
                  'a': 4,
                  'b': 6
              },
              total_num_edges={'a->b': 4},
          ),
          expected_result=True),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=1,
              total_num_nodes={
                  'a': 3,
                  'b': 6
              },
              total_num_edges={'a->b': 4},
          ),
          expected_result=False),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=1,
              total_num_nodes={
                  'a': 2,
                  'b': 6
              },
              total_num_edges={'a->b': 4},
          ),
          expected_result=False),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=3,
              total_num_nodes={
                  'a': 3,
                  'b': 6
              },
              total_num_edges={'a->b': 4},
              min_nodes_per_component={'a': 1}
          ),
          expected_result=True),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=3,
              total_num_nodes={
                  'a': 2,
                  'b': 100
              },
              total_num_edges={'a->b': 3},
          ),
          expected_result=True),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=3,
              total_num_nodes={
                  'a': 2,
                  'b': 100
              },
              total_num_edges={'a->b': 3},
              min_nodes_per_component={'b': 1}
          ),
          expected_result=True),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=3,
              total_num_nodes={
                  'a': 2,
                  'b': 100
              },
              total_num_edges={'a->b': 3},
              min_nodes_per_component={'a': 1}
          ),
          expected_result=False),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=3,
              total_num_nodes={
                  'a': 2,
                  'b': 100
              },
              total_num_edges={'a->b': 3},
              min_nodes_per_component={'a': 1, 'b': 1}
          ),
          expected_result=False),
      dict(
          total_sizes=preprocessing.SizeConstraints(
              total_num_components=4,
              total_num_nodes={
                  'a': 3,
                  'b': 6
              },
              total_num_edges={'a->b': 1},
          ),
          expected_result=False)
  ])
  def testSatifiesTotalSizes(self, total_sizes, expected_result):
    self.assertAllEqual(
        ops.satisfies_size_constraints(
            self.test_2_a2b4_ab3_graph, total_sizes=total_sizes),
        expected_result)


if __name__ == '__main__':
  tf.test.main()
