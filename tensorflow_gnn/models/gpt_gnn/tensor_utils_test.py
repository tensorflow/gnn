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
"""Tests for graph-tensor utils."""

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.gpt_gnn import tensor_utils


class SegmentSamplesToIndicesTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for generating element ids from segment sample counts."""

  def testRespectedSegmentBoundariesAndShape(self):
    samples_per_segment = tf.convert_to_tensor([3, 3, 2])
    segment_sizes = tf.convert_to_tensor([4, 4, 4])
    actual = tensor_utils.segment_samples_to_indices(
        samples_per_segment,
        segment_sizes,
        sum_sample_sizes_hint=8,
        sum_segment_sizes_hint=12,
    )
    self.assertEqual(actual.shape, (8,))
    self.assertAllInRange(actual[0:3], 0, 3)
    self.assertAllInRange(actual[3:6], 4, 7)
    self.assertAllInRange(actual[6:8], 8, 11)

  def testEmptySegments(self):
    samples_per_segment = tf.convert_to_tensor([0, 3, 3, 1])
    segment_sizes = tf.convert_to_tensor([0, 4, 4, 1])
    actual = tensor_utils.segment_samples_to_indices(
        samples_per_segment,
        segment_sizes,
        sum_sample_sizes_hint=7,
        sum_segment_sizes_hint=9,
    )
    self.assertEqual(actual.shape, (7,))
    self.assertAllInRange(actual[0:3], 0, 3)
    self.assertAllInRange(actual[3:6], 4, 7)
    self.assertEqual(actual[6], 8)

  def testDeterministicNodeIds(self):
    samples_per_segment = tf.convert_to_tensor([3, 3, 2])
    segment_sizes = tf.convert_to_tensor([4, 4, 4])
    seed = 1234
    tf.random.set_seed(seed)
    result1 = tensor_utils.segment_samples_to_indices(
        samples_per_segment,
        segment_sizes,
        sum_sample_sizes_hint=8,
        sum_segment_sizes_hint=12,
        seed=seed,
    )
    tf.random.set_seed(seed)
    result2 = tensor_utils.segment_samples_to_indices(
        samples_per_segment,
        segment_sizes,
        sum_sample_sizes_hint=8,
        sum_segment_sizes_hint=12,
        seed=seed,
    )
    self.assertAllEqual(result1, result2)

  def testInvalidSegmentCounts(self):
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r'Segment sizes in .* must be greater than or equal.*',
    ):
      tensor_utils.segment_samples_to_indices(
          tf.convert_to_tensor([4, 5, 5]), tf.convert_to_tensor([3, 3, 3])
      )

  def testMismatchingShapes(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Tensor shapes for .* and.* has to match.*',
    ):
      tensor_utils.segment_samples_to_indices(
          tf.convert_to_tensor([1, 2]), tf.convert_to_tensor([3, 3, 3])
      )


class SamplePerSegmentTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for generating sample count per segment."""

  @parameterized.named_parameters(
      dict(
          testcase_name='SegmentCountsRegular',
          segment_ids=tf.convert_to_tensor([10, 13, 12]),
          max_negative_samples=12,
          expected=tf.convert_to_tensor([4, 4, 4]),
      ),
      dict(
          testcase_name='SegmentCountsEmpty',
          segment_ids=tf.convert_to_tensor([0, 0, 0]),
          max_negative_samples=4,
          expected=tf.convert_to_tensor([0, 0, 0]),
      ),
      dict(
          testcase_name='SegmentCountsSmallSegments',
          segment_ids=tf.convert_to_tensor([13, 3, 5]),
          max_negative_samples=12,
          expected=tf.convert_to_tensor([4, 3, 4]),
      ),
      dict(
          testcase_name='SegmentCountsSmallSampleCount',
          segment_ids=tf.convert_to_tensor([13, 3, 5]),
          max_negative_samples=3,
          expected=tf.convert_to_tensor([1, 1, 1]),
      ),
      dict(
          testcase_name='SegmentCountsZeroPerSegmentSampleCount',
          segment_ids=tf.convert_to_tensor([13, 3, 5, 7]),
          max_negative_samples=3,
          expected=tf.convert_to_tensor([1, 1, 1, 1]),
      ),
  )
  def testNumSamplesPerSegment(
      self, segment_ids, max_negative_samples, expected
  ):
    actual = tensor_utils._num_samples_per_segment(
        segment_ids, max_negative_samples
    )
    self.assertAllEqual(actual, expected)


class FindIndiceDiffsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for finding tf.tensor difference with a max capacity per row."""

  @parameterized.named_parameters(
      dict(
          testcase_name='FindIndicesDiffsAllUnique',
          indices=tf.convert_to_tensor([[10, 13, 12], [10, 13, 12]]),
          another_indices=tf.convert_to_tensor([[1, 2, -1], [3, -1, -1]]),
          max_cols=4,
          expected=tf.convert_to_tensor([[10, 12, 13], [10, 12, 13]]),
      ),
      dict(
          testcase_name='FindIndicesDiffsSmallerDiff',
          indices=tf.convert_to_tensor([[10, 13, 12], [5, 2, 1]]),
          another_indices=tf.convert_to_tensor([[13, 12, -1], [5, -1, -1]]),
          max_cols=4,
          expected=tf.convert_to_tensor([[10], [1]]),
      ),
      dict(
          testcase_name='FindIndicesDiffsRegular',
          indices=tf.convert_to_tensor([[10, 13, 12], [5, 2, 1]]),
          another_indices=tf.convert_to_tensor([[1, 2, -1], [3, -1, -1]]),
          max_cols=4,
          expected=tf.convert_to_tensor([[10, 12, 13], [1, 2, 5]]),
      ),
  )
  def testFindIndiceDiffs(self, indices, another_indices, max_cols, expected):
    actual = tensor_utils._find_different_indices(
        indices, another_indices, max_cols
    )
    self.assertAllEqual(actual, expected)


class GetConnectedNodesTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for gathering connected nodes over a scalar GraphTensor."""

  def testConnectedNodes(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'a': tfgnn.NodeSet.from_fields(
                features={'f': tf.convert_to_tensor([1.0, 2.0])},
                sizes=tf.convert_to_tensor([2]),
            ),
            'b': tfgnn.NodeSet.from_fields(
                features={}, sizes=tf.convert_to_tensor([4])
            ),
        },
        edge_sets={
            'a->b': tfgnn.EdgeSet.from_fields(
                sizes=tf.convert_to_tensor([3]),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('a', tf.convert_to_tensor([0, 1, 1])),
                    ('b', tf.convert_to_tensor([0, 1, 3])),
                ),
            ),
        },
    )
    connected_nodes = tensor_utils._get_connected_node_ids(
        graph, edge_set_names=['a->b'], target_node_tag=tfgnn.SOURCE
    )
    self.assertEqual(connected_nodes.source_node_name, 'b')
    self.assertAllEqual(
        connected_nodes.source_node_ids, tf.convert_to_tensor([0, 1, 3])
    )
    self.assertEqual(connected_nodes.target_node_name, 'a')
    self.assertAllEqual(
        connected_nodes.target_node_ids, tf.convert_to_tensor([0, 1, 1])
    )

  def testIncorrectEdgeSetNames(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'a': tfgnn.NodeSet.from_fields(
                features={'f': tf.convert_to_tensor([1.0, 2.0])},
                sizes=tf.convert_to_tensor([2]),
            ),
            'b': tfgnn.NodeSet.from_fields(
                features={}, sizes=tf.convert_to_tensor([4])
            ),
            'c': tfgnn.NodeSet.from_fields(
                features={}, sizes=tf.convert_to_tensor([3])
            ),
        },
        edge_sets={
            'a->b': tfgnn.EdgeSet.from_fields(
                sizes=tf.convert_to_tensor([3]),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('a', tf.convert_to_tensor([0, 1, 1])),
                    ('b', tf.convert_to_tensor([0, 1, 3])),
                ),
            ),
            'a->c': tfgnn.EdgeSet.from_fields(
                sizes=tf.convert_to_tensor([2]),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('a', tf.convert_to_tensor([0, 1])),
                    ('c', tf.convert_to_tensor([0, 1])),
                ),
            ),
        },
    )
    with self.assertRaisesRegex(
        ValueError,
        r'source_name and target_name does not match among the.*',
    ):
      tensor_utils._get_connected_node_ids(
          graph, edge_set_names=['a->b', 'a->c'], target_node_tag=tfgnn.SOURCE
      )


class SampleUnconnectedNodesTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for sampling unconnected nodes over a scalar GraphTensor."""

  @parameterized.named_parameters(
      dict(
          testcase_name='UnconnectedNodesCapped',
          graph=tfgnn.GraphTensor.from_pieces(
              node_sets={
                  'a': tfgnn.NodeSet.from_fields(
                      features={'f': tf.convert_to_tensor([1.0, 2.0])},
                      sizes=tf.convert_to_tensor([2]),
                  ),
                  'b': tfgnn.NodeSet.from_fields(
                      features={}, sizes=tf.convert_to_tensor([4])
                  ),
              },
              edge_sets={
                  'a->b': tfgnn.EdgeSet.from_fields(
                      sizes=tf.convert_to_tensor([3]),
                      adjacency=tfgnn.Adjacency.from_indices(
                          ('a', tf.convert_to_tensor([0, 1, 1])),
                          ('b', tf.convert_to_tensor([0, 1, 3])),
                      ),
                  ),
              },
          ),
          edge_set_names=['a->b'],
          max_negative_samples=4,
          sample_buffer_scale=2,
          negative_samples_node_tag=tfgnn.TARGET,
          expected=tf.convert_to_tensor([[1, 2], [0, 2]]),
      ),
      dict(
          testcase_name='EmptyNodes',
          graph=tfgnn.GraphTensor.from_pieces(
              node_sets={
                  'a': tfgnn.NodeSet.from_fields(
                      features={}, sizes=tf.convert_to_tensor([0])
                  ),
                  'b': tfgnn.NodeSet.from_fields(
                      features={}, sizes=tf.convert_to_tensor([4])
                  ),
              },
              edge_sets={
                  'a->b': tfgnn.EdgeSet.from_fields(
                      sizes=tf.convert_to_tensor([0]),
                      adjacency=tfgnn.Adjacency.from_indices(
                          ('a', tf.convert_to_tensor([], dtype=tf.int32)),
                          ('b', tf.convert_to_tensor([], dtype=tf.int32)),
                      ),
                  ),
              },
          ),
          edge_set_names=['a->b'],
          max_negative_samples=4,
          sample_buffer_scale=2,
          negative_samples_node_tag=tfgnn.TARGET,
          expected=tf.reshape(tf.convert_to_tensor([]), [0, 4]),
      ),
      dict(
          testcase_name='EmptyEdges',
          graph=tfgnn.GraphTensor.from_pieces(
              node_sets={
                  'a': tfgnn.NodeSet.from_fields(
                      features={}, sizes=tf.convert_to_tensor([3])
                  ),
                  'b': tfgnn.NodeSet.from_fields(
                      features={}, sizes=tf.convert_to_tensor([2])
                  ),
              },
              edge_sets={
                  'a->b': tfgnn.EdgeSet.from_fields(
                      sizes=tf.convert_to_tensor([0]),
                      adjacency=tfgnn.Adjacency.from_indices(
                          ('a', tf.convert_to_tensor([], dtype=tf.int32)),
                          ('b', tf.convert_to_tensor([], dtype=tf.int32)),
                      ),
                  ),
              },
          ),
          edge_set_names=['a->b'],
          max_negative_samples=4,
          sample_buffer_scale=1,
          negative_samples_node_tag=tfgnn.TARGET,
          expected=tf.convert_to_tensor([[0, 1], [0, 1], [0, 1]]),
      ),
  )
  def test(
      self,
      graph,
      edge_set_names,
      max_negative_samples,
      sample_buffer_scale,
      negative_samples_node_tag,
      expected,
  ):
    actual = tensor_utils.sample_unconnected_nodes(
        graph,
        edge_set_names=edge_set_names,
        max_negative_samples=max_negative_samples,
        negative_samples_node_tag=negative_samples_node_tag,
        sample_buffer_scale=sample_buffer_scale,
    )
    self.assertAllEqual(actual, expected)

  def testUnconnectedNodesMultiComponent(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'a': tfgnn.NodeSet.from_fields(
                features={}, sizes=tf.convert_to_tensor([2, 3])
            ),
            'b': tfgnn.NodeSet.from_fields(
                features={}, sizes=tf.convert_to_tensor([4, 2])
            ),
        },
        edge_sets={
            'a->b': tfgnn.EdgeSet.from_fields(
                sizes=tf.convert_to_tensor([3, 1]),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('a', tf.convert_to_tensor([0, 1, 1, 2])),
                    ('b', tf.convert_to_tensor([0, 1, 3, 5])),
                ),
            ),
        },
    )
    actual = tensor_utils.sample_unconnected_nodes(
        graph,
        edge_set_names=['a->b'],
        max_negative_samples=3,
        negative_samples_node_tag=tfgnn.TARGET,
    )
    self.assertEqual(actual.shape, (5, 3))
    # Check for first-component, with maximum common elements sampled from
    # the target node indices with existing edges.
    self.assertAllInRange(actual[0, 0:2], 0, 3)  # existing 1 edge in segment1
    self.assertAllInRange(actual[1, 0:1], 0, 3)  # existing 2 edges in segment1
    self.assertAllInRange(actual[2, 0:3], 0, 3)
    self.assertAllInRange(actual[3, 0:3], 0, 3)
    self.assertAllInRange(actual[4, 0:3], 0, 3)

  def testEmptyEdgesets(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'nodes': tfgnn.NodeSet.from_fields(
                features={'f': tf.convert_to_tensor([1.0, 2.0])},
                sizes=tf.convert_to_tensor([2]),
            ),
        },
        edge_sets={
            'edges': tfgnn.EdgeSet.from_fields(
                features={},
                sizes=tf.convert_to_tensor([3]),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('nodes', tf.convert_to_tensor([0, 1, 1])),
                    ('nodes', tf.convert_to_tensor([0, 0, 1])),
                ),
            ),
        },
    )
    with self.assertRaisesRegex(ValueError, 'edge_set_names cant be empty'):
      tensor_utils.sample_unconnected_nodes(
          graph,
          edge_set_names=[],
          max_negative_samples=3,
          negative_samples_node_tag=tfgnn.TARGET,
      )

  def testInvalidEdgesets(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'nodes1': tfgnn.NodeSet.from_fields(
                features={'f': tf.convert_to_tensor([1.0, 2.0])},
                sizes=tf.convert_to_tensor([2]),
            ),
            'nodes2': tfgnn.NodeSet.from_fields(
                features={'f': tf.convert_to_tensor([1.0])},
                sizes=tf.convert_to_tensor([1]),
            ),
        },
        edge_sets={
            'edges1': tfgnn.EdgeSet.from_fields(
                features={},
                sizes=tf.convert_to_tensor([3]),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('nodes1', tf.convert_to_tensor([0, 1, 1])),
                    ('nodes1', tf.convert_to_tensor([0, 0, 1])),
                ),
            ),
            'edges2': tfgnn.EdgeSet.from_fields(
                features={},
                sizes=tf.convert_to_tensor([1]),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('nodes2', tf.convert_to_tensor([0])),
                    ('nodes2', tf.convert_to_tensor([0])),
                ),
            ),
        },
    )
    with self.assertRaisesRegex(
        ValueError,
        r'source_name and target_name doesnt match among the edge_sets in .*',
    ):
      tensor_utils.sample_unconnected_nodes(
          graph,
          edge_set_names=['edges1', 'edges2'],
          max_negative_samples=3,
          negative_samples_node_tag=tfgnn.TARGET,
      )


if __name__ == '__main__':
  tf.test.main()
