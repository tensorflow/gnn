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
"""Tests for corruption layers."""
from __future__ import annotations

import functools
from typing import Any, Callable, Mapping, Union

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.contrastive_losses import layers


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      {
          "testcase_name": "ShuffleEverything",
          "tensor": tf.reshape(tf.range(0, 8, dtype=tf.float32), (4, 2)),
          "expected": tf.convert_to_tensor(
              [[2.0, 3.0], [6.0, 7.0], [4.0, 5.0], [0.0, 1.0]]
          ),
          "rate": 1,
          "seed": 8191,
      },
      {
          "testcase_name": "ShuffleHalf",
          "tensor": tf.reshape(tf.range(0, 8, dtype=tf.float32), (4, 2)),
          "expected": tf.convert_to_tensor(
              [[0.0, 1.0], [6.0, 7.0], [4.0, 5.0], [2.0, 3.0]]
          ),
          "rate": 0.5,
          "seed": 8191,
      },
      {
          "testcase_name": "ShuffleHalfRagged",
          "tensor": tf.ragged.constant([[1], [2, 3], [4, 5, 6], [7, 8, 9, 10]]),
          "expected": tf.ragged.constant(
              [[7, 8, 9, 10], [2, 3], [4, 5, 6], [1]]
          ),
          "rate": 0.5,
          "seed": 1,  # Our second-best seed.
      },
      {
          "testcase_name": "ShuffleHalfEmpty2DRagged",
          "tensor": tf.ragged.constant([[], []]),
          "expected": tf.ragged.constant([[], []]),
          "rate": 0.5,
          "seed": 8191,
      },
      {
          "testcase_name": "ShuffleHalfEmpty2DDense",
          "tensor": tf.zeros([4, 0]),
          "expected": tf.zeros([4, 0]),
          "rate": 0.5,
          "seed": 8191,
      },
      {
          "testcase_name": "ShuffleHalf1DDense",
          "tensor": tf.constant([1, 2, 3, 4]),
          "expected": tf.constant([1, 4, 3, 2]),
          "rate": 0.5,
          "seed": 8191,
      },
  ])
  def test_shuffle_tensor(
      self, tensor: tfgnn.Field, expected: tfgnn.Field, rate: float, seed: int
  ):
    output = layers._shuffle_tensor(tensor, rate=rate, seed=seed)
    self.assertAllEqual(output, expected)

  def test_shuffle_ragged_tensor_same_row_dtype(self):
    # Check the row splits stay the same dtype.
    tensor = tf.ragged.constant([[1], []], row_splits_dtype=tf.int64)
    shuffled = layers._shuffle_tensor(tensor)
    self.assertEqual(tensor.row_splits.dtype, shuffled.row_splits.dtype)

    # Same for tf.int32.
    tensor = tf.ragged.constant([[1], []], row_splits_dtype=tf.int32)
    shuffled = layers._shuffle_tensor(tensor)
    self.assertEqual(tensor.row_splits.dtype, shuffled.row_splits.dtype)

  def test_shuffle_ragged_tensor_keras_input(self):
    tensor = tf.keras.layers.Input(shape=[None, None], ragged=True)
    _ = layers._shuffle_tensor(tensor)

  def test_shuffle_dense_empty_tensor_keras_input(self):
    tensor = tf.keras.layers.Input(shape=[0])
    _ = layers._shuffle_tensor(tensor, rate=0.5)

  def test_shuffle_tensor_proportion(self):
    # With variable seed, we either expect 50% (4 elements) to be shuffled or
    # no elements at all.
    tensor = tf.reshape(tf.range(0, 8, dtype=tf.float32), (4, 2))
    for seed in range(10):
      output = layers._shuffle_tensor(tensor, rate=0.5, seed=seed)
      num_different = tf.reduce_sum(tf.cast(output != tensor, tf.int32))
      self.assertIn(num_different, [tf.constant(0), tf.constant(4)])

    # Same for a ragged tensor, except it's easier to check row lengths here.
    tensor = tf.ragged.constant([[], [1], [2, 3], [4, 5, 6]])
    for seed in range(10):
      output = layers._shuffle_tensor(tensor, rate=0.5, seed=seed)
      num_different = tf.reduce_sum(
          tf.cast(output.row_lengths() != tensor.row_lengths(), tf.int32)
      )
      self.assertIn(num_different, [tf.constant(0), tf.constant(2)])

  @parameterized.named_parameters([
      {
          # We expect no shuffling to happen here.
          "testcase_name": "NoShuffling",
          "features": {
              "feature": tf.reshape(tf.range(0, 4, dtype=tf.float32), (2, 2)),
              "ragged_feature": tf.ragged.constant([[1], []]),
          },
          "expected": {
              "feature": tf.reshape(tf.range(0, 4, dtype=tf.float32), (2, 2)),
              "ragged_feature": tf.ragged.constant([[1], []]),
          },
          "corruptor": functools.partial(layers._shuffle_tensor, seed=8191),
          "corruption_spec": {"*": 0.0},
      },
      {
          # We expect both features to be shuffled according to default
          # corruption_spec rate.
          "testcase_name": "EverythingShuffled",
          "features": {
              "feature": tf.reshape(tf.range(0, 4, dtype=tf.float32), (2, 2)),
              "ragged_feature": tf.ragged.constant([[1], []]),
          },
          "expected": {
              "feature": tf.convert_to_tensor([[2.0, 3], [0.0, 1.0]]),
              "ragged_feature": tf.ragged.constant([[], [1]]),
          },
          "corruptor": functools.partial(layers._shuffle_tensor, seed=8191),
          "corruption_spec": {"*": 1.0},
      },
      {
          # We expect only the dense feature to be shuffled.
          "testcase_name": "OnlyDenseShuffled",
          "features": {
              "feature": tf.reshape(tf.range(0, 4, dtype=tf.float32), (2, 2)),
              "ragged_feature": tf.ragged.constant([[1], []]),
          },
          "expected": {
              "feature": tf.convert_to_tensor([[2.0, 3], [0.0, 1.0]]),
              "ragged_feature": tf.ragged.constant([[1], []]),
          },
          "corruptor": functools.partial(layers._shuffle_tensor, seed=8191),
          "corruption_spec": {"feature": 1.0, "*": 0.0},
      },
      {
          # We only expect ragged feature to be shuffled because of the
          # corruption_spec specifies no shuffling for dense features.
          "testcase_name": "RaggedShufflingOverride",
          "features": {
              "feature": tf.reshape(tf.range(0, 4, dtype=tf.float32), (2, 2)),
              "ragged_feature": tf.ragged.constant([[1], []]),
          },
          "expected": {
              "feature": tf.reshape(tf.range(0, 4, dtype=tf.float32), (2, 2)),
              "ragged_feature": tf.ragged.constant([[], [1]]),
          },
          "corruptor": functools.partial(layers._shuffle_tensor, seed=8191),
          "corruption_spec": {"feature": 0.0, "*": 1.0},
      },
  ])
  def test_corrupt(
      self,
      features: tfgnn.Fields,
      expected: tfgnn.Fields,
      corruptor: Callable[[tfgnn.Field, float], tfgnn.Field],
      corruption_spec: layers.FieldCorruptionSpec,
  ):
    shuffled = layers._corrupt_features(
        features,
        corruptor,
        corruption_spec=corruption_spec,
    )
    self.assertEqual(shuffled, expected)

  @parameterized.parameters([
      {
          "corruptor": layers.ShuffleFeaturesGlobally(
              corruption_spec=layers.CorruptionSpec(
                  node_set_corruption={"*": {"*": 1.0}},
                  edge_set_corruption={"*": {"*": 1.0}},
                  context_corruption={"*": 1.0},
              ),
              seed=8191,
          ),
          "context": tfgnn.Context.from_fields(
              features={
                  "feature": tf.convert_to_tensor([[1.0], [2.0], [3.0]]),
              },
          ),
          "node_set": tfgnn.NodeSet.from_fields(
              sizes=tf.convert_to_tensor([2, 1]),
              features={
                  "feature": tf.convert_to_tensor([[4.0], [5.0], [6.0]]),
              },
          ),
          "edge_set": tfgnn.EdgeSet.from_fields(
              sizes=tf.convert_to_tensor([2, 3]),
              adjacency=tfgnn.HyperAdjacency.from_indices({
                  tfgnn.SOURCE: ("node", tf.convert_to_tensor([0, 0, 0, 2, 2])),
                  tfgnn.TARGET: ("node", tf.convert_to_tensor([2, 1, 0, 0, 0])),
              }),
              features={
                  "feature": tf.convert_to_tensor([[7.0], [8.0], [9.0]]),
              },
          ),
          "expected_fields": {
              tfgnn.Context: {
                  "feature": tf.convert_to_tensor([[2.0], [1.0], [3.0]])
              },
              tfgnn.NodeSet: {
                  "feature": tf.convert_to_tensor([[5.0], [4.0], [6.0]])
              },
              tfgnn.EdgeSet: {
                  "feature": tf.convert_to_tensor([[8.0], [7.0], [9.0]])
              },
          },
      },
      {
          "corruptor": layers.ShuffleFeaturesGlobally(
              corruption_spec=layers.CorruptionSpec().with_default_rate(1.0),
              seed=8191,
          ),
          "context": tfgnn.Context.from_fields(
              features={
                  "feature": tf.convert_to_tensor([[1.0], [2.0], [3.0]]),
              },
          ),
          "node_set": tfgnn.NodeSet.from_fields(
              sizes=tf.convert_to_tensor([2, 1]),
              features={
                  "feature": tf.convert_to_tensor([[4.0], [5.0], [6.0]]),
              },
          ),
          "edge_set": tfgnn.EdgeSet.from_fields(
              sizes=tf.convert_to_tensor([2, 3]),
              adjacency=tfgnn.HyperAdjacency.from_indices({
                  tfgnn.SOURCE: ("node", tf.convert_to_tensor([0, 0, 0, 2, 2])),
                  tfgnn.TARGET: ("node", tf.convert_to_tensor([2, 1, 0, 0, 0])),
              }),
              features={
                  "feature": tf.convert_to_tensor([[7.0], [8.0], [9.0]]),
              },
          ),
          "expected_fields": {
              tfgnn.Context: {
                  "feature": tf.convert_to_tensor([[2.0], [1.0], [3.0]])
              },
              tfgnn.NodeSet: {
                  "feature": tf.convert_to_tensor([[5.0], [4.0], [6.0]])
              },
              tfgnn.EdgeSet: {
                  "feature": tf.convert_to_tensor([[8.0], [7.0], [9.0]])
              },
          },
      },
  ])
  def test_shuffle_features_globally(
      self,
      corruptor: layers._Corruptor,
      context: tfgnn.Context,
      node_set: tfgnn.NodeSet,
      edge_set: tfgnn.EdgeSet,
      expected_fields: Mapping[
          Union[tfgnn.Context, tfgnn.NodeSet, tfgnn.EdgeSet],
          Mapping[str, tfgnn.Field],
      ],
  ):
    graph = tfgnn.GraphTensor.from_pieces(
        context, {"node": node_set}, {"edge": edge_set}
    )
    shuffled = corruptor(graph)

    for fname, expected in expected_fields.get(tfgnn.Context, {}).items():
      self.assertAllEqual(expected, shuffled.context.features[fname])

    for fname, expected in expected_fields.get(tfgnn.NodeSet, {}).items():
      self.assertAllEqual(expected, shuffled.node_sets["node"].features[fname])

    for fname, expected in expected_fields.get(tfgnn.EdgeSet, {}).items():
      self.assertAllEqual(expected, shuffled.edge_sets["edge"].features[fname])

  @parameterized.parameters([
      {
          "corruptor": layers.DropoutFeatures(
              corruption_spec=layers.CorruptionSpec(
                  node_set_corruption={"*": {"*": 0.999}},
                  edge_set_corruption={"*": {"*": 0.999}},
                  context_corruption={"*": 0.999},
              ),
              seed=8191,
          ),
          "context": tfgnn.Context.from_fields(
              features={
                  "feature": tf.convert_to_tensor([[1.0], [2.0], [3.0]]),
              },
          ),
          "node_set": tfgnn.NodeSet.from_fields(
              sizes=tf.convert_to_tensor([2, 1]),
              features={
                  "feature": tf.convert_to_tensor([[4.0], [5.0], [6.0]]),
              },
          ),
          "edge_set": tfgnn.EdgeSet.from_fields(
              sizes=tf.convert_to_tensor([2, 3]),
              adjacency=tfgnn.HyperAdjacency.from_indices({
                  tfgnn.SOURCE: ("node", tf.convert_to_tensor([0, 0, 0, 2, 2])),
                  tfgnn.TARGET: ("node", tf.convert_to_tensor([2, 1, 0, 0, 0])),
              }),
              features={
                  "feature": tf.convert_to_tensor([[7.0], [8.0], [9.0]]),
              },
          ),
          "expected_fields": {
              tfgnn.Context: {
                  "feature": tf.convert_to_tensor([[0.0], [0.0], [0.0]])
              },
              tfgnn.NodeSet: {
                  "feature": tf.convert_to_tensor([[0.0], [0.0], [0.0]])
              },
              tfgnn.EdgeSet: {
                  "feature": tf.convert_to_tensor([[0.0], [0.0], [0.0]])
              },
          },
      },
  ])
  def test_dropout_features(
      self,
      corruptor: layers._Corruptor,
      context: tfgnn.Context,
      node_set: tfgnn.NodeSet,
      edge_set: tfgnn.EdgeSet,
      expected_fields: Mapping[
          Union[tfgnn.Context, tfgnn.NodeSet, tfgnn.EdgeSet],
          Mapping[str, tfgnn.Field],
      ],
  ):
    graph = tfgnn.GraphTensor.from_pieces(
        context, {"node": node_set}, {"edge": edge_set}
    )
    shuffled = corruptor(graph)

    for fname, expected in expected_fields.get(tfgnn.Context, {}).items():
      self.assertAllEqual(expected, shuffled.context.features[fname])

    for fname, expected in expected_fields.get(tfgnn.NodeSet, {}).items():
      self.assertAllEqual(expected, shuffled.node_sets["node"].features[fname])

    for fname, expected in expected_fields.get(tfgnn.EdgeSet, {}).items():
      self.assertAllEqual(expected, shuffled.edge_sets["edge"].features[fname])

  def test_throws_empty_spec_error(self):
    with self.assertRaisesRegex(ValueError, r"At least one of .*"):
      _ = layers._Corruptor(corruption_fn=lambda: None)

  @parameterized.named_parameters([
      dict(
          testcase_name="InvalidSequenceInput",
          inputs=[tf.constant(range(8))],
          expected_error=r"Expected `TensorShape` \(got .*\)",
      ),
      dict(
          testcase_name="UndefinedInnerDimension",
          inputs=tf.keras.Input((2, None)),
          expected_error=r"Expected a defined inner dimension \(got .*\)",
      ),
  ])
  def test_dgi_logits_value_error(
      self,
      inputs: Any,
      expected_error: str):
    """Verifies invalid input raises `ValueError`."""
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = layers.DeepGraphInfomaxLogits()(inputs)

  def test_deep_graph_infomax(self):
    """Verifies output logits shape."""
    x_clean = tf.random.uniform((1, 4))
    x_corrupted = tf.random.uniform((1, 4))
    logits = layers.DeepGraphInfomaxLogits()(
        tf.stack((x_clean, x_corrupted), axis=1)
    )

    # Clean and corrupted logits (i.e., shape (1, 2)).
    self.assertEqual(logits.shape, (1, 2))

if __name__ == "__main__":
  tf.test.main()
