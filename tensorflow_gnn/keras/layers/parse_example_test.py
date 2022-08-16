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
"""Tests for gt.GraphTensor extension type (go/tf-gnn-api)."""

from absl.testing import parameterized
import google.protobuf.text_format as pbtext
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.keras.layers import parse_example


class ParseExampleTest(tf.test.TestCase, parameterized.TestCase):
  """Tests the wrapping of tfgnn.parse_example() in a Keras layer."""

  def pbtxt_to_dataset(self, examples_pbtxt) -> tf.data.Dataset:
    serialized = []
    for example_pbtxt in examples_pbtxt:
      serialized.append(
          pbtext.Merge(example_pbtxt, tf.train.Example()).SerializeToString())
    return tf.data.Dataset.from_tensor_slices(serialized)

  UNBATCHED_SPEC = gt.GraphTensorSpec.from_piece_specs(
      node_sets_spec={
          "node": gt.NodeSetSpec.from_field_specs(
              features_spec={
                  "words": tf.RaggedTensorSpec(
                      shape=(None, None), ragged_rank=1,
                      row_splits_dtype=tf.int32, dtype=tf.string)},
              sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32)),
      },
      edge_sets_spec={
          "edge": gt.EdgeSetSpec.from_field_specs(
              features_spec={
                  "weight": tf.TensorSpec(shape=(None,), dtype=tf.float32)},
              sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32),
              adjacency_spec=adj.AdjacencySpec.from_incident_node_sets(
                  source_node_set="node", target_node_set="node",
                  index_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32)))
      })

  EXAMPLES_PBTXT = [
      r"""
        features {
          feature {key: "nodes/node.#size" value {int64_list {value: [3]} } }
          feature {key: "nodes/node.words" value {bytes_list {value: ["a", "b", "c"]} } }
          feature {key: "nodes/node.words.d1" value {int64_list {value: [2, 0, 1]} } }
          feature {key: "edges/edge.#size" value {int64_list {value: [5]} } }
          feature {key: "edges/edge.#source" value {int64_list {value: [0, 1, 2, 2, 2]} } }
          feature {key: "edges/edge.#target" value {int64_list {value: [2, 1, 0, 0, 0]} } }
          feature {key: "edges/edge.weight" value {float_list {value: [1., 2., 3., 4., 5.]} } }
        }""", r"""
        features {
          feature {key: "nodes/node.#size" value {int64_list {value: [1]} } }
          feature {key: "nodes/node.words" value {bytes_list {value: ["e", "f"]} } }
          feature {key: "nodes/node.words.d1" value {int64_list {value: [2]} } }
        }"""]

  @parameterized.named_parameters(
      ("", False),
      ("DropRemainder", True))
  def testParseExample(self, drop_remainder):
    batch_size = len(self.EXAMPLES_PBTXT)
    ds = self.pbtxt_to_dataset(self.EXAMPLES_PBTXT)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    model = tf.keras.Sequential([
        parse_example.ParseExample(self.UNBATCHED_SPEC)])
    ds = ds.map(model)

    batched_spec = self.UNBATCHED_SPEC._batch(batch_size if drop_remainder
                                              else None)
    self.assertAllEqual(ds.element_spec, batched_spec)

    graph = ds.get_single_element()
    nodes = graph.node_sets["node"]
    edges = graph.edge_sets["edge"]
    self.assertAllEqual(nodes.sizes, tf.convert_to_tensor([[3], [1]]))
    self.assertAllEqual(nodes["words"], tf.ragged.constant(
        [[[b"a", b"b"], [], [b"c"]], [[b"e", b"f"]]],
        ragged_rank=2, dtype=tf.string, row_splits_dtype=tf.int32))
    self.assertAllEqual(edges.sizes, tf.convert_to_tensor([[5], [0]]))
    self.assertAllEqual(edges.adjacency.source,
                        tf.ragged.constant([[0, 1, 2, 2, 2], []]))
    self.assertAllEqual(edges.adjacency.target,
                        tf.ragged.constant([[2, 1, 0, 0, 0], []]))
    self.assertAllEqual(edges["weight"],
                        tf.ragged.constant([[1., 2., 3., 4., 5.], []]))

  def testParseSingleExample(self):
    ds = self.pbtxt_to_dataset(self.EXAMPLES_PBTXT)

    model = tf.keras.Sequential([
        parse_example.ParseSingleExample(self.UNBATCHED_SPEC)])
    ds = ds.map(model)

    self.assertAllEqual(ds.element_spec, self.UNBATCHED_SPEC)

    graphs = list(ds)
    self.assertLen(graphs, 2)

    nodes = graphs[0].node_sets["node"]
    edges = graphs[0].edge_sets["edge"]
    self.assertAllEqual(nodes.sizes, tf.convert_to_tensor([3]))
    self.assertAllEqual(nodes["words"], tf.ragged.constant(
        [[b"a", b"b"], [], [b"c"]],
        ragged_rank=1, dtype=tf.string, row_splits_dtype=tf.int32))
    self.assertAllEqual(edges.sizes, tf.convert_to_tensor([5]))
    self.assertAllEqual(edges.adjacency.source,
                        tf.convert_to_tensor([0, 1, 2, 2, 2]))
    self.assertAllEqual(edges.adjacency.target,
                        tf.convert_to_tensor([2, 1, 0, 0, 0]))
    self.assertAllEqual(edges["weight"],
                        tf.convert_to_tensor([1., 2., 3., 4., 5.]))

    nodes = graphs[1].node_sets["node"]
    edges = graphs[1].edge_sets["edge"]
    self.assertAllEqual(nodes.sizes, tf.convert_to_tensor([1]))
    self.assertAllEqual(nodes["words"], tf.ragged.constant(
        [[b"e", b"f"]],
        ragged_rank=1, dtype=tf.string, row_splits_dtype=tf.int32))
    self.assertAllEqual(edges.sizes, tf.convert_to_tensor([0]))
    self.assertAllEqual(edges.adjacency.source, tf.convert_to_tensor([]))
    self.assertAllEqual(edges.adjacency.target, tf.convert_to_tensor([]))
    self.assertAllEqual(edges["weight"], tf.convert_to_tensor([]))


if __name__ == "__main__":
  tf.test.main()
