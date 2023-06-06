# pyformat: mode=yapf
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
"""Tests for hgt."""
import enum
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.hgt import layers


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""

  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


def _homogeneous_cycle_graph(node_state, edge_state=None):
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "nodes":
              tfgnn.NodeSet.from_fields(
                  sizes=[3], features={tfgnn.HIDDEN_STATE: node_state}),
      },
      edge_sets={
          "edges":
              tfgnn.EdgeSet.from_fields(
                  sizes=[6],
                  adjacency=tfgnn.Adjacency.from_indices(
                      ("nodes", tf.constant([0, 1, 2])),
                      ("nodes", tf.constant([1, 2, 0])),
                  ),
                  features=(None if edge_state is None else {
                      tfgnn.HIDDEN_STATE: edge_state
                  }),
              ),
      },
  )


def _heterogeneous_example_graph(add_readout=False):
  node_sets = {
      "airframe":
          tfgnn.NodeSet.from_fields(
              sizes=[4], features={tfgnn.HIDDEN_STATE: tf.eye(4, 3)}),
      "engine":
          tfgnn.NodeSet.from_fields(
              sizes=[3], features={tfgnn.HIDDEN_STATE: tf.eye(3, 2)}),
  }
  edge_sets = {
      "powerplant":
          tfgnn.EdgeSet.from_fields(
              sizes=[4],
              features={},
              adjacency=tfgnn.Adjacency.from_indices(
                  source=("airframe", [0, 1, 2, 3]),
                  target=("engine", [0, 1, 2, 1]),
              ),
          )
  }
  if add_readout:
    node_sets["_readout"] = tfgnn.NodeSet.from_fields(sizes=[1])
    edge_sets["_readout/seed"] = tfgnn.EdgeSet.from_fields(
        sizes=[1],
        adjacency=tfgnn.Adjacency.from_indices(
            source=("engine", [1]), target=("_readout", [0])),
    )
  return tfgnn.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)


def _parallel_vee_example_graph():
  """Returns a graph with two parallel V-shaped edge sets."""
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "senders":
              tfgnn.NodeSet.from_fields(
                  sizes=[2],
                  features={
                      tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0.0], [0.0, 1.0]])
                  },
              ),
          "receiver":
              tfgnn.NodeSet.from_fields(
                  sizes=[1],
                  features={tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0.0]])},
              ),
      },
      edge_sets={
          "link_a":
              tfgnn.EdgeSet.from_fields(
                  sizes=[2],
                  adjacency=tfgnn.Adjacency.from_indices(
                      source=("senders", [0, 1]),
                      target=("receiver", [0, 0]),
                  ),
              ),
          "link_b":
              tfgnn.EdgeSet.from_fields(
                  sizes=[2],
                  adjacency=tfgnn.Adjacency.from_indices(
                      source=("senders", [0, 1]),
                      target=("receiver", [0, 0]),
                  ),
              ),
      },
  )


class HgtConvTest(tf.test.TestCase, parameterized.TestCase):

  # TODO(b/269076334): Test with "_readout" node set.
  def test_ndim_input(self):
    """Tests that HGT can handle inputs with more than 2 dimensions."""
    test_graph = _homogeneous_cycle_graph(tf.zeros((3, 2, 3)))
    conv = layers.HGTGraphUpdate(
        num_heads=1,
        per_head_channels=3,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0,
    )
    got = conv(test_graph).node_sets["nodes"]["hidden_state"]
    self.assertAllClose(tf.zeros((3, 2, 3)), got)

  def test_latent_receiver(self):
    """Tests that HGT updates latent features in receiver node sets."""
    test_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "source":
                tfgnn.NodeSet.from_fields(
                    sizes=[3], features={tfgnn.HIDDEN_STATE: tf.zeros((3, 3))}),
            "target":
                tfgnn.NodeSet.from_fields(
                    # Feature depth 0 indicates a latent node set.
                    sizes=[3],
                    features={tfgnn.HIDDEN_STATE: tf.zeros((3, 0))},
                ),
        },
        edge_sets={
            "source->target":
                tfgnn.EdgeSet.from_fields(
                    sizes=[3],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("source", [0, 1, 2]),
                        target=("target", [0, 1, 2]),
                    ),
                ),
            "target->target":
                tfgnn.EdgeSet.from_fields(
                    sizes=[3],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("target", [0, 1, 2]),
                        target=("target", [1, 2, 0]),
                    ),
                ),
        },
    )
    conv = layers.HGTGraphUpdate(
        num_heads=1,
        per_head_channels=3,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0,
    )
    got = conv(test_graph).node_sets["target"]["hidden_state"]
    self.assertAllClose(tf.zeros((3, 3)), got)  # Feature depth is non-zero now.

  def test_wrong_size_receiver(self):
    """Tests that HGT throws an error with wrong-sized receiver features."""
    test_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "source":
                tfgnn.NodeSet.from_fields(
                    sizes=[3], features={tfgnn.HIDDEN_STATE: tf.zeros((3, 3))}),
            "target":
                tfgnn.NodeSet.from_fields(
                    sizes=[3], features={tfgnn.HIDDEN_STATE: tf.zeros((3, 1))}),
        },
        edge_sets={
            "source->target":
                tfgnn.EdgeSet.from_fields(
                    sizes=[3],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("source", [0, 1, 2]),
                        target=("target", [0, 1, 2]),
                    ),
                ),
            "target->target":
                tfgnn.EdgeSet.from_fields(
                    sizes=[3],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("target", [0, 1, 2]),
                        target=("target", [1, 2, 0]),
                    ),
                ),
        },
    )
    conv = layers.HGTGraphUpdate(
        num_heads=1,
        per_head_channels=3,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0,
    )
    self.assertRaisesRegex(
        ValueError,
        "The input features need to either be empty",
        lambda: conv(test_graph),
    )

  def test_homogeneous_multi_head(self):
    test_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes":
                tfgnn.NodeSet.from_fields(
                    sizes=[4], features={tfgnn.HIDDEN_STATE: tf.eye(4)}),
        },
        edge_sets={
            "edges":
                tfgnn.EdgeSet.from_fields(
                    sizes=[8],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("nodes", [0, 1, 2, 3, 0, 1, 2, 3]),
                        target=("nodes", [1, 2, 3, 0, 2, 3, 0, 1]),
                    ),
                )
        },
    )
    conv = layers.HGTGraphUpdate(
        num_heads=2,
        per_head_channels=2,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0,
        # Without weighted_skip, the inputs are simply added to the outputs.
        use_weighted_skip=False,
        use_layer_norm=False,
        activation="relu",
    )
    _ = conv(test_graph)
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 11)  # Includes biases and edge type priors.
    # Use identity transformations for the initial key/message/query states.
    weights["hgt_graph_update/key_node_nodes/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/message_node_nodes/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/query_node_nodes/kernel:0"].assign(tf.eye(4))
    # Use identity transformation for the final node transformation as well.
    weights["hgt_graph_update/aggr_node_nodes/kernel:0"].assign(tf.eye(4))
    # Each attention head should apply identity transformation
    # to its channels.
    weights["hgt_graph_update/attention_edge_edges/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    weights["hgt_graph_update/message_edge_edges/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    self.assertAllClose(
        conv(test_graph).node_sets["nodes"]["hidden_state"],
        # Each node has 2 incoming edges where the node feature is its
        # corresponding unit vector.
        # The first two entries of each unit vector get transformed by
        # att head 0, the other two are transformed by att head 1.
        # The results of each head get added together when computing scores,
        # then scaled by softmax to 0.5 each.
        # After pooling the attention results, the result is eye(4) combined
        # with 0.5 at every coordinate corresponding to a (target, source) pair.
        tf.constant([
            [1, 0, 0.5, 0.5],
            [0.5, 1, 0, 0.5],
            [0.5, 0.5, 1, 0],
            [0, 0.5, 0.5, 1],
        ]),
        rtol=1e-06,
    )

  def test_multi_senders_one_receiver_multi_head(self):
    log20 = tf.math.log(20.0).numpy()
    log80 = tf.math.log(80.0).numpy()
    sqrt2 = tf.math.sqrt(2.0).numpy()
    test_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            # Inputs from sender0 should be non-preferred (log(20))
            "sender0":
                tfgnn.NodeSet.from_fields(
                    sizes=[4],
                    features={tfgnn.HIDDEN_STATE: log20 * tf.eye(4)}),
            # Inputs from sender1 should be preferred (log(80))
            "sender1":
                tfgnn.NodeSet.from_fields(
                    sizes=[4],
                    features={tfgnn.HIDDEN_STATE: log80 * tf.eye(4)}),
            # Inputs from receiver are scaled by an inverse scaling factor
            # sqrt(2)
            "receiver":
                tfgnn.NodeSet.from_fields(
                    sizes=[4],
                    features={tfgnn.HIDDEN_STATE: sqrt2 * tf.eye(4)}),
        },
        edge_sets={
            "sender0->receiver":
                tfgnn.EdgeSet.from_fields(
                    sizes=[8],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("sender0", [0, 1, 2, 3]),
                        target=("receiver", [0, 1, 2, 3]),
                    ),
                ),
            "sender1->receiver":
                tfgnn.EdgeSet.from_fields(
                    sizes=[8],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("sender1", [0, 1, 2, 3]),
                        target=("receiver", [0, 1, 2, 3]),
                    ),
                ),
        },
    )
    conv = layers.HGTGraphUpdate(
        num_heads=2,
        per_head_channels=2,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0,
        use_layer_norm=False,
        activation="relu",
    )
    _ = conv(test_graph)
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 19)  # Includes biases and edge type priors.
    weights["hgt_graph_update/key_node_sender0/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/message_node_sender0/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/key_node_sender1/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/message_node_sender1/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/query_node_receiver/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/aggr_node_receiver/kernel:0"].assign(tf.eye(4))
    # Set skip_receiver so that sigmoid(skip_receiver) = 0.5
    # This means the weighted average is 0.5*input + 0.5*result
    weights["hgt_graph_update/skip_receiver:0"].assign(tf.constant(0.0))
    # Set attention weights to preserve their input values but reshape
    # them from (4,4) to (4,2,2).
    weights[
        "hgt_graph_update/attention_edge_sender0->receiver/kernel:0"].assign(
            tf.stack([tf.eye(2), tf.eye(2)]))
    weights["hgt_graph_update/message_edge_sender0->receiver/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    weights[
        "hgt_graph_update/attention_edge_sender1->receiver/kernel:0"].assign(
            tf.stack([tf.eye(2), tf.eye(2)]))
    weights["hgt_graph_update/message_edge_sender1->receiver/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    self.assertAllClose(
        conv(test_graph).node_sets["receiver"]["hidden_state"],
        # Softmax([log(20), log(80)]) is [0.2, 0.8]
        # After scaling the messages and pooling, the result is
        # log(20)*0.2 + log(80)*0.8 * eye(4).
        # The return value should be (1/2*result + 1/2*input)
        ((log20 * 0.2 + log80 * 0.8 + sqrt2) / 2) * tf.eye(4),
        rtol=1e-06,
    )

  def test_one_sender_multi_receivers_multi_head(self):
    test_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "sender":
                tfgnn.NodeSet.from_fields(
                    sizes=[4], features={tfgnn.HIDDEN_STATE: tf.eye(4)}),
            "receiver0":
                tfgnn.NodeSet.from_fields(
                    sizes=[4], features={tfgnn.HIDDEN_STATE: tf.eye(4)}),
            "receiver1":
                tfgnn.NodeSet.from_fields(
                    sizes=[4], features={tfgnn.HIDDEN_STATE: 2 * tf.eye(4)}),
        },
        edge_sets={
            "sender->receiver0":
                tfgnn.EdgeSet.from_fields(
                    sizes=[4],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("sender", [0, 1, 2, 3]),
                        target=("receiver0", [0, 1, 2, 3]),
                    ),
                ),
            "sender->receiver1":
                tfgnn.EdgeSet.from_fields(
                    sizes=[4],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("sender", [0, 1, 2, 3]),
                        target=("receiver1", [0, 1, 2, 3]),
                    ),
                ),
        },
    )
    conv = layers.HGTGraphUpdate(
        num_heads=2,
        per_head_channels=2,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0,
        use_weighted_skip=False,
        use_layer_norm=False,
        activation="relu",
    )
    _ = conv(test_graph)
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 18)  # Includes biases and edge type priors.
    weights["hgt_graph_update/key_node_sender/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/message_node_sender/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/query_node_receiver0/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/aggr_node_receiver0/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/query_node_receiver1/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/aggr_node_receiver1/kernel:0"].assign(tf.eye(4))
    weights[
        "hgt_graph_update/attention_edge_sender->receiver0/kernel:0"].assign(
            tf.stack([tf.eye(2), tf.eye(2)]))
    weights["hgt_graph_update/message_edge_sender->receiver0/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    weights[
        "hgt_graph_update/attention_edge_sender->receiver1/kernel:0"].assign(
            tf.stack([tf.eye(2), tf.eye(2)]))
    weights["hgt_graph_update/message_edge_sender->receiver1/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    self.assertAllClose(
        conv(test_graph).node_sets["receiver0"]["hidden_state"],
        2 * tf.eye(4),
        rtol=1e-06,
    )
    self.assertAllClose(
        conv(test_graph).node_sets["receiver1"]["hidden_state"],
        3 * tf.eye(4),
        rtol=1e-06,
    )

  def test_multi_senders_multi_receivers_multi_head(self):
    log20 = tf.math.log(20.0).numpy()
    log80 = tf.math.log(80.0).numpy()
    sqrt2 = tf.math.sqrt(2.0).numpy()
    test_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "sender0":
                tfgnn.NodeSet.from_fields(
                    sizes=[4],
                    features={tfgnn.HIDDEN_STATE: log20 * tf.eye(4)}),
            "sender1":
                tfgnn.NodeSet.from_fields(
                    sizes=[4],
                    features={tfgnn.HIDDEN_STATE: log80 * tf.eye(4)}),
            "receiver0":
                tfgnn.NodeSet.from_fields(
                    sizes=[4],
                    features={tfgnn.HIDDEN_STATE: sqrt2 * tf.eye(4)}),
            "receiver1":
                tfgnn.NodeSet.from_fields(
                    sizes=[4],
                    features={tfgnn.HIDDEN_STATE: sqrt2 * tf.eye(4)}),
        },
        edge_sets={
            "sender0->receiver0":
                tfgnn.EdgeSet.from_fields(
                    sizes=[4],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("sender0", [0, 1, 2, 3]),
                        target=("receiver0", [0, 1, 2, 3]),
                    ),
                ),
            "sender1->receiver0":
                tfgnn.EdgeSet.from_fields(
                    sizes=[4],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("sender1", [0, 1, 2, 3]),
                        target=("receiver0", [0, 1, 2, 3]),
                    ),
                ),
            "sender0->receiver1":
                tfgnn.EdgeSet.from_fields(
                    sizes=[4],
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("sender0", [0, 1, 2, 3]),
                        target=("receiver1", [0, 1, 2, 3]),
                    ),
                ),
        },
    )
    conv = layers.HGTGraphUpdate(
        num_heads=2,
        per_head_channels=2,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0,
        use_layer_norm=False,
        activation="relu",
    )
    _ = conv(test_graph)
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 27)  # Includes biases and edge type priors.
    weights["hgt_graph_update/key_node_sender0/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/message_node_sender0/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/key_node_sender1/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/message_node_sender1/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/query_node_receiver0/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/aggr_node_receiver0/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/skip_receiver0:0"].assign(tf.constant(0.0))
    weights["hgt_graph_update/query_node_receiver1/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/aggr_node_receiver1/kernel:0"].assign(tf.eye(4))
    weights["hgt_graph_update/skip_receiver1:0"].assign(tf.constant(0.0))
    weights[
        "hgt_graph_update/attention_edge_sender0->receiver0/kernel:0"].assign(
            tf.stack([tf.eye(2), tf.eye(2)]))
    weights["hgt_graph_update/message_edge_sender0->receiver0/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    weights[
        "hgt_graph_update/attention_edge_sender1->receiver0/kernel:0"].assign(
            tf.stack([tf.eye(2), tf.eye(2)]))
    weights["hgt_graph_update/message_edge_sender1->receiver0/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    weights[
        "hgt_graph_update/attention_edge_sender0->receiver1/kernel:0"].assign(
            tf.stack([tf.eye(2), tf.eye(2)]))
    weights["hgt_graph_update/message_edge_sender0->receiver1/kernel:0"].assign(
        tf.stack([tf.eye(2), tf.eye(2)]))
    self.assertAllClose(
        conv(test_graph).node_sets["receiver0"]["hidden_state"],
        ((log20 * 0.2 + log80 * 0.8 + sqrt2) / 2) * tf.eye(4),
        rtol=1e-06,
    )
    self.assertAllClose(
        conv(test_graph).node_sets["receiver1"]["hidden_state"],
        ((log20 + sqrt2) / 2) * tf.eye(4),
        rtol=1e-06,
    )

  def test_multi_edge_set_attention(self):
    """Tests uniform attention over 2 edge sets with edge-dependent weights."""
    conv = layers.HGTGraphUpdate(
        num_heads=1,
        per_head_channels=2,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0.0,
        use_layer_norm=False,
        use_weighted_skip=False,
        activation="relu",
    )
    input_graph = _parallel_vee_example_graph()
    _ = conv(input_graph)  # Trigger creation of weights.
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 14)  # See below, plus 4 bias terms left at zero.

    # The test graph has node sets "senders" (2 nodes) and "receiver" (1 node).
    # There are two edge sets, "link_a" and "link_b", each of which connects
    # both senders to the receiver.
    #
    # The initial node-states are one-hot encodings of the node id.
    # The receiver node is mapped trivially to the attention query [1, 0].
    weights["hgt_graph_update/query_node_receiver/kernel:0"].assign(
        np.array([[1.0, 0.0], [99.0, 99.0]]))
    # The sender nodes are mapped to the following messages.
    weights["hgt_graph_update/message_node_senders/kernel:0"].assign(
        np.array([[2.0, 5.0], [3.0, 7.0]]))
    # The message projections per edge set are set up such that
    # "link_a" uses only the first component, that is [2,0] and [3,0], resp.,
    # while "link_b" uses only the second component, [0,5] and [0,7].
    weights["hgt_graph_update/message_edge_link_a/kernel:0"].assign(
        np.array([[[1.0, 0.0], [0.0, 0.0]]]))  # Maps [x,y] to [x,0].
    weights["hgt_graph_update/message_edge_link_b/kernel:0"].assign(
        np.array([[[0.0, 0.0], [0.0, 1.0]]]))  # Maps [x,y] to [0,y].
    # Similarly, the sender nodes get attention keys from which we extract
    # the first component for "link_a" and the second component for "link_b" to
    # match up with the attention query [1, 0]. The keys are chosen such that
    # the attention weights will be  10*[2,0] + 1*[3,0] for "link_a" and
    # 2*[0,5] + 20*[0,7] for "link_b", divided by the sum of weights 33.
    weights["hgt_graph_update/key_node_senders/kernel:0"].assign(
        np.log([[10.0, 2.0], [1.0, 20.0]]))  # Softmax will undo the log().
    weights["hgt_graph_update/attention_edge_link_a/kernel:0"].assign(
        np.array([[[1.0, 0.0], [0.0, 0.0]]]))  # Maps [x,y] to [x,0].
    weights["hgt_graph_update/attention_edge_link_b/kernel:0"].assign(
        np.array([[[0.0, 1.0], [0.0, 0.0]]]))  # Maps [x,y] to [y,0].
    # HGT defines an additional scalar weight for use in dot-product similarity
    # of each edge set. We simply use that to cancel out the Transformer-style
    # division of scalar products by the square root of the dimension.
    weights["hgt_graph_update/priors_link_a:0"].assign(np.sqrt([2.0]))
    weights["hgt_graph_update/priors_link_b:0"].assign(np.sqrt([2.0]))
    # HGT applies a feed-forward transformation on attention results.
    # We use it to make the denominator of softmax more readable in decimal:
    # The original weight vectors are [10, 1] / 33 for "link_a" and
    # [2, 20] / 33 for "link_b"; they now become [1.0, 0.1] and 2*[0.1, 1.0],
    # which leads to the result [2.3, 2*7.5].
    # Notice how mapping [[2., 5.], [3., 7.]] to [2.3, 2*7.5] demonstrates
    # the use of one global softmax, but with reversed attention preference
    # between the two edge sets.
    weights["hgt_graph_update/aggr_node_receiver/kernel:0"].assign(
        np.array([[3.3, 0.0], [0.0, 3.3]]))
    # The expected final output is the result discussed above,
    # plus the residual link from the old receiver state [1, 0].
    expected = np.array([[1.0 + 2.3, 2 * 7.5]])

    graph = conv(input_graph)
    actual = graph.node_sets["receiver"][tfgnn.HIDDEN_STATE].numpy()
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      ("NoneInitializer", ReloadModel.SKIP, None),
      ("NoneInitializerRestored", ReloadModel.SAVED_MODEL, None),
      ("NoneInitializerRestoredKeras", ReloadModel.KERAS, None),
      ("StrInitializerRestored", ReloadModel.SAVED_MODEL, "glorot_uniform"),
      ("StrInitializerRestoredKeras", ReloadModel.KERAS, "glorot_uniform"),
      (
          "ObjInitializerRestored",
          ReloadModel.SAVED_MODEL,
          tf.keras.initializers.Constant(4.3),
      ),
      (
          "ObjInitializerRestoredKeras",
          ReloadModel.KERAS,
          tf.keras.initializers.Constant(4.3),
      ),
  )
  def test_hgtconv_saving(self, reload_model, kernel_initializer):
    # Build a Model around the Layer, possibly saved and restored.
    inputs = tf.keras.layers.Input(
        type_spec=_heterogeneous_example_graph().spec)
    layer = layers.HGTGraphUpdate(
        num_heads=2,
        per_head_channels=1,
        receiver_tag=tfgnn.TARGET,
        kernel_initializer=kernel_initializer,
        dropout_rate=0,
        use_layer_norm=False,
    )
    outputs = layer(inputs)
    layer_before_engine_state = layer(
        _heterogeneous_example_graph()).node_sets["engine"][tfgnn.HIDDEN_STATE]
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "hgt-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1), layers.HGTGraphUpdate)
      else:
        model = tf.saved_model.load(export_dir)

    got_gt = model(_heterogeneous_example_graph())
    got = got_gt.node_sets["engine"][tfgnn.HIDDEN_STATE]
    self.assertAllEqual(got.shape, (3, 2))
    # TODO(b/269492127) Re-enable these tests when Keras load issue is resolved.
    if reload_model == ReloadModel.KERAS and (isinstance(
        kernel_initializer, str) or kernel_initializer is None):
      self.skipTest("Bad Test: Known issue in Keras model reloading")
    self.assertAllEqual(got, layer_before_engine_state)

  @parameterized.named_parameters(("baseline", False), ("", True))
  def test_ignores_readout(self, add_readout):
    test_graph = _heterogeneous_example_graph(add_readout=add_readout)
    conv = layers.HGTGraphUpdate(
        num_heads=2,
        per_head_channels=1,
        receiver_tag=tfgnn.TARGET,
    )
    _ = conv(test_graph)
    # Adding "_readout" does not change the node and edge sets used.
    self.assertCountEqual(
        [v.name for v in conv.trainable_weights],
        [
            # Source node set "airframe".
            "hgt_graph_update/key_node_airframe/kernel:0",
            "hgt_graph_update/key_node_airframe/bias:0",
            "hgt_graph_update/message_node_airframe/kernel:0",
            "hgt_graph_update/message_node_airframe/bias:0",
            # Target node set "engine".
            "hgt_graph_update/query_node_engine/kernel:0",
            "hgt_graph_update/query_node_engine/bias:0",
            "hgt_graph_update/aggr_node_engine/kernel:0",
            "hgt_graph_update/aggr_node_engine/bias:0",
            "hgt_graph_update/skip_engine:0",
            "hgt_graph_update/normalization_engine/gamma:0",
            "hgt_graph_update/normalization_engine/beta:0",
            # Edge set "powerplant".
            "hgt_graph_update/message_edge_powerplant/kernel:0",
            "hgt_graph_update/attention_edge_powerplant/kernel:0",
            "hgt_graph_update/priors_powerplant:0",
        ],
    )


class HGTTFLiteTest(tf.test.TestCase, parameterized.TestCase):

  def testBasic(self):
    test_graph_1_dict = {
        # We care that the TFLite interpreter gives the same output as the
        # model, which was tested separately (although not for the randomly
        # initialized weights that we keep here).
        "airframe_features": tf.eye(4, 3),
        "engine_features": tf.eye(3, 2),
        "airframe_source": tf.constant([0, 1, 2, 3]),
        "engine_target": tf.constant([0, 1, 2, 1]),
    }

    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith("2.10."):
      self.skipTest("GNN models are unsupported in TFLite until TF 2.11 but "
                    f"got TF {tf.__version__}")
    layer = layers.HGTGraphUpdate(
        num_heads=2,
        per_head_channels=1,
        receiver_tag=tfgnn.TARGET,
        dropout_rate=0,
    )

    inputs = {
        "airframe_features":
            tf.keras.Input([3], None, "airframe_features", tf.float32),
        "engine_features":
            tf.keras.Input([2], None, "engine_features", tf.float32),
        "airframe_source":
            tf.keras.Input([], None, "airframe_source", tf.int32),
        "engine_target":
            tf.keras.Input([], None, "engine_target", tf.int32),
    }
    graph_in = _MakeGraphTensor()(inputs)
    graph_out = layer(graph_in)
    outputs = tf.keras.layers.Layer(name="final_engine_states")(
        graph_out.node_sets["engine"][tfgnn.HIDDEN_STATE])
    model = tf.keras.Model(inputs, outputs)

    # The other unit tests should verify that this is correct
    expected = model(test_graph_1_dict).numpy()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner("serving_default")
    obtained = signature_runner(**test_graph_1_dict)["final_engine_states"]
    self.assertAllClose(expected, obtained)


# TODO(b/274779989): Replace this layer with a more standard representation
# of GraphTensor as a dict of plain Tensors.
class _MakeGraphTensor(tf.keras.layers.Layer):
  """Makes a homogeneous GraphTensor of rank 0 with a single component."""

  def call(self, inputs):
    airframe_sizes = tf.shape(inputs["airframe_features"])[0]
    engine_sizes = tf.shape(inputs["engine_features"])[0]
    powerplant_sizes = tf.shape(inputs["airframe_source"])
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            "airframe":
                tfgnn.NodeSet.from_fields(
                    sizes=tf.expand_dims(airframe_sizes, axis=0),
                    features={tfgnn.HIDDEN_STATE: inputs["airframe_features"]},
                ),
            "engine":
                tfgnn.NodeSet.from_fields(
                    sizes=tf.expand_dims(engine_sizes, axis=0),
                    features={tfgnn.HIDDEN_STATE: inputs["engine_features"]},
                ),
        },
        edge_sets={
            "powerplant":
                tfgnn.EdgeSet.from_fields(
                    sizes=powerplant_sizes,
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("airframe", inputs["airframe_source"]),
                        target=("engine", inputs["engine_target"]),
                    ),
                )
        },
    )


if __name__ == "__main__":
  tf.test.main()
