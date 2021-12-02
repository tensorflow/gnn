from absl.testing import parameterized
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph.keras.layers import gat_v2

ct = tf.constant
rt = tf.ragged.constant

# TODO(b/205960151): More tests.


class GATv2Test(tf.test.TestCase, parameterized.TestCase):

  def testGAT_single_head(self):
    """Tests that a single-headed GAT is correct given predefined weights."""
    gt_input = gt.GraphTensor.from_pieces(
        node_sets={
            "signals": gt.NodeSet.from_fields(
                sizes=[3],
                # Node states have dimension 4.
                # The first three dimensions one-hot encode the node_id i.
                # The fourth dimension holds a distinct payload value 2**i.
                features={const.DEFAULT_STATE_NAME: ct(
                    [[1., 0., 0., 1.],
                     [0., 1., 0., 2.],
                     [0., 0., 1., 4.]])}),
        },
        edge_sets={
            "edges": gt.EdgeSet.from_fields(
                # The edges contain a cycle 0->1->2->0 (let's call it clockwise)
                # and the reverse cycle 0->2->1->0 (counterclockwise).
                sizes=[6],
                adjacency=adj.Adjacency.from_indices(
                    ("signals", ct([0, 1, 2, 0, 2, 1])),
                    ("signals", ct([1, 2, 0, 2, 1, 0])))),
        })

    log10 = tf.math.log(10.).numpy()
    obj = gat_v2.GATv2(
        num_heads=1,
        per_head_channels=4,
        edge_set_name="edges",
        attention_activation="relu")  # Let's keep it simple.

    _ = obj(gt_input)  # Build weights.
    weights = {v.name: v for v in obj.trainable_weights}
    self.assertLen(weights, 5)
    weights["gat_v2/node_set_update/gat_v2_convolution/query/kernel:0"].assign(
        # The space of attention computation of the single head has dimension 4.
        # The last dimension is used only in the key to carry the node's value,
        # multiplied by 11/10.
        # The first three dimensions are used to hand-construct attention scores
        # (see the running example below) that favor the counterclockwise
        # incoming edge over the other. Recall that weight matrices are
        # multiplied from the right onto batched inputs (in rows).
        #
        # For example, the query vector of node 0 is [0, 1, 0, 0], and ...
        [[0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [1., 0., 0., 0.],
         [0., 0., 0., 0.]])
    weights[
        "gat_v2/node_set_update/gat_v2_convolution/value_node/kernel:0"
    ].assign(
        # ... the key vectors of node 1 and 2, resp., are [-1, 0, -1, 2.2]
        # and [-1, -1, 0, 4.4]. Therefore, ...
        [[0., -1., -1., 0.],
         [-1., 0., -1., 0.],
         [-1., -1., 0., 0.],
         [0., 0., 0., 1.1]])
    weights[
        "gat_v2/node_set_update/gat_v2_convolution/attn_logits/kernel:0"
    ].assign(
        # ... attention from node 0 to node 1 has a sum of key and query vector
        # [-1, 1, -1, 2.2], which gets turned by ReLU and the attention weights
        # below into a pre-softmax score of log(10). Likewise,
        # attention from node 0 to node 2 has a vector sum [-1, 0, 0, 4.4]
        # and pre-softmax score of 0. Altogether, this means: ...
        [[log10], [log10], [log10], [0.]])
    weights["gat_v2/node_set_update/gat_v2_convolution/query/bias:0"].assign(
        [0., 0., 0., 0.])
    weights[
        "gat_v2/node_set_update/gat_v2_convolution/value_node/bias:0"
    ].assign(
        [0., 0., 0., 0.])

    got_gt = obj(gt_input)
    got = got_gt.node_sets["signals"][const.DEFAULT_STATE_NAME]

    # ... The softmax-weighed key vectors on the incoming edges of node 0
    # are  10/11 * [-1, 0, -1, 2.2]  +  1/11 * [-1, -1, 0, 4.4].
    # The final ReLU takes out the non-positive components and leaves 2 + 0.4
    # in the last component of the first row in the resulting node states.
    want = ct([[0., 0., 0., 2.4],  # Node 0.
               [0., 0., 0., 4.1],  # Node 1.
               [0., 0., 0., 1.2]])  # Node 2.
    self.assertAllEqual(got.shape, (3, 4))
    self.assertAllClose(got, want, atol=.0001)

  def testGAT_multihead(self):
    """Tests that a multi-headed GAT is correct given predefined weights."""
    gt_input = gt.GraphTensor.from_pieces(
        node_sets={
            "signals": gt.NodeSet.from_fields(
                sizes=[3],
                # The same node states as in the single_head test above.
                features={const.DEFAULT_STATE_NAME: ct(
                    [[1., 0., 0., 1.],
                     [0., 1., 0., 2.],
                     [0., 0., 1., 4.]])}),
        },
        edge_sets={
            "edges": gt.EdgeSet.from_fields(
                # The same edges as in the single_head test above.
                sizes=[6],
                adjacency=adj.Adjacency.from_indices(
                    ("signals", ct([0, 1, 2, 0, 2, 1])),
                    ("signals", ct([1, 2, 0, 2, 1, 0])))),
        })

    log10 = tf.math.log(10.).numpy()
    obj = gat_v2.GATv2(
        num_heads=2,
        per_head_channels=4,
        edge_set_name="edges",
        attention_activation=tf.keras.layers.LeakyReLU(alpha=0.0))

    _ = obj(gt_input)  # Build weights.
    weights = {v.name: v for v in obj.trainable_weights}
    self.assertLen(weights, 5)

    weights["gat_v2/node_set_update/gat_v2_convolution/query/kernel:0"].assign(
        # Attention head 0 uses the first four dimensions, which are used
        # in the same way as for the single_head test above.
        # Attention head 1 uses the last four dimensions, in which we
        # now favor the clockwise incoming edges and omit the scaling by 11/10.
        [[0., 1., 0., 0., 0., 0., 1., 0,],
         [0., 0., 1., 0., 1., 0., 0., 0.],
         [1., 0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.]])
    weights[
        "gat_v2/node_set_update/gat_v2_convolution/value_node/kernel:0"
    ].assign(
        [[0., -1., -1., 0., 0., -1., -1., 0.],
         [-1., 0., -1., 0., -1., 0., -1., 0.],
         [-1., -1., 0., 0., -1., -1., 0., 0.],
         [0., 0., 0., 1.1, 0., 0., 0., 1.]])
    weights[
        "gat_v2/node_set_update/gat_v2_convolution/attn_logits/kernel:0"
    ].assign(
        # Attention head 0 works out to softmax weights 10/11 and 1/11 as above.
        # Attention head 1 creates very large pre-softmax scores that
        # work out to weights 1 and 0 within floating-point precision.
        [[log10, 100.],
         [log10, 100.],
         [log10, 100.],
         [0., 0.]])
    weights[
        "gat_v2/node_set_update/gat_v2_convolution/query/bias:0"
    ].assign(
        [0., 0., 0., 0., 0., 0., 0., 0.])
    weights[
        "gat_v2/node_set_update/gat_v2_convolution/value_node/bias:0"
    ].assign(
        [0., 0., 0., 0., 0., 0., 0., 0.])

    got_gt = obj(gt_input)
    got = got_gt.node_sets["signals"][const.DEFAULT_STATE_NAME]

    # Attention head 0 generates the first four outout dimensions as in the
    # single_head test above, with weights 10/11 and 1/11,
    # Attention head 1 uses weights 0 and 1 (note the reversed preference).
    want = ct([[0., 0., 0., 2.4, 0., 0., 0., 4.0],
               [0., 0., 0., 4.1, 0., 0., 0., 1.0],
               [0., 0., 0., 1.2, 0., 0., 0., 2.0]])
    self.assertAllEqual(got.shape, (3, 8))
    self.assertAllClose(got, want, atol=.0001)


if __name__ == "__main__":
  tf.test.main()
