"""Tests for convolutions."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph.keras.layers import convolutions


class SimpleConvolutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Forward", False, False),
      ("ForwardWithEdgeFeatre", True, False),
      ("Backward", False, True),
      ("BackwardWithEdgeFeatre", True, True))
  def test(self, include_edges=True, reverse=True):
    values = dict(edges=tf.constant([[1.], [2.]]),
                  nodes=tf.constant([[4.], [8.], [16.]]))
    input_graph = _make_test_graph_01into2(values)
    message_fn = tf.keras.layers.Dense(1, use_bias=False,
                                       kernel_initializer="ones")
    conv = convolutions.SimpleConvolution(
        message_fn,
        node_input_tags=[const.TARGET] if reverse else [const.SOURCE],
        edge_input_feature=const.DEFAULT_STATE_NAME if include_edges else None,
        destination_tag=const.SOURCE if reverse else const.TARGET)

    actual = conv(input_graph, edge_set_name="edges")
    if reverse:
      expected = tf.constant([
          [16. + include_edges*1.],
          [16. + include_edges*2.],
          [0.]])  # No outgoing edge.
    else:
      expected = tf.constant([
          [0.], [0.],  # No incoming edges,
          [(4. + 8.) + include_edges*(1. + 2.)]])
    self.assertAllEqual(expected, actual)


def _make_test_graph_01into2(values):
  """Returns GraphTensor for [v0] --e0--> [v2] <-e1-- [v1] with values."""
  def maybe_features(key):
    features = {const.DEFAULT_STATE_NAME: values[key]} if key in values else {}
    return dict(features=features)
  graph = gt.GraphTensor.from_pieces(
      context=gt.Context.from_fields(**maybe_features("context")),
      node_sets={"nodes": gt.NodeSet.from_fields(
          sizes=tf.constant([3]), **maybe_features("nodes"))},
      edge_sets={"edges": gt.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=adj.Adjacency.from_indices(("nodes", tf.constant([0, 1])),
                                               ("nodes", tf.constant([2, 2]))),
          **maybe_features("edges"))})
  return graph


if __name__ == "__main__":
  tf.test.main()
