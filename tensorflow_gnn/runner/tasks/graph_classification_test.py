"""Tests for graph classification."""
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner.tasks import graph_classification


class GraphMulticlassClassification(tf.test.TestCase):

  gt = tfgnn.GraphTensor.from_pieces(
      node_sets={
          "node":
              tfgnn.NodeSet.from_fields(
                  features={
                      tfgnn.DEFAULT_STATE_NAME:
                          tf.constant((8191., 370., 153., 9474., 54748),
                                      dtype=tf.float32),
                  },
                  sizes=tf.constant((3, 2), dtype=tf.int32),
              )
      },
      context=tfgnn.Context.from_fields(
          features={"labels": tf.constant([3], dtype=tf.int32)}))

  task = graph_classification.GraphMulticlassClassification(
      4,
      node_set_name="node",
      state_name=tfgnn.DEFAULT_STATE_NAME)

  def test_adapt(self):
    inputs = tf.keras.layers.Input(type_spec=self.gt.spec)
    h = tf.keras.layers.Dense(64)(
        inputs.node_sets["node"][tfgnn.DEFAULT_STATE_NAME][:, None])
    output = inputs.replace_features(
        node_sets={"node": {
            tfgnn.DEFAULT_STATE_NAME: h
        }})
    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    model = self.task.adapt(model)

    self.assertIs(model.input, inputs)

    for output in model.output:
      self.assertAllEqual(output.shape, (4,))

    self.assertIsInstance(model.layers[-1], tf.keras.layers.Dense)
    self.assertIs(model.layers[-1].activation, tf.keras.activations.linear)


if __name__ == "__main__":
  tf.test.main()
