"""Tests for node classification."""
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner.tasks import node_classification


class RootNodeMulticlassClassification(tf.test.TestCase):

  gt = tfgnn.GraphTensor.from_pieces(
      node_sets={
          "node":
              tfgnn.NodeSet.from_fields(
                  features={
                      tfgnn.DEFAULT_STATE_NAME:
                          tf.constant((8191., 370., 153., 9474., 54748),
                                      dtype=tf.float32),
                      "l":
                          tf.constant((0, 1, 2, 3, 4), dtype=tf.int32),
                  },
                  sizes=tf.constant((3, 2), dtype=tf.int32),
              ),
      })

  task = node_classification.RootNodeMulticlassClassification(
      5,
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
      self.assertAllEqual(output.shape, (5,))

    self.assertIsInstance(model.layers[-1], tf.keras.layers.Dense)
    self.assertIs(model.layers[-1].activation, tf.keras.activations.linear)


class RootNodeBinaryClassification(tf.test.TestCase):

  gt = tfgnn.GraphTensor.from_pieces(
      node_sets={
          "node":
              tfgnn.NodeSet.from_fields(
                  features={
                      tfgnn.DEFAULT_STATE_NAME:
                          tf.constant((8191., 370., 153., 9474., 54748),
                                      dtype=tf.float32),
                      "l":
                          tf.constant((0, 1, 0, 1, 1), dtype=tf.int32),
                  },
                  sizes=tf.constant((5,), dtype=tf.int32),
              ),
      })

  task = node_classification.RootNodeBinaryClassification(
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
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model = self.task.adapt(model)

    self.assertIs(model.input, inputs)

    for output in model.output:
      self.assertAllEqual(output.shape, (1,))

    self.assertIsInstance(model.layers[-1], tf.keras.layers.Dense)
    self.assertIs(model.layers[-1].activation, tf.keras.activations.linear)

  def test_fit(self):
    inputs = tf.keras.layers.Input(type_spec=self.gt.spec)
    h = tf.keras.layers.Dense(64)(
        inputs.node_sets["node"][tfgnn.DEFAULT_STATE_NAME][:, None])
    output = inputs.replace_features(
        node_sets={"node": {
            tfgnn.DEFAULT_STATE_NAME: h
        }})
    model = tf.keras.Model(inputs=inputs, outputs=output)

    xs = tf.data.Dataset.from_tensors(self.gt).repeat(8)
    ys = tf.data.Dataset.from_tensors([1.]).repeat()
    ds = tf.data.Dataset.zip((xs, ys))
    ds = ds.batch(4).map(lambda x, y: (x.merge_batch_to_components(), y))

    model = self.task.adapt(model)
    model.compile(loss=self.task.losses(), metrics=self.task.metrics())
    model.fit(ds)


if __name__ == "__main__":
  tf.test.main()
