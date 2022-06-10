"""Tests for classification."""
from typing import Sequence

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import classification

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant

SCHEMA = """
node_sets {
  key: "nodes"
  value {
    features {
      key: "%s"
      value {
        dtype: DT_FLOAT
        shape { dim { size: 4 } }
      }
    }
  }
}
edge_sets {
  key: "edges"
  value {
    source: "nodes"
    target: "nodes"
  }
}
""" % tfgnn.HIDDEN_STATE


class Classification(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphBinaryClassification",
          schema=SCHEMA,
          task=classification.GraphBinaryClassification(node_set_name="nodes"),
          y_true=[[1]],
          expected_y_pred=[[-0.4159315]],
          expected_loss=[0.9225837]),
      dict(
          testcase_name="GraphMulticlassClassification",
          schema=SCHEMA,
          task=classification.GraphMulticlassClassification(
              4,
              node_set_name="nodes"),
          y_true=[3],
          expected_y_pred=[[0.35868323, -0.4112632, -0.23154753, 0.20909603]],
          expected_loss=[1.2067872]),
      dict(
          testcase_name="RootNodeBinaryClassification",
          schema=SCHEMA,
          task=classification.RootNodeBinaryClassification(
              node_set_name="nodes"),
          y_true=[[1]],
          expected_y_pred=[[-0.3450081]],
          expected_loss=[0.8804569]),
      dict(
          testcase_name="RootNodeMulticlassClassification",
          schema=SCHEMA,
          task=classification.RootNodeMulticlassClassification(
              3,
              node_set_name="nodes"),
          y_true=[2],
          expected_y_pred=[[-0.4718209, 0.04619305, -0.5249821]],
          expected_loss=[1.3415444]),
  ])
  def test_adapt(self,
                 schema: str,
                 task: classification._Classification,
                 y_true: Sequence[float],
                 expected_y_pred: Sequence[float],
                 expected_loss: Sequence[float]):
    gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(schema))
    inputs = tf.keras.layers.Input(type_spec=gtspec)
    hidden_state = inputs.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    output = inputs.replace_features(
        node_sets={"nodes": {
            tfgnn.HIDDEN_STATE: tf.keras.layers.Dense(16)(hidden_state)
        }})
    model = tf.keras.Model(inputs, output)
    model = task.adapt(model)

    self.assertIs(model.input, inputs)
    self.assertAllEqual(as_tensor(expected_y_pred).shape, model.output.shape)

    y_pred = model(tfgnn.random_graph_tensor(gtspec))
    self.assertAllClose(expected_y_pred, y_pred)

    loss = [loss_fn(y_true, y_pred) for loss_fn in task.losses()]
    self.assertAllClose(expected_loss, loss)

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphBinaryClassification",
          schema=SCHEMA,
          task=classification.GraphBinaryClassification(node_set_name="nodes")),
      dict(
          testcase_name="GraphMulticlassClassification",
          schema=SCHEMA,
          task=classification.GraphMulticlassClassification(
              4,
              node_set_name="nodes")),
      dict(
          testcase_name="RootNodeBinaryClassification",
          schema=SCHEMA,
          task=classification.RootNodeBinaryClassification(
              node_set_name="nodes")),
      dict(
          testcase_name="RootNodeMulticlassClassification",
          schema=SCHEMA,
          task=classification.RootNodeMulticlassClassification(
              3,
              node_set_name="nodes")),
  ])
  def test_fit(self,
               schema: str,
               task: classification._Classification):
    gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(schema))
    inputs = tf.keras.layers.Input(type_spec=gtspec)
    hidden_state = inputs.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    output = inputs.replace_features(
        node_sets={"nodes": {
            tfgnn.HIDDEN_STATE: tf.keras.layers.Dense(16)(hidden_state)
        }})
    model = tf.keras.Model(inputs, output)
    model = task.adapt(model)

    examples = tf.data.Dataset.from_tensors(tfgnn.random_graph_tensor(gtspec))
    labels = tf.data.Dataset.from_tensors([1.])

    dataset = tf.data.Dataset.zip((examples.repeat(2), labels.repeat(2)))

    model.compile(loss=task.losses(), metrics=task.metrics(), run_eagerly=True)
    model.fit(dataset)

  @parameterized.named_parameters([
      dict(
          testcase_name="RootNodeMulticlassClassification",
          klass=classification.RootNodeMulticlassClassification,
      ),
      dict(
          testcase_name="RootNodeBinaryClassification",
          klass=classification.RootNodeBinaryClassification,
      ),
      dict(
          testcase_name="GraphMulticlassClassification",
          klass=classification.GraphMulticlassClassification,
      ),
      dict(
          testcase_name="GraphBinaryClassification",
          klass=classification.GraphBinaryClassification,
      ),
  ])
  def test_protocol(self, klass: object):
    self.assertIsInstance(klass, orchestration.Task)


if __name__ == "__main__":
  tf.test.main()
