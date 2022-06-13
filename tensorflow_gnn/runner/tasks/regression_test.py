"""Tests for node regression."""
from typing import Sequence

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import regression

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant

SCHEMA = """
context {
  features {
    key: "label"
    value: {
      description: "graph-label."
      dtype: DT_FLOAT
      shape { dim { size: 1 } }
    }
  }
}
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


class Regression(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          description="GraphMeanAbsoluteError",
          schema=SCHEMA,
          task=regression.GraphMeanAbsoluteError(node_set_name="nodes"),
          y_true=[[.8191]],
          expected_y_pred=[[0.06039321]],
          expected_loss=[0.7587068]),
      dict(
          description="GraphMeanAbsolutePercentageError",
          schema=SCHEMA,
          task=regression.GraphMeanAbsolutePercentageError(
              node_set_name="nodes"),
          y_true=[[.8191]],
          expected_y_pred=[[0.06039321]],
          expected_loss=[92.626884]),
      dict(
          description="GraphMeanSquaredError",
          schema=SCHEMA,
          task=regression.GraphMeanSquaredError(node_set_name="nodes"),
          y_true=[[.8191]],
          expected_y_pred=[[0.06039321]],
          expected_loss=[0.575636]),
      dict(
          description="GraphMeanSquaredLogarithmicError",
          schema=SCHEMA,
          task=regression.GraphMeanSquaredLogarithmicError(
              node_set_name="nodes",
              units=3),
          y_true=[[-0.407, -0.8191, 0.1634]],
          expected_y_pred=[[-0.4915728, -0.6728454, -0.8126122]],
          expected_loss=[0.00763526]),
      dict(
          description="GraphMeanSquaredLogScaledError",
          schema=SCHEMA,
          task=regression.GraphMeanSquaredLogScaledError(
              node_set_name="nodes",
              units=2),
          y_true=[[0.8208, 0.9]],
          expected_y_pred=[[0.74617755, -0.8193765]],
          expected_loss=[839.05788329]),
      dict(
          description="RootNodeMeanAbsoluteError",
          schema=SCHEMA,
          task=regression.RootNodeMeanAbsoluteError(node_set_name="nodes"),
          y_true=[[.8191]],
          expected_y_pred=[[-0.01075031]],
          expected_loss=[0.8298503]),
      dict(
          description="RootNodeMeanAbsolutePercentageError",
          schema=SCHEMA,
          task=regression.RootNodeMeanAbsolutePercentageError(
              node_set_name="nodes"),
          y_true=[[.8191]],
          expected_y_pred=[[-0.01075031]],
          expected_loss=[101.31245]),
      dict(
          description="RootNodeMeanSquaredError",
          schema=SCHEMA,
          task=regression.RootNodeMeanSquaredError(node_set_name="nodes"),
          y_true=[[.8191]],
          expected_y_pred=[[-0.01075031]],
          expected_loss=[0.68865156]),
      dict(
          description="RootNodeMeanSquaredLogarithmicError",
          schema=SCHEMA,
          task=regression.RootNodeMeanSquaredLogarithmicError(
              node_set_name="nodes",
              units=3),
          y_true=[[-0.407, -0.8191, 0.1634]],
          expected_y_pred=[[-0.24064127, -0.6996877, -0.6812314]],
          expected_loss=[0.00763526]),
      dict(
          description="RootNodeMeanSquaredLogScaledError",
          schema=SCHEMA,
          task=regression.RootNodeMeanSquaredLogScaledError(
              node_set_name="nodes",
              units=2),
          y_true=[[0.8208, 0.9]],
          expected_y_pred=[[0.7054771, -1.0065091]],
          expected_loss=[839.09634471]),
  ])
  def test_adapt(self,
                 description: str,
                 schema: str,
                 task: regression._Regression,
                 y_true: Sequence[float],
                 expected_y_pred: Sequence[float],
                 expected_loss: Sequence[float]):
    gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(SCHEMA))
    inputs = graph = tf.keras.layers.Input(type_spec=gtspec)

    values = graph.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    outputs = graph.replace_features(node_sets={
        "nodes": {
            tfgnn.HIDDEN_STATE: tf.keras.layers.Dense(8)(values)
        }
    })

    model = tf.keras.Model(inputs, outputs)
    model = task.adapt(model)

    self.assertIs(model.input, inputs)
    self.assertAllEqual(as_tensor(expected_y_pred).shape, model.output.shape)

    y_pred = model(tfgnn.random_graph_tensor(gtspec))
    self.assertAllClose(expected_y_pred, y_pred)

    loss = [loss_fn(y_true, y_pred) for loss_fn in task.losses()]
    self.assertAllClose(expected_loss, loss)

  @parameterized.named_parameters([
      dict(
          testcase_name="RootNodeMeanSquaredLogScaledError",
          klass=regression.RootNodeMeanSquaredLogScaledError,
      ),
      dict(
          testcase_name="RootNodeMeanSquaredLogarithmicError",
          klass=regression.RootNodeMeanSquaredLogarithmicError,
      ),
      dict(
          testcase_name="RootNodeMeanSquaredError",
          klass=regression.RootNodeMeanSquaredError,
      ),
      dict(
          testcase_name="RootNodeMeanAbsolutePercentageError",
          klass=regression.RootNodeMeanAbsolutePercentageError,
      ),
      dict(
          testcase_name="RootNodeMeanAbsoluteError",
          klass=regression.RootNodeMeanAbsoluteError,
      ),
      dict(
          testcase_name="GraphMeanSquaredLogScaledError",
          klass=regression.GraphMeanSquaredLogScaledError,
      ),
      dict(
          testcase_name="GraphMeanSquaredLogarithmicError",
          klass=regression.GraphMeanSquaredLogarithmicError,
      ),
      dict(
          testcase_name="GraphMeanSquaredError",
          klass=regression.GraphMeanSquaredError,
      ),
      dict(
          testcase_name="GraphMeanAbsolutePercentageError",
          klass=regression.GraphMeanAbsolutePercentageError,
      ),
      dict(
          testcase_name="GraphMeanAbsoluteError",
          klass=regression.GraphMeanAbsoluteError,
      ),
  ])
  def test_protocol(self, klass: object):
    self.assertIsInstance(klass, orchestration.Task)


if __name__ == "__main__":
  tf.test.main()
