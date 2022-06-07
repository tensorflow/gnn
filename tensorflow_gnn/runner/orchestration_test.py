"""Tests for orchestraion."""
import os
from typing import Mapping, Sequence, Union

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import vanilla_mpnn
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import classification
from tensorflow_gnn.runner.tasks import dgi
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import model_export
from tensorflow_gnn.runner.utils import model_templates

SCHEMA = """
  context {
    features {
      key: "label"
      value {
        dtype: DT_INT32
      }
    }
  }
  node_sets {
    key: "node"
    value {
      features {
        key: "features"
        value {
          dtype: DT_FLOAT
          shape { dim { size: 4 } }
        }
      }
    }
  }
  edge_sets {
    key: "edge"
    value {
      source: "node"
      target: "node"
    }
  }
"""

Task = orchestration.Task


class DatasetProvider:

  def __init__(self, examples: Sequence[bytes]):
    self._examples = list(examples)

  def get_dataset(self, _: tf.distribute.InputContext) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(self._examples)


class OrchestrationTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="Task",
          task=dgi.DeepGraphInfomax(node_set_name="node"),
          task_weights=None,
          output_names="o1",
          output_shapes={
              "o1": (2,)  # Positive and negative logits
          }),
      dict(
          testcase_name="SequenceTask",
          task=[
              dgi.DeepGraphInfomax(node_set_name="node"),
              classification.RootNodeMulticlassClassification(
                  node_set_name="node",
                  num_classes=10)  # Ten classes
          ],
          task_weights=None,
          output_names=["o1", "o2"],
          output_shapes={
              "o1": (2,),  # Positive and negative logits
              "o2": (10,)  # Ten classes
          }),
      dict(
          testcase_name="MappingTask",
          task={
              "dgi": dgi.DeepGraphInfomax(node_set_name="node"),
              "multiclass": classification.RootNodeMulticlassClassification(
                  node_set_name="node",
                  num_classes=10)  # Ten classes
          },
          task_weights=None,
          output_names={"dgi": "o1", "multiclass": "o2"},
          output_shapes={
              "o1": (2,),  # Positive and negative logits
              "o2": (10,)  # Ten classes
          }),
      dict(
          testcase_name="TaskWithWeights",
          task=dgi.DeepGraphInfomax(node_set_name="node"),
          task_weights=4.,
          output_names="o1",
          output_shapes={
              "o1": (2,)  # Positive and negative logits
          }),
      dict(
          testcase_name="SequenceTaskWithWeights",
          task=[
              dgi.DeepGraphInfomax(node_set_name="node"),
              classification.RootNodeMulticlassClassification(
                  node_set_name="node",
                  num_classes=10)  # Ten classes
          ],
          task_weights=[1., .5],
          output_names=["o1", "o2"],
          output_shapes={
              "o1": (2,),  # Positive and negative logits
              "o2": (10,)  # Ten classes
          }),
      dict(
          testcase_name="MappingTaskWithWeights",
          task={
              "dgi": dgi.DeepGraphInfomax(node_set_name="node"),
              "multiclass": classification.RootNodeMulticlassClassification(
                  node_set_name="node",
                  num_classes=10)  # Ten classes
          },
          task_weights={"multiclass": .5, "dgi": 1.},
          output_names={"dgi": "o1", "multiclass": "o2"},
          output_shapes={
              "o1": (2,),  # Positive and negative logits
              "o2": (10,)  # Ten classes
          }),
  ])
  def test_multitask(
      self,
      task: Union[Task, Sequence[Task], Mapping[str, Task]],
      task_weights: Union[Sequence[float], Mapping[str, float]],
      output_names: Mapping[str, str],
      output_shapes: Mapping[str, Sequence[int]]):
    schema = tfgnn.parse_schema(SCHEMA)
    gtspec = tfgnn.create_graph_spec_from_schema_pb(schema)
    example = tfgnn.write_example(tfgnn.random_graph_tensor(gtspec))
    ds_provider = DatasetProvider([example.SerializeToString()] * 6)

    def extract_labels(gt):
      return gt, gt.context["label"] % 10  # Ten labels

    def node_sets_fn(node_set, node_set_name):
      del node_set_name
      return node_set["features"]

    model_fn = model_templates.ModelFromInitAndUpdates(
        init=tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn),
        updates=[vanilla_mpnn.VanillaMPNNGraphUpdate(
            units=1,
            message_dim=2,
            receiver_tag=tfgnn.SOURCE,
            l2_regularization=5e-4,
            dropout_rate=0.1)])

    model_dir = self.create_tempdir()

    trainer = keras_fit.KerasTrainer(
        strategy=tf.distribute.get_strategy(),
        model_dir=model_dir,
        steps_per_epoch=1,
        validation_steps=1,
        restore_best_weights=False)

    orchestration.run(
        train_ds_provider=ds_provider,
        model_fn=model_fn,
        optimizer_fn=tf.keras.optimizers.Adam,
        epochs=1,
        trainer=trainer,
        task=task,
        task_weights=task_weights,
        gtspec=gtspec,
        drop_remainder=False,
        global_batch_size=4,
        feature_processors=[extract_labels],
        model_exporters=[model_export.KerasModelExporter(output_names)],
        valid_ds_provider=ds_provider)

    dataset = ds_provider.get_dataset(tf.distribute.InputContext())
    kwargs = {"examples": next(iter(dataset.batch(2)))}

    saved_model = tf.saved_model.load(os.path.join(model_dir, "export"))
    output = saved_model.signatures["serving_default"](**kwargs)

    self.assertLen(output, len(output_shapes))

    for k, v in output_shapes.items():
      self.assertShapeEqual(
          output[k],
          tf.random.uniform([2, *v]),  # Prepend the batch dim (above: 2)
          f"assertOutputShape({k})")


if __name__ == "__main__":
  tf.test.main()
