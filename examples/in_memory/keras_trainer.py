"""Sample script for full-batch (entire graph) training of tfgnn model on OGBN.

This script runs end-to-end, i.e., requiring no pre- or post-processing scripts.
It holds the dataset in-memory, and processes the entire graph at each step. It
uses barebones tensorflow.

By default, script runs on 'ogbn-arxiv'. You substitute with another
node-classification dataset via flag --dataset.
"""
import functools
import json

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.examples.in_memory import datasets
from tensorflow_gnn.examples.in_memory import models
from tensorflow_gnn.examples.in_memory import reader_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'ogbn-arxiv',
                    'Name of dataset. Assumed to contain node-classification '
                    'task. OGBN datasets and Planetoid datasets are supported.')
flags.DEFINE_string('model', 'GCN',
                    'Model name. Choices: ' +
                    ', '.join(models.MODEL_NAMES))
flags.DEFINE_string('model_kwargs_json', '{}',
                    'JSON object encoding model arguments')
flags.DEFINE_integer('eval_every', 10, 'Eval every this many steps.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('l2_regularization', 1e-5,
                   'L2 Regularization for (non-bias) weights.')
flags.DEFINE_integer('steps', 101,
                     'Total number of training steps. Each step uses full '
                     'graph, so this is also the number of epochs.')


def main(unused_argv):
  dataset_wrapper = datasets.get_dataset(FLAGS.dataset)
  num_classes = dataset_wrapper.num_classes()
  model_kwargs = json.loads(FLAGS.model_kwargs_json)
  prefers_undirected, model = models.make_model_by_name(
      FLAGS.model, num_classes, l2_coefficient=FLAGS.l2_regularization,
      model_kwargs=model_kwargs)

  graph_schema = datasets.create_graph_schema_from_directed(
      dataset_wrapper, make_undirected=prefers_undirected)
  type_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

  input_graph = tf.keras.layers.Input(type_spec=type_spec)
  graph = input_graph

  # Since datasets.py imports features with feature name 'feat', extract it as
  # node feature set: `tfgnn.HIDDEN_STATE`.
  def init_node_state(node_set, node_set_name):
    del node_set_name
    return node_set['feat']
  graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=init_node_state)(graph)

  model_output = model(graph)  # Runs the model.

  # Grab activations corresponding to the seed nodes.
  seed_logits = reader_utils.readout_seed_node_features(model_output)

  keras_model = tf.keras.Model(inputs=input_graph, outputs=seed_logits)

  opt = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  loss = tf.keras.losses.CategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  keras_model.compile(opt, loss=loss, metrics=['acc'])

  train_dataset = dataset_wrapper.iterate_once(
      make_undirected=prefers_undirected)
  train_labels_dataset = train_dataset.map(
      functools.partial(reader_utils.pair_graphs_with_labels, num_classes))

  # Similarly for validation.
  validation_ds = dataset_wrapper.iterate_once(
      split='valid', make_undirected=prefers_undirected)
  validation_ds = validation_ds.map(
      functools.partial(reader_utils.pair_graphs_with_labels, num_classes))

  validation_repeated_ds = validation_ds.repeat()

  FLAGS.alsologtostderr = True  # To print accuracy and training progress.
  keras_model.fit(
      train_labels_dataset, epochs=FLAGS.steps,
      validation_data=validation_repeated_ds, validation_steps=1,
      validation_freq=FLAGS.eval_every)

  test_graph = dataset_wrapper.export_to_graph_tensor(
      split='test', make_undirected=prefers_undirected)
  test_graph, test_labels = reader_utils.pair_graphs_with_labels(
      num_classes, test_graph)
  accuracy = (tf.argmax(keras_model(test_graph), 1) ==
              tf.argmax(test_labels, 1)).numpy().mean()

  print('Final test accuracy=%f' % accuracy)

if __name__ == '__main__':
  app.run(main)
