"""Sample script for full-batch (entire graph) training of tfgnn model on OGBN.

This script runs end-to-end, i.e., requiring no pre- or post-processing scripts.
It holds the dataset in-memory, and processes the entire graph at each step. It
uses barebones tensorflow.

By default, script runs on 'ogbn-arxiv'. You substitute with another OGB
(Stanford Open Graph Benchmark) node-classification dataset via flag --dataset.
"""
from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.examples.in_memory import datasets
from tensorflow_gnn.examples.in_memory import models
from tensorflow_gnn.examples.in_memory import reader_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'ogbn-arxiv',
                    'Name of OGB dataset. Assumed to contain '
                    'node-classification task')
flags.DEFINE_string(
    'ogb_cache_dir', None,
    'Overrides $OGB_CACHE_DIR to set the cache dir for downloading OGB '
    'datasets.')
flags.DEFINE_string('model', 'GCN',
                    'Model name. Choices: ' +
                    ', '.join(models.MODEL_NAMES))
flags.DEFINE_integer('eval_every', 10, 'Eval every this many steps.')
flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate.')
flags.DEFINE_float('l2_regularization', 1e-5,
                   'L2 Regularization for (non-bias) weights.')
flags.DEFINE_integer('steps', 101,
                     'Total number of training steps. Each step uses full '
                     'graph, so this is also the number of epochs.')


def main(unused_argv):
  ogb_wrapper = datasets.NodeClassificationOgbDatasetWrapper(
      FLAGS.dataset, cache_dir=FLAGS.ogb_cache_dir)
  num_classes = ogb_wrapper.num_classes()
  prefers_undirected, model = models.make_model_by_name(
      FLAGS.model, num_classes, l2_coefficient=FLAGS.l2_regularization)

  graph_tensor = ogb_wrapper.export_to_graph_tensor(
      make_undirected=prefers_undirected)
  graph_tensor, seed_y = reader_utils.pair_graphs_with_labels(
      num_classes, graph_tensor)

  # Since datasets.py imports features with feature name 'feat', extract it as
  # node feature set: `tf.HIDDEN_STATE`.
  def init_node_state(node_set, node_set_name):
    del node_set_name
    return node_set['feat']
  graph_tensor = tfgnn.keras.layers.MapFeatures(node_sets_fn=init_node_state)(
      graph_tensor)

  opt = tf.keras.optimizers.Adam(FLAGS.learning_rate)

  @tf.function
  def train_step():
    with tf.GradientTape() as tape:
      # Model output.
      model_out_graph_tensor = model(graph_tensor)
      seed_logits = reader_utils.readout_seed_node_features(
          model_out_graph_tensor)
      # Compare with ground-truth.
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=seed_y, logits=seed_logits)
      loss = tf.reduce_mean(loss)
      loss += sum(model.losses)  # Add regularization losses.

    # Step
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

  valid_graph = ogb_wrapper.export_to_graph_tensor(split='valid')
  valid_graph, valid_y = reader_utils.pair_graphs_with_labels(
      num_classes, valid_graph)
  valid_graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=init_node_state)(
      valid_graph)

  def estimate_validation_accuracy():
    graph_tensor = model(valid_graph)
    predictions = tf.argmax(
        reader_utils.readout_seed_node_features(graph_tensor), axis=1)
    labels = tf.argmax(valid_y, axis=1)
    is_correct_mask_np = (labels == predictions).numpy()

    return is_correct_mask_np.mean()

  print('At start: validation accuracy=%',
        estimate_validation_accuracy())
  for batch_id in range(FLAGS.steps):
    train_step()
    if batch_id % FLAGS.eval_every == 0:
      print('After step %i: validation accuracy=%g' % (
          batch_id, estimate_validation_accuracy()))


if __name__ == '__main__':
  app.run(main)
