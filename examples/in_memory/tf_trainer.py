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
"""Sample script for full-batch (entire graph) training of tfgnn model on OGBN.

This script runs end-to-end, i.e., requiring no pre- or post-processing scripts.
It holds the graph in-memory, and processes the entire graph at each step. It
uses barebones tensorflow.

By default, script runs on 'ogbn-arxiv'. You substitute with another
node-classification dataset via flag --dataset.
"""
import json

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.in_memory import datasets
from tensorflow_gnn.experimental.in_memory import models
from tensorflow_gnn.experimental.in_memory import reader_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'ogbn-arxiv',
                    'Name of dataset. Assumed to contain node-classification '
                    'task. OGBN datasets and Planetoid datasets are supported.')
flags.DEFINE_string(
    'ogb_cache_dir', None,
    'Overrides $OGB_CACHE_DIR to set the cache dir for downloading OGB '
    'datasets.')
flags.DEFINE_string('model', 'GCN',
                    'Model name. Choices: ' +
                    ', '.join(models.MODEL_NAMES))
flags.DEFINE_string('model_kwargs_json', '{}',
                    'JSON object encoding model arguments')
flags.DEFINE_integer('eval_every', 10, 'Eval every this many steps.')
flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate.')
flags.DEFINE_float('l2_regularization', 1e-5,
                   'L2 Regularization for (non-bias) weights.')
flags.DEFINE_integer('steps', 101,
                     'Total number of training steps. Each step uses full '
                     'graph, so this is also the number of epochs.')
flags.DEFINE_bool('train_on_validation', False,
                  'If set, also uses validation set as training set. This is '
                  'allowed by some tasks, such as OGBN. Nonetheless, this flag '
                  'should only be set *after* hyperparameter tuning.')


def main(unused_argv):
  graph_data = datasets.get_in_memory_graph_data(FLAGS.dataset)
  assert isinstance(graph_data, datasets.NodeClassificationGraphData)
  num_classes = graph_data.num_classes()
  model_kwargs = json.loads(FLAGS.model_kwargs_json)
  prefers_undirected, model = models.make_model_by_name(
      FLAGS.model, num_classes, l2_coefficient=FLAGS.l2_regularization,
      model_kwargs=model_kwargs)
  graph_data = graph_data.with_undirected_edges(prefers_undirected)

  # We want graph tensor with labels on nodes. However, function
  # reader_utils.pop_labels_from_graph() pops the labels, before feeding them
  # into model.
  graph_data = graph_data.with_labels_as_features(True)

  train_split = ['train']
  if FLAGS.train_on_validation:
    train_split.append('validation')
  graph_tensor = graph_data.with_split(train_split).as_graph_tensor()
  graph_tensor, seed_y = reader_utils.pop_labels_from_graph(
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
      model_out_graph_tensor = model(graph_tensor, training=True)
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

  valid_split = 'test' if FLAGS.train_on_validation else 'validation'
  valid_graph = graph_data.with_split(valid_split).as_graph_tensor()
  valid_graph, valid_y = reader_utils.pop_labels_from_graph(
      num_classes, valid_graph)
  valid_graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=init_node_state)(
      valid_graph)

  test_graph = graph_data.with_split('test').as_graph_tensor()
  test_graph, test_y = reader_utils.pop_labels_from_graph(
      num_classes, test_graph)
  test_graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=init_node_state)(
      test_graph)

  def estimate_validation_accuracy(y=valid_y, graph=valid_graph):
    graph = model(graph, training=False)
    predictions = tf.argmax(
        reader_utils.readout_seed_node_features(graph), axis=-1)
    labels = tf.argmax(y, axis=-1)
    is_correct_mask_np = (labels == predictions).numpy()

    return is_correct_mask_np.mean()

  print('At start: validation accuracy=%', estimate_validation_accuracy())
  for batch_id in range(FLAGS.steps):
    train_step()
    if batch_id % FLAGS.eval_every == 0:
      print('After step %i: validation accuracy=%g' % (
          batch_id, estimate_validation_accuracy()))
  print('Final test accuracy=%f' % estimate_validation_accuracy(
      y=test_y, graph=test_graph))

if __name__ == '__main__':
  app.run(main)
