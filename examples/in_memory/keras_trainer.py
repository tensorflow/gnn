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
from tensorflow_gnn.experimental.in_memory import datasets
from tensorflow_gnn.experimental.in_memory import models
from tensorflow_gnn.experimental.in_memory import reader_utils

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

  graph_schema = graph_data.graph_schema()  # Without labels.
  type_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

  # We want graph tensor with labels on nodes. However, function
  # reader_utils.pop_labels_from_graph() pops the labels, before feeding them
  # into model.
  graph_data = graph_data.with_labels_as_features(True)

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

  train_split = ['train']
  if FLAGS.train_on_validation:
    train_split.append('validation')

  train_dataset = tf.data.Dataset.from_tensors(
      graph_data.with_split(train_split).as_graph_tensor())
  train_dataset = train_dataset.map(
      functools.partial(reader_utils.pop_labels_from_graph, num_classes))

  # Similarly for validation.
  valid_split = 'test' if FLAGS.train_on_validation else 'validation'
  validation_ds = tf.data.Dataset.from_tensors(
      graph_data.with_split(valid_split).as_graph_tensor())
  validation_ds = validation_ds.map(
      functools.partial(reader_utils.pop_labels_from_graph, num_classes))

  validation_repeated_ds = validation_ds.repeat()

  start_alsologtostderr = FLAGS.alsologtostderr
  FLAGS.alsologtostderr = True  # To print accuracy and training progress.
  keras_model.fit(
      train_dataset, epochs=FLAGS.steps,
      validation_data=validation_repeated_ds, validation_steps=1,
      validation_freq=FLAGS.eval_every, verbose=1)

  test_graph = (graph_data.with_split('test').with_labels_as_features(True)
                .as_graph_tensor())
  test_graph, test_labels = reader_utils.pop_labels_from_graph(
      num_classes, test_graph)
  accuracy = (tf.argmax(keras_model(test_graph), -1) ==
              tf.argmax(test_labels, -1)).numpy().mean()
  FLAGS.alsologtostderr = start_alsologtostderr
  print('\n\n  ****** \n\n Final test accuracy=%f \n\n' % accuracy)

if __name__ == '__main__':
  app.run(main)
