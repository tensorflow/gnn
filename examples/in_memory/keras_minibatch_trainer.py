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

import datasets
from tensorflow_gnn.examples.in_memory import int_arithmetic_sampler as ia_sampler
import models
import reader_utils
from tensorflow_gnn.sampler import sampling_spec_builder


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'ogbn-arxiv',
                    'Name of dataset. Assumed to contain node-classification '
                    'task. OGBN datasets and Planetoid datasets are supported.')
flags.DEFINE_string('model', 'GCN',
                    'Model name. Choices: ' +
                    ', '.join(models.MODEL_NAMES))
flags.DEFINE_string('model_kwargs_json', '{}',
                    'JSON object encoding model arguments')
flags.DEFINE_integer('eval_every', 500, 'Eval every this many steps.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('l2_regularization', 1e-5,
                   'L2 Regularization for (non-bias) weights.')
flags.DEFINE_integer('steps', 10_000,
                     'Total number of training steps. Each step uses '
                     '--batch_size training nodes.')
flags.DEFINE_integer('batch_size', 200,
                     'Number of labeled seed nodes in every batch.')


def main(unused_argv):
  dataset = datasets.get_dataset(FLAGS.dataset)
  assert isinstance(dataset, datasets.NodeClassificationDataset)
  num_classes = dataset.num_classes()
  model_kwargs = json.loads(FLAGS.model_kwargs_json)
  prefers_undirected, model = models.make_model_by_name(
      FLAGS.model, num_classes, l2_coefficient=FLAGS.l2_regularization,
      model_kwargs=model_kwargs)

  graph_schema = dataset.graph_schema(make_undirected=prefers_undirected)
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

  # Subgraph samples for training.
  train_sampling_spec = (sampling_spec_builder.SamplingSpecBuilder(graph_schema)
                         .seed().sample([3, 3]).to_sampling_spec())
  _, train_dataset = ia_sampler.make_node_classification_tf_dataset(
      dataset, sampling_spec=train_sampling_spec,
      batch_size=FLAGS.batch_size,
      sampling=ia_sampler.EdgeSampling.WITH_REPLACEMENT,
      make_undirected=prefers_undirected)

  train_labels_dataset = train_dataset.map(
      functools.partial(reader_utils.pair_graphs_with_labels, num_classes))

  # Subgraph samples for validation.
  _, validation_ds = ia_sampler.make_node_classification_tf_dataset(
      dataset, sampling_spec=train_sampling_spec,
      batch_size=FLAGS.batch_size,
      sampling=ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT,
      split='valid', make_undirected=prefers_undirected)
  validation_ds = validation_ds.map(
      functools.partial(reader_utils.pair_graphs_with_labels, num_classes))

  validation_repeated_ds = validation_ds.repeat()

  FLAGS.alsologtostderr = True  # To print accuracy and training progress.
  keras_model.fit(
      train_labels_dataset,
      epochs=FLAGS.steps,
      steps_per_epoch=1,
      validation_data=validation_repeated_ds,
      validation_steps=10,
      validation_freq=FLAGS.eval_every)

  test_graph = dataset.as_graph_tensor(
      split='test', make_undirected=prefers_undirected)
  test_graph, test_labels = reader_utils.pair_graphs_with_labels(
      num_classes, test_graph)
  accuracy = (tf.argmax(keras_model(test_graph), 1) ==
              tf.argmax(test_labels, 1)).numpy().mean()

  print('Final test accuracy=%f' % accuracy)

if __name__ == '__main__':
  app.run(main)
