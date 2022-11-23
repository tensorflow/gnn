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
"""Sample script for on-the-fly-sampling training of tfgnn model on OGBN.

This script runs end-to-end, i.e., requiring no pre- or post-processing scripts.
It holds the dataset in-memory, and at each step, samples subgraph as mini-batch

By default, script runs on 'ogbn-arxiv'. You substitute with another
node-classification dataset via flag --dataset.
"""
import json
import math

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.experimental.in_memory import datasets
from tensorflow_gnn.experimental.in_memory import int_arithmetic_sampler as ia_sampler
from tensorflow_gnn.experimental.in_memory import models
from tensorflow_gnn.experimental.in_memory import reader_utils
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
flags.DEFINE_integer('epochs', 50,
                     'Total number of training epochs. Each step uses '
                     '--num_seeds training nodes.')
flags.DEFINE_integer('num_seeds', 200,
                     'Number of labeled nodes to seed subgraphs in every '
                     'training step.')


def main(unused_argv):
  graph_data = datasets.get_in_memory_graph_data(FLAGS.dataset)
  assert isinstance(graph_data, datasets.NodeClassificationGraphData)
  num_classes = graph_data.num_classes()
  model_kwargs = json.loads(FLAGS.model_kwargs_json)
  prefers_undirected, model = models.make_model_by_name(
      FLAGS.model, num_classes, l2_coefficient=FLAGS.l2_regularization,
      model_kwargs=model_kwargs)
  graph_data = graph_data.with_undirected_edges(prefers_undirected)

  graph_schema = graph_data.graph_schema()
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

  sampling_kwargs = dict(
      sampling_spec=(
          sampling_spec_builder.SamplingSpecBuilder(
              graph_schema,
              sampling_spec_builder.SamplingStrategy.RANDOM_UNIFORM)
          .seed().sample([3, 3]).build()),
      num_seed_nodes=FLAGS.num_seeds,
      sampling_mode=ia_sampler.EdgeSampling.WITH_REPLACEMENT
  )
  # Subgraph samples for training.
  train_dataset = ia_sampler.NodeClassificationGraphSampler(
      graph_data.with_split('train')).as_dataset(**sampling_kwargs)

  # Subgraph samples for validation.
  validation_ds = ia_sampler.NodeClassificationGraphSampler(
      graph_data.with_split('validation')).as_dataset(**sampling_kwargs)
  validation_repeated_ds = validation_ds.repeat()

  FLAGS.alsologtostderr = True  # To print accuracy and training progress.
  steps_per_epoch = math.ceil(graph_data.node_split().train.shape[0]
                              / FLAGS.num_seeds)
  keras_model.fit(
      train_dataset,
      verbose=1,
      epochs=FLAGS.epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=validation_repeated_ds,
      validation_steps=1,
      validation_freq=FLAGS.eval_every)

  test_graph = (graph_data.with_split('test').with_labels_as_features(True)
                .as_graph_tensor())
  test_graph, test_labels = reader_utils.pop_labels_from_graph(
      num_classes, test_graph)
  accuracy = (tf.argmax(keras_model(test_graph), -1) ==
              tf.argmax(test_labels, -1)).numpy().mean()

  print('Final test accuracy=%f' % accuracy)

if __name__ == '__main__':
  app.run(main)
