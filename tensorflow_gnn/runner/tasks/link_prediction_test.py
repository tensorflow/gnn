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
"""Tests for Link Prediction."""

import functools

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner.tasks import link_prediction


def _get_graph_tensor(
    readout_ns_name='_readout',
    readout_src_es_names=('_readout/source', 'nodes1', '_readout'),
    readout_tgt_es_names=('_readout/target', 'nodes2', '_readout'),
    ) -> tfgnn.GraphTensor:
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          'nodes1': tfgnn.NodeSet.from_fields(
              sizes=[3],
              features={
                  'id': ['a1', 'a2', 'a3'],
                  tfgnn.HIDDEN_STATE: tf.constant([
                      [1.0, 2.0], [3.0, 4.0], [4.0, 5.0]
                  ], dtype=tf.float32),
              }),
          'nodes2': tfgnn.NodeSet.from_fields(
              sizes=[3],
              features={
                  'id': ['b1', 'b2', 'b3'],
                  tfgnn.HIDDEN_STATE: tf.constant([
                      [1.0, -2.0], [3.0, -4.0], [4.0, -5.0]
                  ], dtype=tf.float32),
              }),
          readout_ns_name: tfgnn.NodeSet.from_fields(
              sizes=[2],
              features={
                  'label': tf.constant([
                      [1.0],  # Positive label.
                      [0.0],  # Negative label.
                  ], dtype=tf.float32),
              },
          )
      },
      edge_sets={
          'my_edgeset': tfgnn.EdgeSet.from_fields(
              sizes=[3],
              features={},
              adjacency=tfgnn.Adjacency.from_indices(
                  source=('nodes1', [0, 0, 1]),
                  target=('nodes2', [0, 1, 0]))),
          'e22': tfgnn.EdgeSet.from_fields(
              sizes=[4],
              features={},
              adjacency=tfgnn.Adjacency.from_indices(
                  source=('nodes2', [0, 0, 1, 2]),
                  target=('nodes2', [1, 2, 0, 0]))),
          readout_src_es_names[0]: tfgnn.EdgeSet.from_fields(
              # ^ should be "_readout/source"
              sizes=[2],
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(readout_src_es_names[1], [2, 2]),
                  #       ^ should be "nodes1"
                  target=(readout_src_es_names[2], [0, 1]))
              #           ^ should be "_readout"
          ),
          readout_tgt_es_names[0]: tfgnn.EdgeSet.from_fields(
              # ^ should be "_readout/target"
              sizes=[2],
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(readout_tgt_es_names[1], [0, 1]),
                  #       ^ should be "nodes2"
                  target=(readout_tgt_es_names[2], [0, 1]))
              #           ^ should be "_readout"
          ),
      })


class LinkPredictionTest(tf.test.TestCase, parameterized.TestCase):

  def test_predict_on_dot_product_link_prediction(self):
    task = link_prediction.DotProductLinkPrediction()
    similarities = task.predict(_get_graph_tensor())
    src_feats_2 = tf.constant([4.0, 5.0])
    tgt_feats_0 = tf.constant([1.0, -2.0])
    tgt_feats_1 = tf.constant([3.0, -4.0])
    # Should compute: sim(2, 0) and sim(2, 1).
    sim_0 = similarities[0, 0]
    sim_1 = similarities[1, 0]
    self.assertAllClose(sim_0, tf.reduce_sum(tgt_feats_0 * src_feats_2))
    self.assertAllClose(sim_1, tf.reduce_sum(tgt_feats_1 * src_feats_2))

  def test_predict_on_hadamard_product_link_prediction(self):
    task = link_prediction.HadamardProductLinkPrediction()
    similarities = task.predict(_get_graph_tensor())
    src_feats_2 = tf.constant([4.0, 5.0])
    tgt_feats_0 = tf.constant([1.0, -2.0])
    tgt_feats_1 = tf.constant([3.0, -4.0])
    # Should compute: sim(2, 0) and sim(2, 1).
    sim_0 = similarities[0, 0]
    sim_1 = similarities[1, 0]
    weight_vector = task._dense_layer.trainable_variables[0][:, 0]
    self.assertAllClose(
        sim_0, tf.reduce_sum(tgt_feats_0 * src_feats_2 * weight_vector))
    self.assertAllClose(
        sim_1, tf.reduce_sum(tgt_feats_1 * src_feats_2 * weight_vector))

  def test_preprocess_reads_label_field(self):
    task = link_prediction.DotProductLinkPrediction()
    input_gt = _get_graph_tensor()
    output_gt, label = task.preprocess(input_gt)
    self.assertEqual(tfgnn.write_example(input_gt),
                     tfgnn.write_example(output_gt))
    self.assertAllClose(label, tf.constant([[1.0], [0.0]]))

  @parameterized.named_parameters([
      dict(
          testcase_name='mal_named_readout_nodeset_name',
          readout_ns_name='_readout:misspelled'),
      dict(
          testcase_name='mal_named_readout_source_edge_set_name',
          readout_src_es_names=('_readout:AA/source', 'nodes1', '_readout')),
      dict(
          testcase_name='mal_named_readout_source_edge_set_target',
          readout_src_es_names=('_readout/source', 'nodes1', '_readout:BB')),
      dict(
          testcase_name='mal_named_readout_target_edge_set_name',
          readout_tgt_es_names=('_readout:AA/source', 'nodes1', '_readout')),
      dict(
          testcase_name='mal_named_readout_target_edge_set_target',
          readout_tgt_es_names=('_readout/source', 'nodes1', '_readout:BB')),
  ])
  def test_preprocess_fails_on_invalid_input(self, **kwargs):
    task = link_prediction.DotProductLinkPrediction()
    gt = _get_graph_tensor(**kwargs)
    self.assertRaises(ValueError, functools.partial(task.preprocess, gt))


if __name__ == '__main__':
  tf.test.main()
