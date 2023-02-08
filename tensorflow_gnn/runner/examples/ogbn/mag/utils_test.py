# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Tests for masking seed labels."""
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner.examples.ogbn.mag import utils

as_tensor = tf.convert_to_tensor


class MaskPaperLabelsTest(tf.test.TestCase, parameterized.TestCase):

  def test(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'paper': tfgnn.NodeSet.from_fields(
                sizes=as_tensor([2, 3, 4]),
                features={
                    'year': as_tensor([[2013], [2014],
                                       [2015], [2016], [2017],
                                       [2018], [2019], [2020], [2021]]),
                    'labels': as_tensor([[0], [1],
                                         [2], [3], [4],
                                         [5], [6], [7], [8]]),
                }
            )
        }
    )
    num_classes = 9
    year_feature = graph.node_sets['paper']['year']
    validation_and_test_mask = year_feature >= 2018
    masked_labels = utils.mask_paper_labels(
        graph.node_sets['paper'],
        label_feature_name='labels',
        mask_value=num_classes,
        extra_label_mask=validation_and_test_mask,
    )
    expected = as_tensor([[9], [1],
                          [9], [3], [4],
                          [9], [9], [9], [9]])
    self.assertAllEqual(masked_labels, expected)

    # If input_dim < num_classes + 1, there is an error on CPU.
    _ = tf.keras.layers.Embedding(num_classes+1, 4)(masked_labels)


class MakeCausalMaskTest(tf.test.TestCase, parameterized.TestCase):

  def test(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'paper': tfgnn.NodeSet.from_fields(
                sizes=as_tensor([2, 3, 4]),
                features={
                    'year': as_tensor([2013, 2012,
                                       2015, 2014, 2017,
                                       2018, 2019, 2020, 2021]),
                    'labels': as_tensor([0, 1,
                                         2, 3, 4,
                                         5, 6, 7, 8]),
                }
            )
        }
    )
    mask = utils.make_causal_mask(graph.node_sets['paper'])
    expected = as_tensor([True, False,
                          True, False, True,
                          True, True, True, True])
    self.assertAllEqual(mask, expected)


if __name__ == '__main__':
  tf.test.main()


