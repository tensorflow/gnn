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
import os

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.experimental.in_memory import datasets


class ToyDataset(datasets.NodeClassificationGraphData):

  def __init__(self):
    super().__init__()
    self._f1 = tf.random.uniform((3, 8), maxval=55, seed=1)
    self._f2 = tf.random.uniform((3, 5), maxval=55, seed=2)
    self._f3 = tf.random.uniform((10, 7), maxval=55, seed=3)
    self._train_labels = tf.constant([-1]*3 + [0]*3 + [1]*4)
    self._test_labels = tf.constant([0]*3 + [0]*3 + [1]*4)

  def node_features_dicts_without_labels(self) -> dict[
      tfgnn.NodeSetName, dict[str, tf.Tensor]]:
    return {
        'n1': {
            'f1': self._f1,
            'f2': self._f2,
        },
        'n2': {
            'f3': self._f3,
        }
    }

  def node_counts(self) -> dict[tfgnn.NodeSetName, int]:
    return {'n1': 3, 'n2': 10}

  def edge_lists(self) -> dict[tuple[str, str, str], tf.Tensor]:
    return {
        ('n1', 'e11', 'n1'): tf.constant([
            [0, 1, 1],
            [1, 1, 0],
        ]),
        ('n1', 'e12', 'n2'): tf.constant([
            [0, 0, 0, 1, 2, 2],
            [8, 7, 1, 3, 5, 2],
        ]),
    }

  def num_classes(self) -> int:
    return 2

  def node_split(self) -> datasets.NodeSplit:
    return datasets.NodeSplit(
        test=tf.constant([0, 1, 2]),
        validation=tf.constant([3, 4]),
        train=tf.constant([5, 6, 7, 8, 9]))

  def labels(self) -> tf.Tensor:
    return self._train_labels

  def test_labels(self) -> tf.Tensor:
    return self._test_labels

  @property
  def labeled_nodeset(self) -> tfgnn.NodeSetName:
    return 'n2'


class DatasetsTest(tf.test.TestCase):

  def test_save_then_load_node_classification_data(self):
    dataset = ToyDataset()
    dataset_path = os.path.join(self.get_temp_dir(), 'toy_dataset.npz')
    dataset.save(dataset_path)
    restored_dataset = datasets.NodeClassificationGraphData.load(dataset_path)
    gt1 = dataset.as_graph_tensor()
    gt2 = restored_dataset.as_graph_tensor()
    ex1 = tfgnn.write_example(gt1)
    ex2 = tfgnn.write_example(gt2)
    self.assertAllEqual(ex1, ex2)


if __name__ == '__main__':
  tf.test.main()
