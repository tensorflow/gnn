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
"""Wraps OGBN and Planetoid datasets to use within tfgnn in-memory example.

* classes `NodeClassificationOgbDatasetWrapper` and `PlanetoidDatasetWrapper`,
  respectively, wrap datasets of OGBN and Planetoid. Both classes inherit class
  `NodeClassificationDatasetWrapper`. Therefore, they inherit methods
  `export_to_graph_tensor` and `iterate_once`, respectively, which return
  `GraphTensor` object (that can be fed into TF-GNN model) and return a
  tf.data which yields the `GraphTensor` object (once -- you may call .repeat())

* `create_graph_schema_from_directed` creates `tfgnn.GraphSchema` proto.
"""

import os
import pickle
import sys
from typing import Any, Mapping, List, Union
import urllib.request

import numpy as np
import ogb.nodeproppred
import scipy.sparse
import tensorflow as tf
import tensorflow_gnn as tfgnn


def get_ogbn_dataset(dataset_name, root_dir):
  return ogb.nodeproppred.NodePropPredDataset(dataset_name, root=root_dir)


class NodeClassificationDatasetWrapper:
  """Wraps graph datasets (nodes, edges, features).

  Inheriting classes implement straight-forward functions to adapt any external
  dataset into TFGNN, by exposing methods `iterate_once` and
  `export_to_graph_tensor` that yield GraphTensor objects that can be passed to
  TFGNN's modeling framework.
  """

  def num_classes(self) -> int:
    """Number of node classes. Entries in `labels` should be `< num_classes`."""
    raise NotImplementedError('num_classes')

  def node_features(self) -> np.ndarray:
    """Returns numpy float32 array of shape (num_nodes, feature_dimension)."""
    raise NotImplementedError()

  def node_features_dict(self, add_id=True) -> Mapping[str, np.ndarray]:
    raise NotImplementedError()

  def num_nodes(self) -> int:
    """Returns total number of graph nodes."""
    raise NotImplementedError()

  def edge_list(self) -> np.ndarray:
    """Returns numpy int array of shape (2, num_edges)."""
    raise NotImplementedError()

  def node_split(self) -> Mapping[str, np.ndarray]:
    """Returns dict with node IDs in {train, validation, test} partitions."""
    raise NotImplementedError()

  def labels(self) -> np.ndarray:
    """Returns int numpy array of length num_nodes with training labels.

    For test nodes, the label will be set to -1.
    """
    raise NotImplementedError()

  def test_labels(self) -> np.ndarray:
    """int numpy array of length num_nodes containing train and test labels."""
    raise NotImplementedError()

  def iterate_once(self, add_self_connections: bool = False,
                   split: Union[str, List[str]] = 'train',
                   make_undirected: bool = False) -> tf.data.Dataset:
    """tf.data iterator with one example containg entire graph (full-batch)."""
    graph_tensor = self.export_to_graph_tensor(
        add_self_connections, split, make_undirected)
    spec = graph_tensor.spec

    def once():
      yield graph_tensor

    return tf.data.Dataset.from_generator(once, output_signature=spec)

  def export_to_graph_tensor(
      self, add_self_connections: bool = False,
      split: Union[str, List[str]] = 'train',
      make_undirected: bool = False) -> tfgnn.GraphTensor:
    """Makes GraphTensor corresponding to the *entire graph*.

    It populates the nodes under node_set "nodes" (== tfgnn.NODES). Argument
    `make_undirected` controls `edge_set`. Further, `context` will contain
    `tf.int64` feature with name "seed" to store the indices for training nodes.

    Args:
      add_self_connections: If set, edges will be added:
        `[(i, i) for i in range(num_nodes)]`.
      split: Must be one of "train", "test", "valid". It controls the indices
        stored on `graph_tensor.context.features["seed"]`.
      make_undirected: If not set (default), result GraphTensor will contain two
        `edge_set` adjacencies: "edges" and "rev_edges" where the first encodes
        the original graph and the second encodes its transpose.

    Returns:
      GraphTensor containing the entire graph at-once.
    """
    edge_list = self.edge_list()
    num_nodes = self.num_nodes()
    all_nodes = np.arange(num_nodes, dtype=edge_list.dtype)
    if make_undirected:
      edge_list = np.concatenate([edge_list, edge_list[::-1]], axis=0)
    if add_self_connections:
      self_connections = np.stack([all_nodes, all_nodes], axis=0)
      edge_list = np.concatenate([edge_list, self_connections], axis=0)

    # Construct `GraphTensor` with `node_sets`, `edge_sets`, and `context`.
    node_features_dict = self.node_features_dict()
    node_features_dict = {k: _t(v) for k, v in node_features_dict.items()}
    if split == 'test' or 'test' in split:
      node_features_dict['label'] = _t(self.test_labels())
    else:
      node_features_dict['label'] = _t(self.labels())
    node_sets = {
        # Recall: tfgnn.NODES == 'nodes'
        tfgnn.NODES: tfgnn.NodeSet.from_fields(
            sizes=_t([num_nodes]),
            features=node_features_dict)
    }

    edge_sets = {
        'edges': tfgnn.EdgeSet.from_fields(
            sizes=_t([edge_list.shape[1]]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=(tfgnn.NODES, _t(edge_list[0])),
                target=(tfgnn.NODES, _t(edge_list[1]))))
    }

    if not make_undirected:
      edge_sets['rev_edges'] = tfgnn.EdgeSet.from_fields(
          sizes=_t([edge_list.shape[1]]),
          adjacency=tfgnn.Adjacency.from_indices(
              source=(tfgnn.NODES, _t(edge_list[1])),
              target=(tfgnn.NODES, _t(edge_list[0]))))

    # Expand seed nodes.
    if not isinstance(split, (tuple, list)):
      splits = [split]
    else:
      splits = split
    seed_nodes = np.concatenate(
        [self.node_split()[split].reshape(-1) for split in splits], axis=0)

    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets, edge_sets=edge_sets,
        context=tfgnn.Context.from_fields(features={'seed': _t(seed_nodes)}))

    return graph_tensor


def create_graph_schema_from_directed(
    dataset: NodeClassificationDatasetWrapper,
    make_undirected=False,
) -> tfgnn.GraphSchema:
  """Creates `GraphSchema` proto from directed OGBN graph.

  Output of this function can be used to create a `tf.TypeSpec` object as:

  ```
  obgn_dataset = ogb.nodeproppred.NodePropPredDataset(dataset_name)
  graph_schema = create_graph_schema_from_directed(obgn_dataset)
  type_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
  ```

  The `TypeSpec` can then be used to bootstrap a Keras model or to prepare the
  input pipeline.

  Args:
    dataset: NodeClassificationDatasetWrapper. Feature shapes and types returned
      by `dataset.node_features_dict()` will be added to graph schema.
    make_undirected: If set, only edge with type name 'edges' will be registerd.
      Otherwise, edges with name 'rev_edges' will additionally be registered in
      the schema. If you set this variable, you are expected to also set it in
      `make_full_graph_tensor()`.

  Returns:
    `tfgnn.GraphSchema` describing the node and edge features of the ogbn graph.
  """
  # Populate node features.
  schema = tfgnn.GraphSchema()
  node_features = schema.node_sets[tfgnn.NODES]

  for feat_name, graph_feats in dataset.node_features_dict().items():
    node_features.features[feat_name].dtype = (
        tf.dtypes.as_dtype(graph_feats).as_datatype_enum)
    for dim in graph_feats.shape[1:]:
      node_features.features[feat_name].shape.dim.add().size = dim

  # Populate edges with adjacency and it transpose.
  schema.edge_sets['edges'].source = tfgnn.NODES
  schema.edge_sets['edges'].target = tfgnn.NODES
  if not make_undirected:
    schema.edge_sets['rev_edges'].source = tfgnn.NODES
    schema.edge_sets['rev_edges'].target = tfgnn.NODES

  schema.context.features['seed'].dtype = tf.dtypes.int64.as_datatype_enum

  return schema


class NodeClassificationOgbDatasetWrapper(NodeClassificationDatasetWrapper):
  """Wraps OGBN dataset for in-memory learning."""

  def __init__(self, dataset_name, cache_dir=None):
    if cache_dir is None:
      cache_dir = os.environ.get(
          'OGB_CACHE_DIR', os.path.expanduser(os.path.join('~', 'data', 'ogb')))

    self.ogb_dataset = get_ogbn_dataset(dataset_name, cache_dir)
    self._graph, self._node_labels = self.ogb_dataset[0]
    self._node_split = self.ogb_dataset.get_idx_split()
    self._node_labels = self._node_labels[:, 0]  # rehape from [N, 1] to [N].

    self._train_labels = self._node_labels + 0  # Make a copy.
    self._train_labels[self._node_split['test']] = -1

  def num_classes(self) -> int:
    """Number of node classes. Entries in `labels` should be `< num_classes`."""
    return self.ogb_dataset.num_classes

  def node_features(self) -> np.ndarray:
    """Returns numpy float32 array of shape (num_nodes, feature_dimension)."""
    return self._graph['node_feat']

  def node_features_dict(self, add_id=True) -> Mapping[str, np.ndarray]:
    features = {key.split('node_', 1)[-1]: feats
                for key, feats in self._graph.items()
                if key.startswith('node_')}
    if add_id:
      features['#id'] = np.arange(self.num_nodes(), dtype='int32')
    return features

  def num_nodes(self) -> int:
    """Returns total number of graph nodes."""
    return self._graph['num_nodes']

  def edge_list(self) -> np.ndarray:
    """Returns numpy int array of shape (2, num_edges)."""
    return self._graph['edge_index']

  def node_split(self) -> Mapping[str, np.ndarray]:
    """Returns dict with node IDs in {train, validation, test} partitions."""
    return self._node_split

  def labels(self) -> np.ndarray:
    """Returns int numpy array of length num_nodes with training labels.

    For test nodes, the label will be set to -1.
    """
    return self._train_labels

  def test_labels(self) -> np.ndarray:
    """int numpy array of length num_nodes containing train and test labels."""
    return self._node_labels


def _maybe_download_file(source_url, destination_path, make_dirs=True):
  """Downloads URL `source_url` onto file `destination_path` if not present."""
  if not os.path.exists(destination_path):
    dir_name = os.path.dirname(destination_path)
    if make_dirs:
      try:
        os.makedirs(dir_name)
      except FileExistsError:
        pass

    with urllib.request.urlopen(source_url) as fin:
      with open(destination_path, 'wb') as fout:
        fout.write(fin.read())


class PlanetoidDatasetWrapper(NodeClassificationDatasetWrapper):
  """Wraps Planetoid node-classificaiton datasets.

  These datasets first appeared in the Planetoid [1] paper and popularized by
  the GCN paper [2].

  [1] Yang et al, ICML'16
  [2] Kipf & Welling, ICLR'17.
  """

  def __init__(self, dataset_name, cache_dir=None):
    allowed_names = ('pubmed', 'citeseer', 'cora')

    url_template = (
        'https://github.com/kimiyoung/planetoid/blob/master/data/'
        'ind.%s.%s?raw=true')
    file_parts = ['ally', 'allx', 'graph', 'ty', 'tx', 'test.index']
    if dataset_name not in allowed_names:
      raise ValueError('Dataset must be one of: ' + ', '.join(allowed_names))
    if cache_dir is None:
      cache_dir = os.environ.get(
          'PLANETOID_CACHE_DIR', os.path.expanduser(
              os.path.join('~', 'data', 'planetoid')))
    base_path = os.path.join(cache_dir, 'ind.%s' % dataset_name)
    # Download all files.
    for file_part in file_parts:
      source_url = url_template % (dataset_name, file_part)
      destination_path = os.path.join(
          cache_dir, 'ind.%s.%s' % (dataset_name, file_part))
      _maybe_download_file(source_url, destination_path)

    # Load data files.
    edge_lists = pickle.load(open(base_path + '.graph', 'rb'))
    allx = PlanetoidDatasetWrapper.load_x(base_path + '.allx')
    ally = np.load(base_path + '.ally', allow_pickle=True)

    testx = PlanetoidDatasetWrapper.load_x(base_path + '.tx')

    # Add test
    test_idx = list(
        map(int, open(base_path + '.test.index').read().split('\n')[:-1]))

    num_test_examples = max(test_idx) - min(test_idx) + 1
    sparse_zeros = scipy.sparse.csr_matrix((num_test_examples, allx.shape[1]),
                                           dtype='float32')

    allx = scipy.sparse.vstack((allx, sparse_zeros))
    llallx = allx.tolil()
    llallx[test_idx] = testx
    self._allx = np.array(llallx.todense())

    testy = np.load(base_path + '.ty', allow_pickle=True)
    ally = np.pad(ally, [(0, num_test_examples), (0, 0)], mode='constant')
    ally[test_idx] = testy

    self._num_nodes = len(edge_lists)
    self._num_classes = ally.shape[1]
    self._node_labels = np.argmax(ally, axis=1)
    self._train_labels = self._node_labels + 0  # Copy.
    self._train_labels[test_idx] = -1
    self._test_idx = np.array(test_idx, dtype='int32')

    # Will be used to construct (sparse) adjacency matrix.
    adj_src = []
    adj_target = []
    for node, neighbors in edge_lists.items():
      adj_src.extend([node] * len(neighbors))
      adj_target.extend(neighbors)

    self._edge_list = np.stack([adj_src, adj_target], axis=0)

  @staticmethod
  def load_x(filename):
    if sys.version_info > (3, 0):
      return pickle.load(open(filename, 'rb'), encoding='latin1')
    else:
      return np.load(filename)

  def num_classes(self) -> int:
    """Number of node classes. Entries in `labels` should be `< num_classes`."""
    return self._num_classes

  def node_features(self) -> np.ndarray:
    """Returns numpy float32 array of shape (num_nodes, feature_dimension)."""
    return self._allx

  def node_features_dict(self, add_id=True) -> Mapping[str, np.ndarray]:
    features = {'feat': self.node_features()}
    if add_id:
      features['#id'] = np.arange(self.num_nodes(), dtype='int32')
    return features

  def num_nodes(self) -> int:
    """Returns total number of graph nodes."""
    return self._num_nodes

  def edge_list(self) -> np.ndarray:
    """Returns numpy int array of shape (2, num_edges)."""
    return self._edge_list

  def node_split(self) -> Mapping[str, np.ndarray]:
    """Returns dict with node IDs in {train, validation, test} partitions."""
    # By default, we mimic Planetoid & GCN setup -- i.e., 20 labels per class.
    labels_per_class = int(os.environ.get('PLANETOID_LABELS_PER_CLASS', '20'))
    num_train_nodes = labels_per_class * self.num_classes()
    num_validate_nodes = 500
    train_ids = np.arange(num_train_nodes, dtype='int32')
    validate_ids = np.arange(
        num_train_nodes, num_train_nodes + num_validate_nodes, dtype='int32')
    return {'train': train_ids, 'valid': validate_ids, 'test': self._test_idx}

  def labels(self) -> np.ndarray:
    """Returns int numpy array of length num_nodes with training labels.

    For test nodes, the label will be set to -1.
    """
    return self._train_labels

  def test_labels(self) -> np.ndarray:
    """int numpy array of length num_nodes containing train and test labels."""
    return self._node_labels


def get_dataset(dataset_name):
  if dataset_name.startswith('ogbn-'):
    return NodeClassificationOgbDatasetWrapper(dataset_name)
  elif dataset_name in ('cora', 'citeseer', 'pubmed'):
    return PlanetoidDatasetWrapper(dataset_name)
  else:
    raise ValueError('Unknown Dataset name: ' + dataset_name)


# Can be replaced with: `_t = tf.convert_to_tensor`.
def _t(obj: Any) -> tf.Tensor:
  """short-hand for tf.convert_to_tensor."""
  return tf.convert_to_tensor(obj)
