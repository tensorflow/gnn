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

* classes `OgbnDataset` and `PlanetoidDataset`, respectively, wrap datasets of
  OGBN and Planetoid. Both classes inherit class
  `NodeClassificationDatasetWrapper`. Therefore, they inherit methods
  `export_to_graph_tensor` and `iterate_once`, respectively, which return
  `GraphTensor` object (that can be fed into TF-GNN model) and return a
  tf.data which yields the `GraphTensor` object (once -- you may call .repeat())

* `create_graph_schema_from_directed` creates `tfgnn.GraphSchema` proto.
"""
import os
import pickle
import sys
from typing import Any, Mapping, MutableMapping, List, Union, Tuple, NamedTuple
import urllib.request

import numpy as np
import ogb.nodeproppred
import scipy.sparse
import tensorflow as tf
import tensorflow_gnn as tfgnn


def get_ogbn_dataset(dataset_name, root_dir):
  return ogb.nodeproppred.NodePropPredDataset(dataset_name, root=root_dir)


class NodeSplit(NamedTuple):
  """Contains 1D int tensors holding positions of {train, valid, test} nodes."""
  train: tf.Tensor
  valid: tf.Tensor
  test: tf.Tensor


class NodeClassificationDatasetWrapper:
  """Wraps graph datasets (nodes, edges, features).

  Inheriting classes implement straight-forward functions to adapt any external
  dataset into TFGNN, by exposing methods `iterate_once` and
  `export_to_graph_tensor` that yield GraphTensor objects that can be passed to
  TFGNN's modeling framework.
  """

  def num_classes(self) -> int:
    """Number of node classes. Max of `labels` should be `< num_classes`."""
    raise NotImplementedError('num_classes')

  def node_features_dicts(self, add_id=True) -> Mapping[
      tfgnn.NodeSetName, MutableMapping[str, tf.Tensor]]:
    raise NotImplementedError()

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    """Returns total number of graph nodes per node set."""
    raise NotImplementedError()

  def edge_lists(self) -> Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
      tf.Tensor]:
    """Returns dict from "edge type tuple" to int array of shape (2, num_edges).

    "edge type tuple" string-tuple: (src_node_set, edge_set, target_node_set).
    """
    raise NotImplementedError()

  def node_split(self) -> NodeSplit:
    """Returns dict with keys "train", "valid", "test" to node indices.

    These indices correspond to self.target_node_set.
    """
    raise NotImplementedError()

  def labels(self) -> tf.Tensor:
    """Returns int32 vector of length num_nodes with training labels.

    For test nodes, the label will be set to -1.
    """
    raise NotImplementedError()

  def test_labels(self) -> tf.Tensor:
    """int numpy array of length num_nodes containing train and test labels."""
    raise NotImplementedError()

  @property
  def labeled_nodeset(self) -> str:
    """Name of node set which `labels` and `node_splits` reference."""
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
        `[(i, i) for i in range(num_nodes)]` for edge_sets that have source node
        set name equals to target, and `num_nodes` is the number of nodes of
        said node set. NOTE: If self-connections already exist, *they will be
        duplicated*.
      split: Must be one of "train", "test", "valid". It controls the indices
        stored on `graph_tensor.context.features["seed"]`.
      make_undirected: If set, adds reverse edges to edge_sets connecting
        an endpoint node_set to the same node set. If not set (default), then
        result GraphTensor will contain twice as many edge sets. Edge set: "<X>"
        will contain original (forward) edges and edgeset "rev_<X>" will contain
        the reverse edges. edge set name "<X>" is dataset-dependant. We set <X>
        to "edges" for homogeneous graphs. For example, "<X>" can be "cites" for
        a citation network.

    Returns:
      GraphTensor containing the entire graph at-once.
    """
    # Prepare node sets, edge sets, context, to construct graph tensor.
    ## Node sets.
    node_counts = self.node_counts()
    node_features_dicts = self.node_features_dicts()

    if not isinstance(split, (tuple, list)):
      splits = [split]
    else:
      splits = split
    if 'test' in split:
      node_features_dicts[self.labeled_nodeset]['label'] = self.test_labels()
    else:
      node_features_dicts[self.labeled_nodeset]['label'] = self.labels()

    node_sets = {}
    for node_set_name, node_features_dict in node_features_dicts.items():
      node_sets[node_set_name] = tfgnn.NodeSet.from_fields(
          sizes=as_tensor([node_counts[node_set_name]]),
          features=node_features_dict)

    ## Edge set.
    edge_sets = {}
    for edge_type, edge_list in self.edge_lists().items():
      (source_node_set_name, edge_set_name, target_node_set_name) = edge_type

      if make_undirected and source_node_set_name == target_node_set_name:
        edge_list = tf.concat([edge_list, edge_list[::-1]], axis=0)
      if add_self_connections and source_node_set_name == target_node_set_name:
        all_nodes = tf.range(node_counts[source_node_set_name],
                             dtype=edge_list.dtype)
        self_connections = tf.stack([all_nodes, all_nodes], axis=0)
        edge_list = tf.concat([edge_list, self_connections], axis=0)
      edge_sets[edge_set_name] = tfgnn.EdgeSet.from_fields(
          sizes=tf.shape(edge_list)[1:2],
          adjacency=tfgnn.Adjacency.from_indices(
              source=(source_node_set_name, edge_list[0]),
              target=(target_node_set_name, edge_list[1])))
      if not make_undirected:
        edge_sets['rev_' + edge_set_name] = tfgnn.EdgeSet.from_fields(
            sizes=tf.shape(edge_list)[1:2],
            adjacency=tfgnn.Adjacency.from_indices(
                source=(target_node_set_name, edge_list[1]),
                target=(source_node_set_name, edge_list[0])))

    ## Context.
    # Expand seed nodes.
    node_split = self.node_split()
    seed_nodes = tf.concat(
        [getattr(node_split, split) for split in splits], axis=0)

    # Construct `GraphTensor` with `node_sets`, `edge_sets`, and `context`.
    seed_feature_name = 'seed_nodes.' + self.labeled_nodeset
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets, edge_sets=edge_sets,
        context=tfgnn.Context.from_fields(
            features={seed_feature_name: seed_nodes}))

    return graph_tensor

  def export_graph_schema(
      self, make_undirected: bool = False) -> tfgnn.GraphSchema:
    return create_graph_schema_from_directed(
        self, make_undirected=make_undirected)


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
  # Populate node features specs.
  schema = tfgnn.GraphSchema()
  for node_set_name, node_set in dataset.node_features_dicts().items():
    node_features = schema.node_sets[node_set_name]
    for feat_name, feature in node_set.items():
      node_features.features[feat_name].dtype = feature.dtype.as_datatype_enum
      for dim in feature.shape[1:]:
        node_features.features[feat_name].shape.dim.add().size = dim

  # Populate edge specs.
  for edge_type in dataset.edge_lists().keys():
    src_node_set_name, edge_set_name, dst_node_set_name = edge_type
    # Populate edges with adjacency and it transpose.
    schema.edge_sets[edge_set_name].source = src_node_set_name
    schema.edge_sets[edge_set_name].target = dst_node_set_name
    if not make_undirected:
      schema.edge_sets['rev_' + edge_set_name].source = dst_node_set_name
      schema.edge_sets['rev_' + edge_set_name].target = src_node_set_name

  schema.context.features['seed_nodes.' + dataset.labeled_nodeset].dtype = (
      tf.int64.as_datatype_enum)

  return schema


class _OgbnGraph(NamedTuple):
  # Maps "node set name" -> number of nodes.
  num_nodes_dict: Mapping[str, int]

  # Maps "node set name" to dict of "feature name"->tf.Tensor.
  node_feat_dict: Mapping[str, MutableMapping[str, tf.Tensor]]

  # maps (source node set name, edge set name, target node set name) -> edges,
  # where edges is tf.Tensor of shape (2, num edges).
  edge_index_dict: Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName], tf.Tensor]


class OgbnDataset(NodeClassificationDatasetWrapper):
  """Wraps OGBN dataset for in-memory learning."""

  def __init__(self, dataset_name, cache_dir=None):
    if cache_dir is None:
      cache_dir = os.environ.get(
          'OGB_CACHE_DIR', os.path.expanduser(os.path.join('~', 'data', 'ogb')))

    self.ogb_dataset = get_ogbn_dataset(dataset_name, cache_dir)
    self._graph, self._node_labels, self._node_split, self._labeled_nodeset = (
        OgbnDataset._to_heterogenous(self.ogb_dataset))

    # rehape from [N, 1] to [N].
    self._node_labels = self._node_labels[:, 0]

    # train labels (test set to -1).
    self._train_labels = np.copy(self._node_labels)
    self._train_labels[self._node_split.test] = -1

    self._train_labels = as_tensor(self._train_labels)
    self._node_labels = as_tensor(self._node_labels)

  @staticmethod
  def _to_heterogenous(
      ogb_dataset: ogb.nodeproppred.NodePropPredDataset) -> Tuple[
          _OgbnGraph,  # graph_dict.
          np.ndarray,   # node_labels.
          NodeSplit,   # idx_split.
          str]:
    """Returns heterogeneous dicts from homogenous or heterogeneous dataset.

    Args:
      ogb_dataset: OGBN dataset. It can be homogeneous (single node set type,
        single edge set type), or heterogeneous (various node/edge set types),
        and returns data structure as-if the dataset is heterogeneous (i.e.,
        names each node/edge set). If input is a homogeneous graph, then the
        node set will be named "nodes" and the edge set will be named "edges".

    Returns:
      tuple: `(graph_dict, node_labels, idx_split, labeled_nodeset)`, where:
        `graph_dict` is instance of _OgbnGraph.
        `node_labels`: np.array of labels, with .shape[0] equals number of nodes
          in node set with name `labeled_nodeset`.
        `idx_split`: instance of NodeSplit. Members `train`, `test` and `valid`,
          respectively, contain indices of nodes in node set with name
          `labeled_nodeset`.
        `labeled_nodeset`: name of node set that the node-classification task is
          designed over.
    """
    graph, node_labels = ogb_dataset[0]
    if 'edge_index_dict' in graph:  # Graph is already heterogeneous
      assert 'num_nodes_dict' in graph
      assert 'node_feat_dict' in graph
      labeled_nodeset = list(node_labels.keys())
      if len(labeled_nodeset) != 1:
        raise ValueError('Expecting OGB dataset with *one* node set with '
                         'labels. Found: ' + ', '.join(labeled_nodeset))
      labeled_nodeset = labeled_nodeset[0]

      node_labels = node_labels[labeled_nodeset]
      # idx_split is dict: {'train': {target_node_set: np.array}, 'test': ...}.
      idx_split = ogb_dataset.get_idx_split()
      # Change to {'train': Tensor, 'test': Tensor, 'valid': Tensor}
      idx_split = {split_name: as_tensor(split_dict[labeled_nodeset])
                   for split_name, split_dict in idx_split.items()}
      idx_split = NodeSplit(**idx_split)
      # node set name -> feature name -> feature matrix (numNodes x featDim).
      node_set = {node_set_name: {'feat': as_tensor(feat)}
                  for node_set_name, feat in graph['node_feat_dict'].items()}
      # Populate remaining features
      for key, node_set_name_to_feat in graph.items():
        if key.startswith('node_') and key != 'node_feat_dict':
          feat_name = key.split('node_', 1)[-1]
          for node_set_name, feat in node_set_name_to_feat.items():
            node_set[node_set_name][feat_name] = as_tensor(feat)
      ogbn_graph = _OgbnGraph(
          num_nodes_dict=graph['num_nodes_dict'],
          node_feat_dict=node_set,
          edge_index_dict={k: as_tensor(v)
                           for k, v in graph['edge_index_dict'].items()})

      return ogbn_graph, node_labels, idx_split, labeled_nodeset

    # Homogenous graph. Make heterogeneous.
    ogbn_graph = _OgbnGraph(
        edge_index_dict={
            (tfgnn.NODES, tfgnn.EDGES, tfgnn.NODES): as_tensor(
                graph['edge_index']),
        },
        num_nodes_dict={tfgnn.NODES: graph['num_nodes']},
        node_feat_dict={tfgnn.NODES: {'feat': as_tensor(graph['node_feat'])}},
    )
    # Copy other node information.
    for key, value in graph.items():
      if key != 'node_feat' and key.startswith('node_'):
        key = key.split('node_', 1)[-1]
        ogbn_graph.node_feat_dict[tfgnn.NODES][key] = as_tensor(value)
    idx_split = NodeSplit(**tf.nest.map_structure(
        tf.convert_to_tensor, ogb_dataset.get_idx_split()))
    return ogbn_graph, node_labels, idx_split, tfgnn.NODES

  def num_classes(self) -> int:
    return self.ogb_dataset.num_classes

  def node_features_dicts(self, add_id=True) -> Mapping[
      tfgnn.NodeSetName, MutableMapping[str, tf.Tensor]]:
    # Deep-copy dict (*but* without copying tf.Tensor objects).
    node_sets = self._graph.node_feat_dict
    node_sets = {node_set_name: dict(node_set.items())
                 for node_set_name, node_set in node_sets.items()}
    if add_id:
      node_counts = self.node_counts()
      for node_set, feat_dict in node_sets.items():
        feat_dict['#id'] = tf.range(node_counts[node_set], dtype=tf.int32)
    return node_sets

  @property
  def labeled_nodeset(self):
    return self._labeled_nodeset

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    return self._graph.num_nodes_dict

  def edge_lists(self) -> Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
      tf.Tensor]:
    return self._graph.edge_index_dict

  def node_split(self) -> NodeSplit:
    return self._node_split

  def labels(self) -> tf.Tensor:
    return self._train_labels

  def test_labels(self) -> tf.Tensor:
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


class PlanetoidDataset(NodeClassificationDatasetWrapper):
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
    allx = PlanetoidDataset.load_x(base_path + '.allx')
    ally = np.load(base_path + '.ally', allow_pickle=True)

    testx = PlanetoidDataset.load_x(base_path + '.tx')

    # Add test
    test_idx = list(
        map(int, open(base_path + '.test.index').read().split('\n')[:-1]))

    num_test_examples = max(test_idx) - min(test_idx) + 1
    sparse_zeros = scipy.sparse.csr_matrix((num_test_examples, allx.shape[1]),
                                           dtype='float32')

    allx = scipy.sparse.vstack((allx, sparse_zeros))
    llallx = allx.tolil()
    llallx[test_idx] = testx
    self._allx = as_tensor(np.array(llallx.todense()))

    testy = np.load(base_path + '.ty', allow_pickle=True)
    ally = np.pad(ally, [(0, num_test_examples), (0, 0)], mode='constant')
    ally[test_idx] = testy

    self._num_nodes = len(edge_lists)
    self._num_classes = ally.shape[1]
    self._node_labels = np.argmax(ally, axis=1)
    self._train_labels = self._node_labels + 0  # Copy.
    self._train_labels[test_idx] = -1
    self._node_labels = as_tensor(self._node_labels)
    self._train_labels = as_tensor(self._train_labels)
    self._test_idx = tf.convert_to_tensor(np.array(test_idx, dtype='int32'))
    self._node_split = None  # Populated on `node_split()`

    # Will be used to construct (sparse) adjacency matrix.
    adj_src = []
    adj_target = []
    for node, neighbors in edge_lists.items():
      adj_src.extend([node] * len(neighbors))
      adj_target.extend(neighbors)

    self._edge_list = as_tensor(np.stack([adj_src, adj_target], axis=0))

  @staticmethod
  def load_x(filename):
    if sys.version_info > (3, 0):
      return pickle.load(open(filename, 'rb'), encoding='latin1')
    else:
      return np.load(filename)

  def num_classes(self) -> int:
    return self._num_classes

  def node_features_dicts(self, add_id=True) -> Mapping[
      tfgnn.NodeSetName, MutableMapping[str, tf.Tensor]]:
    features = {'feat': self._allx}
    if add_id:
      features['#id'] = tf.range(self._num_nodes, dtype=tf.int32)
    return {tfgnn.NODES: features}

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    return {tfgnn.NODES: self._num_nodes}

  def edge_lists(self) -> Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
      tf.Tensor]:
    return {(tfgnn.NODES, tfgnn.EDGES, tfgnn.NODES): self._edge_list}

  def node_split(self) -> NodeSplit:
    """Returns dict with keys "train", "valid", "test" to node indices."""
    if self._node_split is None:
      # By default, we mimic Planetoid & GCN setup -- i.e., 20 labels per class.
      labels_per_class = int(os.environ.get('PLANETOID_LABELS_PER_CLASS', '20'))
      num_train_nodes = labels_per_class * self.num_classes()
      num_validate_nodes = 500
      train_ids = tf.range(num_train_nodes, dtype=tf.int32)
      validate_ids = tf.range(
          num_train_nodes, num_train_nodes + num_validate_nodes, dtype=tf.int32)
      self._node_split = NodeSplit(train=train_ids, valid=validate_ids,
                                   test=self._test_idx)
    return self._node_split

  @property
  def labeled_nodeset(self):
    return tfgnn.NODES

  def labels(self) -> tf.Tensor:
    return self._train_labels

  def test_labels(self) -> tf.Tensor:
    """int numpy array of length num_nodes containing train and test labels."""
    return self._node_labels


def get_dataset(dataset_name):
  if dataset_name.startswith('ogbn-'):
    return OgbnDataset(dataset_name)
  elif dataset_name in ('cora', 'citeseer', 'pubmed'):
    return PlanetoidDataset(dataset_name)
  else:
    raise ValueError('Unknown Dataset name: ' + dataset_name)


# Shorthand. Can be replaced with: `as_tensor = tf.convert_to_tensor`.
def as_tensor(obj: Any) -> tf.Tensor:
  """short-hand for tf.convert_to_tensor."""
  return tf.convert_to_tensor(obj)
