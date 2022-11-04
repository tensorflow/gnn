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
"""Infrastructure and implementation of in-memory dataset.

Abstract classes:

  * `Dataset`: provides nodes, edges, and features, for a heteregenous graph.
  * `NodeClassificationDataset`: a `Dataset` that also provides list of
    {train, test, validate} nodes, as well as their labels.
  * `LinkPredictionDataset`: a `Dataset` that also provides lists of edges for
    {train, test, validate}.


All `Dataset` implementations automatically inherit abilities of:

  * `as_graph_tensor()` which constructs `GraphTensor` holding entire graph.
  * `graph_schema()` returning `GraphSchema` describing `GraphTensor` above.
  * More importantly, they can be plugged-into training pipelines, e.g., for
    node classification (see `tf_trainer.py` and `keras_trainer.py`).
  * In addition, they can be plugged-into in-memory sampling (see
    `int_arithmetic_sampler.py`, and example trainer script,
    `keras_minibatch_trainer.py`).


Concrete implementations:

  * Node classification (inheriting `NodeClassificationDataset`)

    * `OgbnDataset`: Wraps node classification datasets from OGB, i.e., with
      name prefix of "ogbn-", such as, "ogbn-arxiv".

    * `PlanetoidDataset`: wraps datasets that are popularized by GCN paper
      (cora, citeseer, pubmed).

  * Link Prediction (inherting `LinkPredictionDataset`)

    * `OgblDataset`: Wraps link-prediction datasets from OGB, i.e., with name
      prefix of "ogbl-", such as, "ogbl-citation2".
"""
import collections
import functools
import os
import pickle
import sys
from typing import Any, Dict, List, Mapping, MutableMapping, NamedTuple, Optional, Tuple, Union
import urllib.request

from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.interactive import interactive_beam as ib
import numpy as np
import ogb.linkproppred
import ogb.nodeproppred
import scipy
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph

Example = tf.train.Example


class Dataset:
  """Abstract class for hold a dataset in-memory."""

  def graph_schema(
      self, make_undirected: bool = False) -> tfgnn.GraphSchema:
    raise NotImplementedError()

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

  def node_sets(self, node_features_dicts_fn=None) -> MutableMapping[
      tfgnn.NodeSetName, tfgnn.NodeSet]:
    """Returns node sets of entire graph (dict: node set name -> NodeSet)."""
    node_features_dicts_fn = node_features_dicts_fn or self.node_features_dicts
    node_counts = self.node_counts()
    node_features_dicts = node_features_dicts_fn()

    node_sets = {}
    for node_set_name, node_features_dict in node_features_dicts.items():
      node_sets[node_set_name] = tfgnn.NodeSet.from_fields(
          sizes=as_tensor([node_counts[node_set_name]]),
          features=node_features_dict)
    return node_sets

  def edge_sets(
      self, add_self_connections: bool = False,
      make_undirected: bool = False) -> MutableMapping[
          tfgnn.EdgeSetName, tfgnn.EdgeSet]:
    """Returns edge sets of entire graph (dict: edge set name -> EdgeSet)."""
    edge_sets = {}
    node_counts = self.node_counts() if add_self_connections else None
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
    return edge_sets


class NodeSplit(NamedTuple):
  """Contains 1D int tensors holding positions of {train, valid, test} nodes.

  This is returned by `NodeClassificationDataset.node_split()`
  """
  train: tf.Tensor
  valid: tf.Tensor
  test: tf.Tensor


class NodeClassificationDataset(Dataset):
  """Wraps graph datasets (nodes, edges, features).

  Inheriting classes implement straight-forward functions to adapt any external
  dataset into TFGNN, by exposing methods `iterate_once` and `as_graph_tensor`
  that yield GraphTensor objects that can be passed to TFGNN's models.
  """

  def num_classes(self) -> int:
    """Number of node classes. Max of `labels` should be `< num_classes`."""
    raise NotImplementedError('num_classes')

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
    graph_tensor = self.as_graph_tensor(
        add_self_connections, split, make_undirected)
    spec = graph_tensor.spec

    def once():
      yield graph_tensor

    return tf.data.Dataset.from_generator(once, output_signature=spec)

  def node_feature_dicts_with_labels(
      self, split: Union[str, List[str]] = 'train') -> Mapping[
          tfgnn.NodeSetName, MutableMapping[str, tf.Tensor]]:
    node_features_dicts = self.node_features_dicts()
    splits = split if isinstance(split, (tuple, list)) else [split]
    if 'test' in splits:
      node_features_dicts[self.labeled_nodeset]['label'] = self.test_labels()
    else:
      node_features_dicts[self.labeled_nodeset]['label'] = self.labels()
    return node_features_dicts

  def as_graph_tensor(
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
    # Node and edge sets.
    node_sets = self.node_sets(functools.partial(
        self.node_feature_dicts_with_labels, split=split))
    edge_sets = self.edge_sets(add_self_connections=add_self_connections,
                               make_undirected=make_undirected)

    # Context.
    splits = split if isinstance(split, (tuple, list)) else [split]
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

  def graph_schema(
      self, make_undirected: bool = False) -> tfgnn.GraphSchema:
    graph_schema = _create_graph_schema_from_directed(
        self, make_undirected=make_undirected)
    context_features = graph_schema.context.features
    context_features['seed_nodes.' + self.labeled_nodeset].dtype = (
        tf.int64.as_datatype_enum)
    return graph_schema


class LinkPredictionDataset(Dataset):
  """Superclasses must wrap dataset of graph(s) for link-prediction tasks."""

  def as_graph_tensor(
      self, add_self_connections: bool = False,
      make_undirected: bool = False) -> tfgnn.GraphTensor:
    node_sets = self.node_sets()
    edge_sets = self.edge_sets(add_self_connections=add_self_connections,
                               make_undirected=make_undirected)
    return tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets, edge_sets=edge_sets)

  def graph_schema(
      self, make_undirected: bool = False) -> tfgnn.GraphSchema:
    return _create_graph_schema_from_directed(
        self, make_undirected=make_undirected)

  def edge_split(self):
    raise NotImplementedError()


def _create_graph_schema_from_directed(
    dataset: Dataset, make_undirected=False) -> tfgnn.GraphSchema:
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
    dataset: NodeClassificationDataset. Feature shapes and types returned
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

  return schema


class _OgbGraph:
  """Wraps data exposed by OGB graph objects, while enforcing heterogeneity."""

  @property
  def num_nodes_dict(self) -> Mapping[str, int]:
    """Maps "node set name" -> number of nodes."""
    return self._num_nodes_dict

  @property
  def node_feat_dict(self) -> Mapping[str, MutableMapping[str, tf.Tensor]]:
    """Maps "node set name" to dict of "feature name"->tf.Tensor."""
    return self._node_feat_dict

  @property
  def edge_index_dict(self) -> Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
      tf.Tensor]:
    """Adjacency lists for all edge sets.

    Returns:
      Dict (source node set name, edge set name, target node set name) -> edges.
      Where `edges` is tf.Tensor of shape (2, num edges), with `edges[0]` and
      `edges[1]`, respectively, containing source and target node IDs (as 1D int
      tf.Tensor).
    """
    return self._edge_index_dict

  def __init__(self, graph: Mapping[str, Any]):
    if 'edge_index_dict' in graph:  # Heterogeneous graph
      assert 'num_nodes_dict' in graph
      assert 'node_feat_dict' in graph

      # node set name -> feature name -> feature matrix (numNodes x featDim).
      node_set = {node_set_name: {'feat': as_tensor(feat)}
                  for node_set_name, feat in graph['node_feat_dict'].items()
                  if feat is not None}
      # Populate remaining features
      for key, node_set_name_to_feat in graph.items():
        if key.startswith('node_') and key != 'node_feat_dict':
          feat_name = key.split('node_', 1)[-1]
          for node_set_name, feat in node_set_name_to_feat.items():
            node_set[node_set_name][feat_name] = as_tensor(feat)
      self._num_nodes_dict = graph['num_nodes_dict']
      self._node_feat_dict = node_set
      self._edge_index_dict = tf.nest.map_structure(
          as_tensor, graph['edge_index_dict'])
    else:  # Homogenous graph. Make heterogeneous.
      if graph.get('node_feat', None) is not None:
        node_features = {
            tfgnn.NODES: {'feat': as_tensor(graph['node_feat'])}
        }
      else:
        node_features = {
            tfgnn.NODES: {
                'feat': tf.zeros([graph['num_nodes'], 0], dtype=tf.float32)
            }
        }

      self._edge_index_dict = {
          (tfgnn.NODES, tfgnn.EDGES, tfgnn.NODES): as_tensor(
              graph['edge_index']),
      }
      self._num_nodes_dict = {tfgnn.NODES: graph['num_nodes']}
      self._node_feat_dict = node_features


class OgblDataset(LinkPredictionDataset):
  """Wraps link prediction datasets of ogbl-* for in-memory learning."""

  def __init__(self, dataset_name, cache_dir=None):
    if cache_dir is None:
      cache_dir = os.environ.get(
          'OGB_CACHE_DIR', os.path.expanduser(os.path.join('~', 'data', 'ogb')))

    self.ogb_dataset = ogb.linkproppred.LinkPropPredDataset(
        dataset_name, root=cache_dir)

    # dict with keys 'train', 'valid', 'test'
    self._edge_split = self.ogb_dataset.get_edge_split()
    self.ogb_graph = _OgbGraph(self.ogb_dataset.graph)

  def node_features_dicts(self, add_id=True) -> Mapping[
      tfgnn.NodeSetName, MutableMapping[str, tf.Tensor]]:
    features = self.ogb_graph.node_feat_dict
    features = {node_set_name: {feat: value for feat, value in features.items()}
                for node_set_name, features in features.items()}
    if add_id:
      counts = self.node_counts()
      for node_set_name, feats in features.items():
        feats['#id'] = tf.range(counts[node_set_name], dtype=tf.int32)
    return features

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    return self.ogb_graph.num_nodes_dict

  def edge_lists(self) -> Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
      tf.Tensor]:
    return self.ogb_graph.edge_index_dict

  def edge_split(self):
    return self._edge_split


class OgbnDataset(NodeClassificationDataset):
  """Wraps node classification datasets of ogbn-* for in-memory learning."""

  def __init__(self, dataset_name, cache_dir=None):
    if cache_dir is None:
      cache_dir = os.environ.get(
          'OGB_CACHE_DIR', os.path.expanduser(os.path.join('~', 'data', 'ogb')))

    self.ogb_dataset = ogb.nodeproppred.NodePropPredDataset(
        dataset_name, root=cache_dir)
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
          _OgbGraph,    # ogb_graph.
          np.ndarray,   # node_labels.
          NodeSplit,    # idx_split.
          str]:
    """Returns heterogeneous dicts from homogenous or heterogeneous dataset.

    Args:
      ogb_dataset: OGBN dataset. It can be homogeneous (single node set type,
        single edge set type), or heterogeneous (various node/edge set types),
        and returns data structure as-if the dataset is heterogeneous (i.e.,
        names each node/edge set). If input is a homogeneous graph, then the
        node set will be named "nodes" and the edge set will be named "edges".

    Returns:
      tuple: `(ogb_graph, node_labels, idx_split, labeled_nodeset)`, where:
        `ogb_graph` is instance of _OgbGraph.
        `node_labels`: np.array of labels, with .shape[0] equals number of nodes
          in node set with name `labeled_nodeset`.
        `idx_split`: instance of NodeSplit. Members `train`, `test` and `valid`,
          respectively, contain indices of nodes in node set with name
          `labeled_nodeset`.
        `labeled_nodeset`: name of node set that the node-classification task is
          designed over.
    """
    graph, node_labels = ogb_dataset[0]
    ogb_graph = _OgbGraph(graph)
    if 'edge_index_dict' in graph:  # Graph is heterogeneous
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

      return ogb_graph, node_labels, idx_split, labeled_nodeset

    # Copy other node information.
    for key, value in graph.items():
      if key != 'node_feat' and key.startswith('node_'):
        key = key.split('node_', 1)[-1]
        ogb_graph.node_feat_dict[tfgnn.NODES][key] = as_tensor(value)
    idx_split = NodeSplit(**tf.nest.map_structure(
        tf.convert_to_tensor, ogb_dataset.get_idx_split()))
    return ogb_graph, node_labels, idx_split, tfgnn.NODES

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


class PlanetoidDataset(NodeClassificationDataset):
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


def get_dataset(dataset_name) -> Dataset:
  if dataset_name.startswith('ogbn-'):
    return OgbnDataset(dataset_name)
  elif dataset_name.startswith('ogbl-'):
    return OgblDataset(dataset_name)
  elif dataset_name in ('cora', 'citeseer', 'pubmed'):
    return PlanetoidDataset(dataset_name)
  else:
    raise ValueError('Unknown Dataset name: ' + dataset_name)


# Shorthand. Can be replaced with: `as_tensor = tf.convert_to_tensor`.
def as_tensor(obj: Any) -> tf.Tensor:
  """short-hand for tf.convert_to_tensor."""
  return tf.convert_to_tensor(obj)


# Copied from cs/third_party/py/tensorflow_gnn/graph/graph_tensor_random.py
def _get_feature_values(feature: tf.train.Feature) -> Union[List[str],
                                                            List[int],
                                                            List[float]]:
  """Return the values from a TF feature proto."""
  if feature.HasField('float_list'):
    return list(feature.float_list.value)
  elif feature.HasField('int64_list'):
    return list(feature.int64_list.value)
  elif feature.HasField('bytes_list'):
    return list(feature.bytes_list.value)
  return []


class UnigraphInMemeoryDataset(Dataset):
  """Implementation of in-memory dataset loader for unigraph format."""

  def __init__(self,
               graph_schema: tfgnn.GraphSchema,
               graph_directory: Optional[str] = None,
               pipeline_options: Optional[PipelineOptions] = None,
               autocompress=True):
    """Constructor to represent a unigraph in memory.

    Args:
      graph_schema: A tfgnn.GraphSchema protobuf message.
      graph_directory: Optional graph directory if the graph_schema specifies
        relative paths.
      pipeline_options: Optional beam pipeline options that can be passed to the
        interactive runner. `pipeline_options` will probably not be needed.
      autocompress: If set (default), populates tf.Tensors that are needed for
        model training and sampling, keeping not the intermediate data
        structures. Once graph is compressed, `.get_adjacency_list()` and
        `.node_features` can no longer be accessed. Constructing with
        `autocompress=False` then calling `.compress()` is equivalent to setting
        `autocompress=True`.
    """
    self._graph_schema = graph_schema
    self.graph_directory = graph_directory
    self.pipeline_options = pipeline_options

    # Node Set Name -> Node ID ->  auto-incrementing int (`node_idx`)`.
    self.compression_maps: Dict[
        tfgnn.NodeSetName, Dict[bytes, int]] = collections.defaultdict(dict)
    # Node Set Name -> list of Node ID (from above) sorted per int (`node_idx`).
    self.rev_compression_maps: Dict[
        tfgnn.NodeSetName, List[bytes]] = collections.defaultdict(list)

    # Mapping from node set name to a mapping of node id to tf.train.Example
    # pairs.
    self.node_features: Dict[str, Dict[bytes, tf.train.Example]] = {}

    # Mapping from an edge set name to a list of [src, target] pairs
    self.flat_edge_list: Dict[str, List[Tuple[str, str, tf.train.Example]]] = {}

    with beam.Pipeline(
        runner='InteractiveRunner', options=pipeline_options) as p:
      graph_pcoll: Dict[str, Dict[str, beam.PCollection]] = unigraph.read_graph(
          graph_schema, self.graph_directory, p)
      for node_set_name, ns in graph_pcoll[tfgnn.NODES].items():
        self.node_features[node_set_name] = {}
        df = ib.collect(ns)
        for node_order, (node_id, example) in enumerate(zip(df[0], df[1])):
          if node_id in self.node_features[node_set_name]:
            raise ValueError('More than one node with ID %s' % node_id)
          self.node_features[node_set_name][node_id] = example
          self.compression_maps[node_set_name][node_id] = node_order
          self.rev_compression_maps[node_set_name].append(node_id)

      for edge_set_name, es in graph_pcoll[tfgnn.EDGES].items():
        self.flat_edge_list[edge_set_name] = []
        df = ib.collect(es)
        for src, target, example in zip(df[0], df[1], df[2]):
          self.flat_edge_list[edge_set_name].append((src, target, example))

    self._node_features_dict: Dict[tfgnn.NodeSetName, Dict[str, tf.Tensor]] = {}
    self._edge_lists: Dict[
        Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
        tf.Tensor] = {}

    if autocompress:
      self.compress()

  def compress(self, cleanup=False):
    """Creates compression map from nodes to store edge endpoints as ints.

    Calling this enables functions `.edge_lists()` and `.node_features_dicts()`.

    Args:
      cleanup: If set, data structures will be removed, making function
      `.adjacency()` and member `.node_features` return empty results.
    """
    schema = self.graph_schema()
    # Node set name -> feature name -> feature tensor.
    # All features under a node set must have same `feature_tensor.shape[0]`.
    node_features_dict: Dict[
        tfgnn.NodeSetName, Dict[str, List[np.ndarray]]] = {}
    node_features_dict = collections.defaultdict(
        lambda: collections.defaultdict(list))

    for node_set_name, node_order in self.rev_compression_maps.items():
      feature_schema = schema.node_sets[node_set_name]
      for node_id in node_order:
        example = self.node_features[node_set_name][node_id]
        for feature_name, feature_value in example.features.feature.items():
          np_feature_value = np.array(_get_feature_values(feature_value))
          np_feature_value = np_feature_value.reshape(
              feature_schema.features[feature_name].shape.dim)
          node_features_dict[node_set_name][feature_name].append(
              np_feature_value)

    self._node_features_dict = {}
    for node_set_name, feature_dict in node_features_dict.items():
      self._node_features_dict[node_set_name] = {}
      for feature_name, list_np_feature_values in feature_dict.items():
        feature_tensor = tf.convert_to_tensor(
            np.stack(list_np_feature_values, axis=0))
        self._node_features_dict[node_set_name][feature_name] = feature_tensor

    edge_lists: Dict[
        Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
        List[np.ndarray]] = collections.defaultdict(list)
    for edge_set_name, connection in self.flat_edge_list.items():
      source_node_set_name = schema.edge_sets[edge_set_name].source
      target_node_set_name = schema.edge_sets[edge_set_name].target
      edge_key = (source_node_set_name, edge_set_name, target_node_set_name)
      #
      for source_id, target_id, example in connection:
        for feature_name, feature_value in example.features.feature.items():
          if feature_name in ('#source', '#target'):
            continue
          logging.warn('Ignoring all edge features, including feature (%s) for '
                       'edge set (%s)', feature_name, edge_set_name)
        edge_endpoints = (
            self._compression_id(source_node_set_name, source_id),
            self._compression_id(target_node_set_name, target_id))
        edge_lists[edge_key].append(np.array(edge_endpoints))

    # Mapping from an edge set name to a list of [src, target] pairs
    self._edge_lists = {}
    for edge_key, list_np_edge_list in edge_lists.items():
      self._edge_lists[edge_key] = tf.convert_to_tensor(
          np.stack(list_np_edge_list, -1))

    # TODO(haija): Ensure that all features are populated, in case
    # self._compression_id, when assembling `edge_endpoints`, has invented new
    # nodes. In this case, we should pad with zeros. Ask bmayer@ if needed.

    if cleanup:
      self.flat_edge_list = {}
      self.node_features = {}

  def _compression_id(self, node_set_name, node_id):
    if node_id not in self.compression_maps[node_set_name]:
      next_int = len(self.compression_maps[node_set_name])
      self.compression_maps[node_set_name][node_id] = next_int
      self.rev_compression_maps[node_set_name].append(node_id)

    return self.compression_maps[node_set_name][node_id]

  def get_adjacency_list(self) -> Dict[tfgnn.EdgeSetName, Dict[str, Example]]:
    """Returns weighted edges as an adjacency list of nested dictionaries.

    This function is useful for testing for fast access to edge features based
    on (source, target) IDs. Beware, this function will create an object that
    may increase memory usage.
    """
    adjacency_sets = collections.defaultdict(dict)
    for edge_set_name, flat_edge_list in self.flat_edge_list.items():
      adjacency_sets[edge_set_name] = collections.defaultdict(dict)
      for source, target, example in flat_edge_list:
        adjacency_sets[edge_set_name][source][target] = example

    return adjacency_sets

  def graph_schema(self, make_undirected: bool = False) -> tfgnn.GraphSchema:
    return self._graph_schema

  def node_features_dicts(self, add_id=True) -> Mapping[
      tfgnn.NodeSetName, MutableMapping[str, tf.Tensor]]:
    del add_id  # Features should already have '#id' field, with dtype string.
    return self._node_features_dict

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    """Returns total number of graph nodes per node set."""
    return {node_set_name: len(ids)
            for node_set_name, ids in self.rev_compression_maps.items()}

  def edge_lists(self) -> Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
      tf.Tensor]:
    """Returns dict from "edge type tuple" to int array of shape (2, num_edges).

    "edge type tuple" string-tuple: (src_node_set, edge_set, target_node_set).
    """
    return self._edge_lists

  def as_graph_tensor(
      self, add_self_connections: bool = False,
      make_undirected: bool = False) -> tfgnn.GraphTensor:
    node_sets = self.node_sets()
    edge_sets = self.edge_sets(add_self_connections=add_self_connections,
                               make_undirected=make_undirected)
    return tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets, edge_sets=edge_sets)

