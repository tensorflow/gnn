"""Wraps OGB datasets to use within tfgnn.

* class `NodeClassificationOgbDatasetWrapper` can yield GraphTensor instances
  from node classification OGB dataset.

* `create_graph_schema_from_directed` creates `tfgnn.GraphSchema` proto.
"""

import os
from typing import Any, Mapping

import numpy as np
import ogb.nodeproppred
import tensorflow as tf
import tensorflow_gnn as tfgnn


def create_graph_schema_from_directed(
    ogb_dataset: ogb.nodeproppred.NodePropPredDataset,
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
    ogb_dataset: NodePropPredDataset
    make_undirected: If set, only edge with type name 'edges' will be registerd.
      Otherwise, edges with name 'rev_edges' will additionally be registered in
      the schema. If you set this variable, you are expected to also set it in
      `make_full_graph_tensor()`.

  Returns:
    `tfgnn.GraphSchema` describing the node and edge features of the ogbn graph.
  """
  graph, unused_label = ogb_dataset[0]
  # Populate node features.
  schema = tfgnn.GraphSchema()
  node_features = schema.node_sets[tfgnn.NODES]
  node_features.features['#id'].dtype = tf.dtypes.string.as_datatype_enum
  for key, graph_feats in graph.items():
    if graph_feats is None:
      continue
    if key.startswith('node_'):
      feat_name = key[len('node_'):]
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

  def node_features_dict(self) -> Mapping[str, np.ndarray]:
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

  def iterate_once(self, add_self_connections=False, split='train',
                   make_undirected=False) -> tf.data.Dataset:
    """tf.data iterator with one example containg entire graph (full-batch)."""
    graph_tensor = self.export_to_graph_tensor(
        add_self_connections, split, make_undirected)
    spec = graph_tensor.spec

    def once():
      yield graph_tensor

    return tf.data.Dataset.from_generator(once, output_signature=spec)

  def export_to_graph_tensor(self, add_self_connections=False, split='train',
                             make_undirected=False) -> tfgnn.GraphTensor:
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
    node_features_dict['#id'] = _t(all_nodes)
    if split == 'test':
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

    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets, edge_sets=edge_sets,
        context=tfgnn.Context.from_fields(
            features={'seed': _t(self.node_split()[split].reshape(-1))}))

    return graph_tensor


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

  def node_features_dict(self) -> Mapping[str, np.ndarray]:
    return {'feat': self.node_features(), 'year': self._graph['node_year']}

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


# Can be replaced with: `_t = tf.convert_to_tensor`.
def _t(obj: Any) -> tf.Tensor:
  """short-hand for tf.convert_to_tensor."""
  return tf.convert_to_tensor(obj)
