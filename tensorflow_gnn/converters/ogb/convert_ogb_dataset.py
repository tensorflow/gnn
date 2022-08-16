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
"""Convert an OGB dataset to Unigraph format.

The Unigraph format can be consumed by various other TF-GNN tools. For example,
the TF-GNN graph sampler can sample the Unigraph format.
"""
import math
from os import path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import numpy
import ogb.graphproppred
import ogb.linkproppred
import ogb.nodeproppred
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.converters.ogb.ogb_lib import write_parquet
from tensorflow_gnn.converters.ogb.ogb_lib import write_tfrecords

# Placeholder for Google-internal OGB outputs


FLAGS = flags.FLAGS
Array = numpy.ndarray

DataTable = List[Tuple[str, Array]]


def extract_features(graph: Dict[str, Array],
                     prefix: str,
                     num_items: int) -> DataTable:
  """Extract prefixed features from a dict."""
  matchfn = re.compile("{}_(.*)".format(prefix)).match
  features = []
  logging.info("extract_features num_items: %s", num_items)
  for name in list(graph.keys()):
    array = graph[name]
    logging.info("extract_features %s with shape %s",
                 name, array.shape if array is not None else None)

    match = matchfn(name)
    if not match:
      continue
    array = graph.pop(name)
    if array is None:
      continue

    if array.shape[0] != num_items:
      if array.shape[0] % num_items == 0:
        # Reshape the array if it's a multiple of the number of items (happens
        # in ogbg-molchembl). This feature will be a dense tensor feature.
        new_shape = [num_items, -1]
        logging.info("Reshaping array from %s to %s", array.shape, new_shape)
        assert len(array.shape) == 1, "Invalid rank: {}".format(array.shape)
        array = numpy.reshape(array, [num_items, -1])
      else:
        # Otherwise, the concatenated array of variable-size features should be
        # ragged.
        logging.warning("Array %s has variable-size feature; not reshaping",
                        name)

    feature_name = match.group(1)
    features.append((feature_name, array))
  return features


def extract_features_dict(graph: Dict[str, Dict[str, Array]],
                          set_name: Union[str, Tuple[str, str, str]],
                          prefix: str,
                          num_items: int) -> DataTable:
  """Extract prefixed features from a dict of dicts."""
  matchfn = re.compile("{}_(.*)".format(prefix)).match
  features = []
  for name in list(graph.keys()):
    match = matchfn(name)
    if not match:
      continue
    array_dict = graph.get(name, None)
    if array_dict is None:
      continue
    array = array_dict.pop(set_name, None)
    if array is None:
      continue
    assert array.shape[0] == num_items
    feature_name = match.group(1)
    features.append((feature_name, array))
  return features


def create_schema(
    context_features: Optional[Tuple[str, DataTable]],
    node_features_dict: Dict[str, Tuple[str, DataTable]],
    edge_features_dict: Dict[str, Tuple[str, str, str, DataTable]]
) -> tfgnn.GraphSchema:
  """Create and output a graph schema proto to the given filename."""

  schema = tfgnn.GraphSchema()
  if context_features:
    filename, features = context_features
    context = schema.context
    context.metadata.filename = filename
    for name, array in features:
      feature = context.features[name]
      feature.dtype = tf.dtypes.as_dtype(array.dtype).as_datatype_enum
      if array[0].shape:
        feature.shape.MergeFrom(tf.TensorShape(array[0].shape).as_proto())

    if features:
      _, array = features[0]
      schema.context.metadata.cardinality = array.shape[0]

  for set_name, (filename, node_features) in node_features_dict.items():
    nodes = schema.node_sets[set_name]
    nodes.metadata.filename = filename
    for name, array in node_features:
      feature = nodes.features[name]
      feature.dtype = tf.dtypes.as_dtype(array.dtype).as_datatype_enum
      if array[0].shape:
        feature.shape.MergeFrom(tf.TensorShape(array[0].shape).as_proto())

    if node_features:
      _, array = node_features[0]
      nodes.metadata.cardinality = array.shape[0]

  for set_name, value in edge_features_dict.items():
    (filename, source, target, edge_features) = value
    edges = schema.edge_sets[set_name]
    edges.metadata.filename = filename
    edges.source = source
    edges.target = target
    for name, array in edge_features:
      if name in {tfgnn.SOURCE_NAME, tfgnn.TARGET_NAME}:
        continue
      feature = edges.features[name]
      feature.dtype = tf.dtypes.as_dtype(array.dtype).as_datatype_enum
      if array[0].shape:
        feature.shape.MergeFrom(tf.TensorShape(array[0].shape).as_proto())

    if edge_features:
      _, array = edge_features[0]
      edges.metadata.cardinality = array.shape[0]

  return schema


def remove_empty_dicts(obj: Any) -> Any:
  """Process recursively, if a dict, remove items if the value is empty dict."""
  if isinstance(obj, dict):
    # pylint: disable=g-complex-comprehension,g-explicit-bool-comparison
    return {key: value
            for key, value in obj.items()
            if not ((value is None) or isinstance(value, dict) and not value)}
  else:
    return obj


def write_table(output_dir: str,
                basename: str,
                features: DataTable,
                num_items: int) -> str:
  """Fill up Example protos with each node feature and output to a table."""
  output_format = FLAGS.format
  if output_format == "tfrecords":
    filename = "{}.tfrecords".format(basename)
    writer = write_tfrecords
  elif output_format == "parquet":
    filename = "{}.pq".format(basename)
    writer = write_parquet
  # Placeholder for Google-internal output formats
  else:
    raise ValueError("Invalid format: {}".format(output_format))

  # Compute number of shards and shard filenames.
  total_bytes = sum(array.nbytes
                    for _, array in features
                    if array is not None)
  if total_bytes > FLAGS.target_shard_size:
    num_shards = int(math.ceil(total_bytes / FLAGS.target_shard_size))
    pattern = "{}@{}".format(filename, num_shards)
    filenames = [
        path.join(output_dir,
                  "{}-{:05d}-of-{:05d}".format(filename, shard, num_shards))
        for shard in range(num_shards)]
  else:
    pattern = filename
    filenames = [path.join(output_dir, filename)]

  # Write out the file and return the relative filename.
  writer(filenames, features, num_items)
  return pattern


def convert_homogeneous_graph(graph: Dict[str, Any],
                              num_graphs: int,
                              output_dir: str):
  """Process a homogeneous graph."""

  # NOTE(blais): We could in theory stash the data in the same format as their
  # heterogeneous graphs in Python and just use convert_heterogeneous_graph().

  # Gather node features.
  logging.info("Processing node features")
  num_nodes = graph.pop("num_nodes")
  graph["node_#id"] = numpy.arange(num_nodes).astype(bytes)

  node_features = extract_features(graph, "node", num_nodes)
  filename = write_table(output_dir, "nodes", node_features, num_nodes)
  node_features_dict = {}
  node_features_dict["nodes"] = (filename, node_features)

  # Gather edge features.
  logging.info("Processing edge features")
  indices = graph.pop("edge_index")
  assert len(indices.shape) == 2
  num_edges = indices.shape[1]
  graph["edge_{}".format(tfgnn.SOURCE_NAME)] = indices[0].astype(bytes)
  graph["edge_{}".format(tfgnn.TARGET_NAME)] = indices[1].astype(bytes)

  # NOTE(blais): If external edge features are needed and each edge is
  # unique, you can use this:
  # graph["edge_#id"] = ["{}_{}".format(edge_index[0, i], edge_index[1, i])
  #                      for i in range(num_edges)]
  edge_features = extract_features(graph, "edge", num_edges)
  filename = write_table(output_dir, "edges", edge_features, num_edges)
  edge_features_dict = {}
  edge_features_dict["edges"] = (filename, "nodes", "nodes", edge_features)

  # Gather context features.
  logging.info("Processing graph context features")
  if num_graphs > 1:
    graph_features = extract_features(graph, "graph", num_graphs)
    filename = write_table(output_dir, "graph", graph_features, num_graphs)
    context_features = (filename, graph_features)
  else:
    context_features = None

  # Make sure we processed everything.
  graph = remove_empty_dicts(graph)
  if graph:
    logging.error("Graph is not empty: %s", graph)

  # Produce a corresponding graph schema.
  logging.info("Producing graph schema")
  return create_schema(context_features, node_features_dict, edge_features_dict)


def make_id(set_name: str, node_id: int) -> bytes:
  return "{}{}".format(set_name, node_id).encode("ascii")


def convert_heterogeneous_graph(graph: Dict[str, Dict[str, Any]],
                                output_dir: str):
  """Process a heterogeneous graph."""

  # Translate feature names.
  graph["node_feat"] = graph.pop("node_feat_dict")
  graph["edge_feat"] = graph.pop("edge_feat_dict")

  # Gather node features.
  logging.info("Processing node features.")
  num_nodes_dict = graph.pop("num_nodes_dict")
  graph["node_#id"] = {key: numpy.array([make_id(key, idd)
                                         for idd in range(num_items)])
                       for key, num_items in num_nodes_dict.items()}

  node_features_dict = {}
  for set_name, num_nodes in num_nodes_dict.items():
    logging.info("Processing nodes: %s", set_name)
    node_features = extract_features_dict(graph, set_name, "node", num_nodes)
    filename = write_table(output_dir, "nodes-{}".format(set_name),
                           node_features, num_nodes)
    node_features_dict[set_name] = (filename, node_features)

  # Gather edge features.
  logging.info("Processing edge features.")
  edge_index_dict = graph.pop("edge_index_dict")
  assert all(len(indices.shape) == 2 and indices.shape[0] == 2
             for indices in edge_index_dict.values())
  num_edges_dict = {key: indices.shape[1]
                    for key, indices in edge_index_dict.items()}

  # Remove the useless 'reltype' feature and ensure that those arrays always
  # contain a single value.
  reltype = graph.pop("edge_reltype", {})
  for key, array in reltype.items():
    unique_shape = numpy.unique(array).shape
    assert unique_shape == (1,), unique_shape

  sources = graph["edge_{}".format(tfgnn.SOURCE_NAME)] = {}
  targets = graph["edge_{}".format(tfgnn.TARGET_NAME)] = {}
  for key, indices in edge_index_dict.items():
    (source, set_name, target) = key
    sources[key] = numpy.array([make_id(source, idd) for idd in indices[0]])
    targets[key] = numpy.array([make_id(target, idd) for idd in indices[1]])
  del sources
  del targets

  edge_features_dict = {}
  for key, indices in edge_index_dict.items():
    (source, set_name, target) = key
    logging.info("Processing edges: %s", key)
    num_edges = num_edges_dict[key]
    edge_features = extract_features_dict(graph, key, "edge", num_edges)
    filename = write_table(output_dir, "edges-{}".format(set_name),
                           edge_features, num_edges)
    edge_features_dict[set_name] = (filename, source, target, edge_features)

  # NOTE(blais): Not needed in any of the datasets so far.
  # # Gather context features.
  # logging.info("Processing graph context features")
  # if num_graphs > 1:
  #   graph_features = extract_features(graph, "graph", num_graphs)
  #   write_table(output_dir, "graph", graph_features, num_graphs)
  context_features = None

  # Make sure we processed everything.
  graph = remove_empty_dicts(graph)
  if graph:
    logging.error("Graph is not empty: %s", graph)

  # # Produce a corresponding graph schema.
  # logging.info("Producing graph schema")
  return create_schema(context_features, node_features_dict, edge_features_dict)


def concatenate_graphs(dataset: Any) -> Dict[str, Array]:
  """Concatenate all the graphs from a dataset into a single graph."""

  # Important: The nodes are allocated unique ids into the joint graph; and edge
  # indices are offset accordingly.

  # Build up accumulator.
  graph, labels = dataset[0]
  accumulator = {key: [] for key in graph.keys()}
  accumulator["node_graph"] = []
  accumulator["edge_graph"] = []
  accumulator["graph_label"] = []
  accumulator["graph_graph"] = []

  # We will insert a graph id as a separate column over all the features.
  count = 0
  for index, (graph, labels) in enumerate(dataset):
    indices = graph["edge_index"]
    num_nodes = graph["num_nodes"]
    graph["edge_index"] = numpy.transpose(indices) + num_nodes
    graph["node_graph"] = numpy.full(num_nodes, index)
    graph["edge_graph"] = numpy.full(indices.shape[1], index)
    graph["graph_label"] = labels
    graph["graph_graph"] = numpy.array([index], dtype=numpy.int64)
    graph["num_nodes"] = numpy.array([num_nodes])
    count += num_nodes

    for key, array_list in accumulator.items():
      value = graph[key]
      if value is None:
        continue
      array_list.append(value)

  # Concatenate all the accumulators.
  concat_graph = {}
  for key, array_list in accumulator.items():
    if not array_list:
      continue
    concat_graph[key] = numpy.concatenate(array_list, axis=0)
  concat_graph["num_nodes"] = numpy.sum(concat_graph["num_nodes"])
  concat_graph["edge_index"] = numpy.transpose(concat_graph["edge_index"])

  return concat_graph


def convert_dataset(dataset: Any, output_dir: str):
  """Convert a node prediction problem."""

  # Write out the metadata to a file.
  logging.info("Write out metadata")
  meta_filename = path.join(output_dir, "meta_info.csv")
  with tf.io.gfile.GFile(meta_filename, "w") as outfile:
    dataset.meta_info.to_csv(outfile)

  # Insert labels in the graph, so they get processed as node features.
  num_graphs = len(dataset)
  if num_graphs > 1:
    graph_labels = concatenate_graphs(dataset)
  else:
    graph_labels = dataset[0]

  if isinstance(graph_labels, tuple):
    graph, labels = graph_labels
    graph["node_labels"] = labels
  else:
    assert isinstance(graph_labels, dict)
    graph = graph_labels

  is_heterogeneous = dataset.meta_info.get("is hetero", False)
  if isinstance(is_heterogeneous, str):
    is_heterogeneous = (is_heterogeneous == "True")
  if is_heterogeneous:
    schema = convert_heterogeneous_graph(graph, output_dir)
  else:
    schema = convert_homogeneous_graph(graph, len(dataset), output_dir)

  # Write out the schema produced by the conversion.
  with tf.io.gfile.GFile(path.join(output_dir, "schema.pbtxt"), "w") as outfile:
    print(schema, file=outfile)


def arrays_to_shape(data: Any) -> Any:
  """Convert arrays to their shape, recursively."""
  if isinstance(data, Array):
    return data.shape
  elif isinstance(data, dict):
    return {key: arrays_to_shape(value) for key, value in data.items()}
  elif isinstance(data, tuple):
    return data.__class__(*map(arrays_to_shape, data))
  elif isinstance(data, list):
    return list(map(arrays_to_shape, data))
  else:
    return data


def create_dataset(dataset: str, datasets_root: Optional[str] = None) -> Any:
  """Create the dataset to process."""
  problem_type = dataset.split("-")[0]
  kwargs = dict(name=dataset, root=datasets_root)
  if problem_type == "ogbn":
    dataset = ogb.nodeproppred.NodePropPredDataset(**kwargs)
  elif problem_type == "ogbl":
    dataset = ogb.linkproppred.LinkPropPredDataset(**kwargs)
  elif problem_type == "ogbg":
    dataset = ogb.graphproppred.GraphPropPredDataset(**kwargs)
  else:
    raise ValueError("Invalid problem type for {}".format(FLAGS.dataset))
  return dataset


def define_flags():
  """Define the program flags."""

  flags.DEFINE_string("dataset", None,
                      "Dataset name, e.g. 'ogbn-arxiv'.")

  flags.DEFINE_string("output", None,
                      "Output directory for all the files produced.")

  flags.DEFINE_string("ogb_datasets_dir", "/tmp/ogb-preprocessed",
                      "Root directory for preprocessed downloaded datasets "
                      "cache (default '/tmp/data/ogb-preprocessed')).")

  flags.DEFINE_string("format", "tfrecords",
                      "Output file format.")

  flags.DEFINE_integer("target_shard_size", 10**6,
                       "The target shard size. If a table is under that size, "
                       "put it in a single shard.")

  flags.mark_flags_as_required(["dataset", "output"])


def app_main(unused_argv):
  """App runner main function."""

  # Ensure the output directory exists.
  output = FLAGS.output
  if not tf.io.gfile.exists(output):
    tf.io.gfile.makedirs(output)

  # Create and convert the dataset.
  dataset = create_dataset(FLAGS.dataset, FLAGS.ogb_datasets_dir)
  convert_dataset(dataset, output)


def main():
  define_flags()
  app.run(app_main)


if __name__ == "__main__":
  main()
