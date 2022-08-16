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
"""Module that provides docstrings for protos.

This is used to accompany the documentation generation and provide text for the
type on the website.
"""

from tensorflow_gnn.proto import graph_schema_pb2


GraphSchema = graph_schema_pb2.GraphSchema
GraphSchema.__doc__ = """\
A schema definition for graphs.

The `GraphSchema` message describes the sets of nodes and edges provided by the
graph data structure. It also provides the lists of available features within
each node set, each edge set and over each graph ("context" features).

The purpose of this schema is to describe the tensors available in a graph data
structure container and the associations between them, their data types, shapes
and feature descriptions. It exists to allow the user to declare the topology of
the graph, and for the I/O routines to automatically parse the data. It also
contains metadata about the graph's various features. See the "Describing your
Graph" section of the documentation for full details on how to create an
instance of this message to define the shape and encoding of your dataset.

Note that the schema does not provide a full definition for how the features are
intended to be used or combined; that belongs to a separate description for a
model, which is beyond the scope of this schema object. For instance, whether a
context feature is broadcast over a particular node feature during learning is
information that isn't related to the data stored in the container of features.

Intended usage:

* To accompany a graph container data structure, as documentation reporting
  entities, edges and features available during training.
* To be serialized in the metadata of training data files.
* To be safeguarded along with model checkpoints in order to keep track of input
  features used historically.
* To be utilized to automatically infer good default models.

Note that a feature names beginnning with `#` are explicitly reserved and
disallowed. (These are used in serialization.)
"""


Feature = graph_schema_pb2.Feature
Feature.__doc__ = """\
A schema for a single feature.

This proto message contains the description, shape, data type and some more
fields about a feature in the schema.
"""
