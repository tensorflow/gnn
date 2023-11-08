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


_SEE_PROTOFILE_SUFFIX = """

For detailed documentation, see the comments in the `graph_schema.proto` file.
"""

# Same order as in the proto file.

GraphSchema = graph_schema_pb2.GraphSchema
GraphSchema.__doc__ = """
The top-level container for the schema of a graph dataset.
""" + _SEE_PROTOFILE_SUFFIX

Feature = graph_schema_pb2.Feature
Feature.__doc__ = """\
The schema entry for a single feature.
""" + _SEE_PROTOFILE_SUFFIX

BigQuery = graph_schema_pb2.BigQuery
BigQuery.__doc__ = """
Describes a BigQuery table or SQL statement as datasource of a graph piece.
""" + _SEE_PROTOFILE_SUFFIX

Metadata = graph_schema_pb2.Metadata
Metadata.__doc__ = """
Extra information optionally provided on a context, node set or edge set.
""" + _SEE_PROTOFILE_SUFFIX

Context = graph_schema_pb2.Context
Context.__doc__ = """
The schema for the features that apply across the entire input graph.
""" + _SEE_PROTOFILE_SUFFIX

NodeSet = graph_schema_pb2.NodeSet
NodeSet.__doc__ = """
The schema shared by a set of nodes in the graph.
""" + _SEE_PROTOFILE_SUFFIX

EdgeSet = graph_schema_pb2.EdgeSet
EdgeSet.__doc__ = """
The schema shared by a set of edges that connect the same pair of node sets.
""" + _SEE_PROTOFILE_SUFFIX

GraphType = graph_schema_pb2.GraphType
GraphType.__doc__ = """
An enumeration of graph types according to the method of creation.
""" + _SEE_PROTOFILE_SUFFIX

OriginInfo = graph_schema_pb2.OriginInfo
OriginInfo.__doc__ = """
Metadata about the origin of the graph data.
""" + _SEE_PROTOFILE_SUFFIX
