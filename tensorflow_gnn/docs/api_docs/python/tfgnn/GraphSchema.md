<!-- lint-g3mark -->

# tfgnn.GraphSchema

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/proto/graph_schema.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A schema definition for graphs.

<!-- Placeholder for "Used in" -->

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

#### Intended usage:

  - To accompany a graph container data structure, as documentation reporting
    entities, edges and features available during training.
  - To be serialized in the metadata of training data files.
  - To be safeguarded along with model checkpoints in order to keep track of
    input features used historically.
  - To be utilized to automatically infer good default models.

Note that a feature names beginnning with `#` are explicitly reserved and
disallowed. (These are used in serialization.)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`context`<a id="context"></a>
</td>
<td>
`Context context`
</td>
</tr><tr>
<td>
`edge_sets`<a id="edge_sets"></a>
</td>
<td>
`repeated EdgeSetsEntry edge_sets`
</td>
</tr><tr>
<td>
`info`<a id="info"></a>
</td>
<td>
`OriginInfo info`
</td>
</tr><tr>
<td>
`node_sets`<a id="node_sets"></a>
</td>
<td>
`repeated NodeSetsEntry node_sets`
</td>
</tr>
</table>
