# tfgnn.Feature

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



A schema for a single feature.

<!-- Placeholder for "Used in" -->

This proto message contains the description, shape, data type and some more
fields about a feature in the schema.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`description`
</td>
<td>
`string description`
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
`DataType dtype`
</td>
</tr><tr>
<td>
`example_values`
</td>
<td>
`repeated Feature example_values`
</td>
</tr><tr>
<td>
`sample_values`
</td>
<td>
`Feature sample_values`
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
`TensorShapeProto shape`
</td>
</tr><tr>
<td>
`source`
</td>
<td>
`string source`
</td>
</tr>
</table>



