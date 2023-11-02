<!-- lint-g3mark -->

# tfgnn.check_required_features

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_validation.py#L80-L149">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Checks the requirements of a given schema against another.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.check_required_features(
    requirements: <a href="../tfgnn/GraphSchema.md"><code>tfgnn.GraphSchema</code></a>,
    actual: <a href="../tfgnn/GraphSchema.md"><code>tfgnn.GraphSchema</code></a>
)
</code></pre>

<!-- Placeholder for "Used in" -->

This function is used to enable the specification of required features to a
function. A function accepting a `GraphTensor` instance can this way document
what features it is expecting to find on it. The function accepts two schemas: a
`requirements` schema which describes what the function will attempt to fetch
and use on the `GraphTensor`, and an `actual` schema instance, which is the
schema describing the dataset. You can use this in your model code to ensure
that a dataset contains all the expected node sets, edge sets and features that
the model uses.

Note that a dimension with a size of `0` in a feature from the `requirements`
schema is interpreted specially: it means "accept any value for this dimension."
The special value `-1` is still used to represent a ragged dimension.

(Finally, note that this function predates the existence of `GraphTensorSpec`,
which is a runtime descriptor for a `GraphTensor`. We may eventually perovide an
equivalent construct using the \`GraphTensorSpec.)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`requirements`<a id="requirements"></a>
</td>
<td>
An instance of a GraphSchema object, with optional shapes.
</td>
</tr><tr>
<td>
`actual`<a id="actual"></a>
</td>
<td>
The instance of actual schema to check is a matching superset
of the required schema.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValidationError`<a id="ValidationError"></a>
</td>
<td>
If the given schema does not fulfill the requirements.
</td>
</tr>
</table>
