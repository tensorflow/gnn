<!-- lint-g3mark -->

# tfgnn.random_graph_tensor

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_random.py#L144-L243">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Generate a graph tensor from a schema, with random features.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.random_graph_tensor(
    spec: <a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>,
    sample_dict: Optional[SampleDict] = None,
    row_lengths_range: Tuple[int, int] = (2, 8),
    validate: bool = True
) -> <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

NOTE: This function does not (yet?) support the generation of the auxiliary node
set for
<a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>.
It should not be included in the `spec`, and if needed, should be added
separately in a later step.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`spec`<a id="spec"></a>
</td>
<td>
A GraphTensorSpec instance that describes the graph tensor.
</td>
</tr><tr>
<td>
`sample_dict`<a id="sample_dict"></a>
</td>
<td>
A dict of (set-type, set-name, field-name) to list-of-values to
sample from. The intended purpose is to generate random values that are
more realistic, more representative of what the actual dataset will
contain. You can provide such if the values aren't provided for a feature,
random features are inserted of the right type.
</td>
</tr><tr>
<td>
`row_lengths_range`<a id="row_lengths_range"></a>
</td>
<td>
Minimum and maximum values for each row lengths in a
ragged range.
</td>
</tr><tr>
<td>
`validate`<a id="validate"></a>
</td>
<td>
If true, then use assertions to check that the arguments form a
valid RaggedTensor. Note: these assertions incur a runtime cost, since
they must be checked for each tensor value.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance of a GraphTensor.
</td>
</tr>

</table>
