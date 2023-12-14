# Module: tfgnn.sampler

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/sampler/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Public interface for GNN Sampler.

## Classes

[`class SamplingOp`](../tfgnn/sampler/SamplingOp.md): A ProtocolMessage

[`class SamplingSpec`](../tfgnn/sampler/SamplingSpec.md): A ProtocolMessage

[`class SamplingSpecBuilder`](../tfgnn/sampler/SamplingSpecBuilder.md): Mimics
builder pattern that eases creation of `tfgnn.SamplingSpec`.

## Functions

[`make_sampling_spec_tree(...)`](../tfgnn/sampler/make_sampling_spec_tree.md):
Automatically creates `SamplingSpec` by starting from seed node set.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
SamplingStrategy<a id="SamplingStrategy"></a>
</td>
<td>
['TOP_K', 'RANDOM_UNIFORM', 'RANDOM_WEIGHTED']
</td>
</tr>
</table>
