<!-- lint-g3mark -->

# contrastive_losses.CorruptionSpec

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/layers.py#L36-L82">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Class for defining corruption specification.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>contrastive_losses.CorruptionSpec(
    node_set_corruption: NodeCorruptionSpec = dataclasses.field(default_factory=dict),
    edge_set_corruption: EdgeCorruptionSpec = dataclasses.field(default_factory=dict),
    context_corruption: ContextCorruptionSpec = dataclasses.field(default_factory=dict)
)
</code></pre>

<!-- Placeholder for "Used in" -->

This has three fields for specifying the corruption behavior of node-, edge-,
and context sets.

A value of the key "\*" is a wildcard value that is used for either all features
or all node/edge sets.

#### Some example usages:

Want: corrupt everything with parameter 1.0. Solution: either set default to 1.0
or set all corruption specs to `{"*": 1.}`.

Want: corrupt all context features with parameter 1.0 except for "feat", which
should not be corrupted. Solution: set `context_corruption` to `{"feat": 0.,
"*": 1.}`

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`node_set_corruption`<a id="node_set_corruption"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`edge_set_corruption`<a id="edge_set_corruption"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`context_corruption`<a id="context_corruption"></a>
</td>
<td>
Dataclass field
</td>
</tr>
</table>

## Methods

<h3 id="with_default"><code>with_default</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/layers.py#L67-L82">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_default(
    default: T
)
</code></pre>

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.
