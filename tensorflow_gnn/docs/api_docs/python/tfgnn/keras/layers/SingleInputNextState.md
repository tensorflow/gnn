# tfgnn.keras.layers.SingleInputNextState

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/next_state.py#L217-L244">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Replaces a state from a single input.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.SingleInputNextState(
    trainable=True, name=None, dtype=None, dynamic=False, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

In a NodeSetUpdate, it replaces the node state with a single edge set input. For
an EdgeSetUpdate, it replaces the edge_state with the incident node set's input.
For a ContextUpdate, it replaces the context state with a single node set input.

#### Call returns:

A tensor to use as the new state.
