<!-- lint-g3mark -->

# tfgnn.broadcast_context_to_edges

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

Dispatches function calls for KerasTensor inputs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.broadcast_context_to_edges(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

Wraps a TF-GNN library function as a TFGNNOpLambda Keras layer if any of the
call inputs are Keras tensors. In particular, this allows to use TFGNN
functions, such as `tf.broadcast(...)`, with the Keras Functional API.

See `_GraphPieceClassMethodDispatcher` for details on how function arguments are
translated into the Keras Layer inputs.
