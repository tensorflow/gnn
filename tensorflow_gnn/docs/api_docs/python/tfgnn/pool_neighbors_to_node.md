<!-- lint-g3mark -->

# tfgnn.pool_neighbors_to_node

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

Dispatches function calls for KerasTensor inputs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool_neighbors_to_node(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

Wraps a TF-GNN library function as a TFGNNOpLambda Keras layer if any of the
call inputs are Keras tensors. In particular, this allows to use TFGNN
functions, such as `tf.broadcast(...)`, with the Keras Functional API.

See `_GraphPieceClassMethodDispatcher` for details on how function arguments are
translated into the Keras Layer inputs.