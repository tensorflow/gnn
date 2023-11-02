<!-- lint-g3mark -->

# tfgnn.satisfies_size_constraints

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

Dispatches function calls for KerasTensor inputs.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfgnn.satisfies_total_sizes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.satisfies_size_constraints(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

Wraps a TF-GNN library function as a TFGNNOpLambda Keras layer if any of the
call inputs are Keras tensors. In particular, this allows to use TFGNN
functions, such as `tf.broadcast(...)`, with the Keras Functional API.

See `_GraphPieceClassMethodDispatcher` for details on how function arguments are
translated into the Keras Layer inputs.
