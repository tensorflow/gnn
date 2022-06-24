# tfgnn.keras.layers.TotalSize

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/map_features.py#L288-L323">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the .total_size of a graph piece.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.TotalSize(
    *, constant_from_spec: bool = False, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer returns the total size of an input EdgeSet, NodeSet or Context
as a scalar tensor (akin to `input.total_size`), with a dependency on the
input tensor as required by the Keras functional API. This layer can be used
to generate new feature values for a scalar GraphTensor inside a callback
passed to MapFeatures.

#### Init args:


* <b>`constant_from_spec`</b>: Setting this to true guarantees that the output is a
  constant Tensor (suitable for environments in which constant shapes are
  required, like models distributed to TPU). Setting this requires that
  `input.spec.total_size is not None`. If unset, the output Tensor may or
  may not be constant.


