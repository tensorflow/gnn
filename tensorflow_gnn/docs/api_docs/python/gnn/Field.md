description: The central part of internal API.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.Field" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.Field

<!-- Insert buttons and diff -->
This symbol is a **type alias**.

The central part of internal API.

#### Source:

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Field = Union[
    google3.third_party.tensorflow.python.framework.ops.Tensor,
    google3.third_party.tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor
]
</code></pre>



<!-- Placeholder for "Used in" -->

This represents a generic version of type 'origin' with type arguments 'params'.
There are two kind of these aliases: user defined and special. The special ones
are wrappers around builtin collections and ABCs in collections.abc. These must
have 'name' always set. If 'inst' is False, then the alias can't be instantiated,
this is used by e.g. typing.List and typing.Dict.