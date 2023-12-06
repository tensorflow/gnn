# tfgnn.keras.layers.MapFeatures

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/map_features.py#L27-L317">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Transforms features on a GraphTensor by user-defined callbacks.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.MapFeatures(
    context_fn=None,
    node_sets_fn=None,
    edge_sets_fn=None,
    *,
    allowed_aux_node_sets_pattern: Optional[str] = None,
    allowed_aux_edge_sets_pattern: Optional[str] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This layer transforms the feature maps of graph pieces (that is, EdgeSets,
NodeSets, or the Context) by applying Keras Models to them. Those Models are
built by user-supplied callbacks that receive a KerasTensor for the graph piece
as input and return a dict of output features computed with the Keras functional
API, see https://tensorflow.org/guide/keras/functional.

Auxiliary graph pieces (e.g., for
<a href="../../../tfgnn/keras/layers/StructuredReadout.md"><code>tfgnn.keras.layers.StructuredReadout</code></a>)
are skipped, unless explicitly requested via `allowed_aux_node_sets_pattern` or
`allowed_aux_edge_sets_pattern`.

#### Examples:

```python
# Hashes edge features called "id", leaves others unchanged:
def edge_sets_fn(edge_set, *, edge_set_name):
  features = edge_set.get_features_dict()
  ids = features.pop("id")
  num_bins = 100_000 if edge_set_name == "views" else 20_000
  hashed_ids = tf.keras.layers.Hashing(num_bins=num_bins)(ids)
  features["hashed_id"] = hashed_ids
  return features
graph = tfgnn.keras.layers.MapFeatures(edge_sets_fn=edge_sets_fn)(graph)
```

```python
# A simplistic way to map node features to an initial state.
def node_sets_fn(node_set, *, node_set_name):
  state_dims_by_node_set = {"author": 32, "paper": 64}  # ...and so on.
  state_dim = state_dims_by_node_set[node_set_name]
  features = node_set.features  # Immutable view.
  if features: # Concatenate and project all inputs (assumes they are floats).
    return tf.keras.layers.Dense(state_dim)(
        tf.keras.layers.Concatenate([v for _, v in sorted(features.items())]))
  else:  # There are no inputs, create an empty state.
    return tfgnn.keras.layers.MakeEmptyFeature()(node_set)
graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn)(graph)
```

```python
# Doubles all feature values, with one callback used for all graph pieces,
# including auxiliary ones.
def fn(inputs, **unused_kwargs):
  return {k: tf.add(v, v) for k, v in inputs.features.items()}
graph = tfgnn.keras.layers.MapFeatures(
    context_fn=fn, node_sets_fn=fn, edge_sets_fn=fn,
    allowed_aux_node_sets_pattern=r".*", allowed_aux_edge_sets_pattern=r".*"
)(graph)
```

When this layer is called on a GraphTensor, it transforms the feature map of
each graph piece with the model built by the respective callbacks. The very
first call to this layer triggers building the models. Subsequent calls to this
layer do not use the callbacks again, but check that their input does not have
more graph pieces or features than seen by the callbacks:

*   It is an error to call with a node set or edge set that was not present in
    the first call. (After the first call, it is too late to initialize another
    model for it and find out what the callback would have done.) An exception
    is made for auxiliary node sets and edge sets: If they would have been
    ignored in the first call anyways, they may be present in later calls and
    get ignored there.
*   It is an error to call with a set of feature names of some graph piece that
    has changed since the first call, except for those graph pieces for which
    the callback was `None` or returned `None` to request passthrough. (Without
    this check, the model for the graph piece would silently drop new features,
    even though the callback might have handled them.)

More details on the callbacks:

The model-building callbacks are passed as arguments when initializing this
layer (see "Init args" below). Each callback is invoked as `fn(graph_piece,
**kwargs)` where

*   `graph_piece` is a KerasTensor for the EdgeSet, NodeSet or Context that is
    being transformed. It provides access to the input features.
*   the keyword argument (if any) is
    *   `edge_set_name=...` when transforming the features of that EdgeSet,
    *   `node_set_name=...` when transforming the features of that NodeSet,
    *   absent when transforming the features of the Context.

The output of the callbacks can take one of the following forms:

*   A returned dict of feature values is used as the new feature map of the
    respective graph piece in this layer's output. Returning the empty dict `{}`
    is allowed and results in an empty feature map.
*   A returned feature `value` not wrapped in a dict is a shorthand for
    `{tfgnn.HIDDEN_STATE: value}`, to simplify the set-up of initial states.
*   Returning `None` as the callback's result indicates to leave this graph
    piece alone and not even validate that subsequent inputs have the same
    features.

The output values are required to

*   have the correct shape for a feature on the respective piece of the
    GraphTensor;
*   depend on the input, so that the Keras functional API can use them as Model
    outputs.

This happens naturally for outputs of transformed input features. Outputs
created from scratch still need to depend on the input for its size. The helper
`tfgnn.keras.layers.MakeEmptyFeature()(graph_piece)` does this for the common
case of creating an empty hidden state for a latent node; see its documentation
for details on how to use it with TPUs. If TPUs and shape inference are no
concern, the callback can simply use `graph_piece.sizes` or (esp. for rank 0)
graph_piece.total_size`to construct outputs of the right shape, but
not`graph_piece.spec.total_size`, which breaks the dependency chain of
KerasTensors.

Weight sharing between the transformation of different graph pieces is possible
by sharing the Keras objects between the respective callback invocations.

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`.

WARNING: Weight sharing fails in `tf.keras.models.load_model()` with an error
message on weights missing from the checkpoint. (Most users don't need to
re-load their models this way.)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
<code>context_fn</code><a id="context_fn"></a>
</td>
<td>
A callback to build a Keras model for transforming context
features. It will be called as <code>output = context_fn(g.context)</code>.
Leaving this at the default <code>None</code> is equivalent to returning <code>None</code>.
</td>
</tr><tr>
<td>
<code>node_sets_fn</code><a id="node_sets_fn"></a>
</td>
<td>
A callback to build a Keras model for transforming node set
features. It will be called for every node sets as
<code>node_sets_fn(g.node_sets[node_set_name], node_set_name=node_set_name)</code>.
Leaving this at the default <code>None</code> is equivalent to returning <code>None</code>
for every node set.
</td>
</tr><tr>
<td>
<code>edge_sets_fn</code><a id="edge_sets_fn"></a>
</td>
<td>
A callback to build a Keras model for transforming edge set
features. It will be called for every edge sets as
<code>edge_sets_fn(g.edge_sets[edge_set_name], edge_set_name=edge_set_name)</code>.
Leaving this at the default <code>None</code> is equivalent to returning <code>None</code>
for every edge set.
</td>
</tr><tr>
<td>
<code>allowed_aux_node_sets_pattern</code><a id="allowed_aux_node_sets_pattern"></a>
</td>
<td>
If set, <code>node_sets_fn</code> is also invoked for
those auxiliary node sets that match this pattern, according to Python's
<code>re.fullmatch(pattern, node_set_name)</code>.
</td>
</tr><tr>
<td>
<code>allowed_aux_edge_sets_pattern</code><a id="allowed_aux_edge_sets_pattern"></a>
</td>
<td>
If set, <code>edge_sets_fn</code> is also invoked for
those auxiliary edge sets that match this pattern, according to Python's
<code>re.fullmatch(pattern, edge_set_name)</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
A GraphTensor. The very first call triggers the building of
the models that map the various feature maps, with tensor specs
taken from the GraphTensorSpec of the first input.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A GraphTensor with the same nodes and edges as the input, but with
transformed feature maps.
</td>
</tr>

</table>
