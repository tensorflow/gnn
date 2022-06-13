description: Transforms features on a GraphTensor by user-defined callbacks.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.keras.layers.MapFeatures" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfgnn.keras.layers.MapFeatures

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/map_features.py#L12-L244">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Transforms features on a GraphTensor by user-defined callbacks.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.MapFeatures(
    context_fn=None, node_sets_fn=None, edge_sets_fn=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer transforms the feature maps of graph pieces (that is, EdgeSets,
NodeSets, or the Context) by applying Keras Models to them. Those Models
are built by user-supplied callbacks that receive a KerasTensor for the
graph piece as input and return a dict of output features computed with
the Keras functional API, see https://tensorflow.org/guide/keras/functional.

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
  else:  # There are no inputs, create a zero state.
    total_size = tfgnn.keras.layers.TotalSize()(node_set)
    return tf.zeros([total_size, state_dim])
graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn)(graph)
```

```python
# Doubles all feature values, with one callback used for all graph pieces.
def fn(inputs, **unused_kwargs):
  return {k: tf.add(v, v) for k, v in inputs.features.items()}
graph = tfgnn.keras.layers.MapFeatures(
    context_fn=fn, node_sets_fn=fn, edge_sets_fn=fn)(graph)
```

When this layer is called on a GraphTensor, it transforms the feature map
of each graph piece with the model built by the respective callbacks.
The very first call to this layer triggers building the models. Subsequent
calls to this layer do not use the callbacks again, but check that their
input does not have more graph pieces or features than seen by the callbacks:

  * It is an error to call with a node set or edge set that was not present
    in the first call. (After the first call, it is too late to initialize
    another model for it and find out what the callback would have done.)
  * It is an error to call with a set of feature names of some graph piece
    that has changed since the first call, except for those graph pieces for
    which the callback was `None` or returned `None` to request passthrough.
    (Without this check, the model for the graph piece would silently drop
    new features, even though the callback might have handled them.)

More details on the callbacks:

The model-building callbacks are passed as arguments when initializing this
layer (see "Init args" below). Each callback is invoked as
`fn(graph_piece, **kwargs)` where

  * `graph_piece` is a KerasTensor for the EdgeSet, NodeSet or Context
    that is being transformed. It provides access to the input features.
  * the keyword argument (if any) is
      * `edge_set_name=...` when transforming the features of that EdgeSet,
      * `node_set_name=...` when transforming the features of that NodeSet,
      * absent when transforming the features of the Context.

The output of the callbacks can take one of the following forms:

  * A returned dict of feature values is used as the new feature map of
    the respective graph piece in this layer's output. Returning the
    empty dict `{}` is allowed and results in an empty feature map.
  * A returned feature `value` not wrapped in a dict is a shorthand for
    `{tfgnn.HIDDEN_STATE: value}`, to simplify the set-up of initial
    states.
  * Returning `None` as the callback's result indicates to leave this graph
    piece alone and not even validate that subsequent inputs have the same
    features.

The output values are required to

  * have the correct shape for a feature on the respective piece of the
    GraphTensor;
  * depend on the input, so that the Keras functional API can use them
    as Model outputs.

This happens naturally for outputs of transformed input features.
Outputs created from scratch still need to depend on the input for its size.
In case of scalar GraphTensors, users are recommended to call
`tfgnn.keras.layers.TotalSize()(graph_piece)` and use the result as the
leading dimension of outputs, as seen in the example code snippet above.
(For constant shapes on TPUs, see the documentation of TotalSize.)

#### Init args:


* <b>`context_fn`</b>: A callback to build a Keras model for transforming context
  features. It will be called as `output = context_fn(g.context)`.
  Leaving this at the default `None` is equivalent to returning `None`.
* <b>`node_sets_fn`</b>: A callback to build a Keras model for transforming node set
  features. It will be called for every node sets as
  `node_sets_fn(g.node_sets[node_set_name], node_set_name=node_set_name)`.
  Leaving this at the default `None` is equivalent to returning `None`
  for every node set.
* <b>`edge_sets_fn`</b>: A callback to build a Keras model for transforming edge set
  features. It will be called for every edge sets as
  `edge_sets_fn(g.edge_sets[edge_set_name], edge_set_name=edge_set_name)`.
  Leaving this at the default `None` is equivalent to returning `None`
  for every edge set.


#### Call args:


* <b>`graph`</b>: A GraphTensor. The very first call triggers the building of
  the models that map the various feature maps, with tensor specs
  taken from the GraphTensorSpec of the first input.


#### Call returns:

A GraphTensor with the same nodes and edges as the input, but with
transformed feature maps.


