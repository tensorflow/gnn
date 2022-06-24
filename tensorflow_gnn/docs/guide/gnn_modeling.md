# TF-GNN Modeling Guide

## Introduction

This document provides an in-depth introduction to building Graph Neural Network
models (GNNs for short) in Keras with the TF-GNN library.

The input to a GNN is a [GraphTensor](graph_tensor.md). Recall that it consists
of one or more node sets, a number of edge sets connecting them, and a so-called
context to hold graph-wide information. Each of these pieces comes with a dict
from feature names to feature tensors.

The [data input pipeline](input_pipeline.md) has already pre-processed
arbitrary input data into the inputs for the trainable part of the GNN model:
floating-point features that can be input directly into a neural network, or
integer indices into trainable embedding tables that supply such inputs. For
training and evaluation, these inputs are typically presented as a
`tf.data.Dataset` whose elements are pairs of a GraphTensor
(after `.merge_batch_to_components()`) and a labels Tensor.

This document uses primarily the
[Functional API](https://www.tensorflow.org/guide/keras/functional) of Keras,
in which `Layer`s are composed into `Model`s by invoking `Layer` objects on
symbolic KerasTensors, starting from a placeholder for the inputs that will be
fed during training, and later during evaluation and inference:

```python
train_ds = ...  # A tf.data.Dataset of (graph, label) pairs.
model_input_graph_spec, label_spec = train_ds.element_spec
input_graph = tf.keras.layers.Input(type_spec=model_input_graph_spec)
```


## The big picture: initialization, graph updates, and readout

Typically, a model with a GNN architecture at its core consists of three parts:

 1. The initialization of hidden states on nodes (and possibly also edges and/or
    the graph context) from their respective preprocessed features.
 2. The actual Graph Neural Network: several rounds of updating hidden states
    from neighboring items in the graph.
 3. The readout of one or more hidden states into some prediction head, such as
    a linear classifier.


### Initializing the hidden states

The hidden states on nodes are created by mapping a dict of features (possibly
already preprocessed by the input pipeline) to fixed-size hidden states for
nodes. This happens separately for each node set, but batched for all nodes of
the node set, so the result for each node set is a Tensor of shape
`[num_nodes, state_size]`. The `tfgnn.keras.layers.MapFeatures` layer lets you
specify such a transformation as a callback function that transforms feature
dicts, with GraphTensor mechanics taken off your shoulders:

```python
def set_initial_node_state(node_set, node_set_name):
  if node_set_name == "authors":
    author_id_embedding = tf.keras.layers.Embedding(num_authors, 128)
    return tf.keras.layers.Concatenate()(
        [node_set["dense_feature"], author_id_embedding(node_set["author_id"])])
  elif node_set_name == ...:
    ...

graph = tf.keras.layers.MapFeatures(
    node_sets_fn=set_initial_node_state)(input_graph)
```

`MapFeatures` takes a single callback function for all node sets, so the user
is free to organize feature handling separately by node set name (like in the
if-elif cascade sketched above), by feature name or dtype, or by a combination
of these.

If applicable, initializing the hidden states of edges and/or the context works
exactly the same, just provide `MapFeatures` with an `edge_sets_fn=...` (which
will receive an `edge_set` and the `edge_set_name`) and/or a `context_fn`
(which will receive the `context`),

The body of the callback function uses the Functional API of Keras to build a
small model for each node set, starting from the `node_set` input (actually,
a KerasTensor wrapping a `tfgnn.NodeSet`).
This allows us to introduce new stateful parts of the model on the fly, like the
`author_id_embedding` in the example code above. Keras takes care of tracking
the variables involved for initialization, training and saving. The code above
creates a fresh `author_id_embedding` when called for `node_set_name="authors"`.
Alternatively, it could share one `author_id_embedding` object between calls to
use a shared embedding table for author id features seen on different node sets.

Recall that each input feature `node_set["..."]` has shape `[num_nodes, ...]`,
with the leading dimension indexing over all nodes in this node set. This is not
the batch size in the usual sense (the number of distinct training examples
drawn at random), but plays a very similar role when used with standard Keras
layers (a number of independently processed entities).

In a graph with multiple node sets, some of them may start out **latent**, that
is to say, with no input features from which a meaningful initial state could be
derived. For those, we recommend creating a hidden state of size 0, so that
formally there is a tensor for the state, just without any entries. Nonetheless,
its leading dimension must match the number of nodes in the node set. This is
achieved as follows:

```python
def set_initial_node_state(node_set, node_set_name):
  total_size = tfgnn.keras.layers.TotalSize()(node_set)
  if node_set_name == "some_latent_node_set":
    return tf.zeros([total_size, 0])
  ...
```

Either way, the tensors returned by `set_initial_node_state` are stored in the
output `GraphTensor` of `MapFeatures` as features with the name
`"hidden_state"`, for which TF-GNN defines the constant `tfgnn.HIDDEN_STATE`:

```python
graph = tfgnn.keras.layers.MapFeatures(
    node_sets_fn=set_initial_node_state)(input_graph)
assert list(graph.node_sets["authors"].keys()) == [tfgnn.HIDDEN_STATE]
```

Observe how all other node features are dropped. (If that's undesired for
advanced applications, check out the docstring of MapFeatures on how to return
a modified dict instead.)


### Updating hidden states from the neighborhood: the GNN

After the initial states of nodes have been set up, it's time for the main part
of the Graph Neural Network: one or more rounds of updating hidden states from
the hidden states of adjacent nodes (or incident edges, as applicable),
expressed as one or several Keras layers that take a GraphTensor as input and
return an updated GraphTensor as output:

```python
def gnn(graph):
  graph = AGraphUpdate(...)(graph)
  graph = AGraphUpdate(...)(graph)
  ...
  return graph
```

You can get `AGraphUpdate` from several sources:

**Use predefined `tensorflow_gnn.models`** (missing doc).
This subdirectory collects implementations of several standard GNN
architectures. Typically, they let you do something along the lines of

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import foo

def gnn(graph):
  graph = foo.FooGraphUpdate(...)(graph)
  graph = foo.FooGraphUpdate(...)(graph)
  ...
  return graph
```

...but it depends on the `FooGraphUpdate` class how to initialize it and how it
treats multiple node and edge sets in the input. Please browse the models
documentation to find out more.

**Write your own GNN from scratch.** If you want, you can take matters
completely into your own hands by defining your own subclass of
`tf.keras.layers.Layer` with the
[subclassing API of Keras](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
and applying whatever mix of TF and TF-GNN tensor ops you want to perform one
round of graph updates – or even multiple rounds at once. The input and output
should be a GraphTensor, to fit in with the rest of TF-GNN.

**Composing GraphUpdates from convolutions and next-state layers** is a middle
ground between these two: it uses the highly configurable
`tfgnn.keras.layers.GraphUpdate` layer to define custom GNNs for the
heterogeneous graph at hand from standard building blocks without having to
write low-level code. This approach turned out very useful in the initial
practical applications of the TF-GNN library, and we will discuss it at length
in its own section further down.


### Reading out final states and making predictions

After some rounds of graph updates by the GNN, it is time to read out suitable
features from the graph and feed them to a "prediction head" at the top of your
model. It depends on the task at hand and on the dataflow in the GNN where in
the graph the suitable features can be found.

The following code snippets illustrate some typical cases. For simplicity, they
are formulated for binary classification with a linear classifier. They all come
after the code from the previous sections, which can be summarized as

```python
input_graph = tf.keras.layers.Input(type_spec=model_input_graph_spec)
graph = tf.keras.layers.MapFeatures(
    node_sets_fn=set_initial_node_state)(input_graph)
graph = gnn(graph)
```

**Classifying each node** in one particular node set, based on its hidden state.

```python
...  # As above.
logits = tf.keras.layers.Dense(1)(graph.node_sets["papers"][tfgnn.HIDDEN_STATE])
model = tf.keras.Model(input_graph, logits)
```

**Classifying each graph as a whole**, based on an aggregation of the node
states from one node set.

```python
...  # As above.
pooled_features = tfgnn.keras.layers.Pool(
     tfgnn.CONTEXT, "mean", node_set_name="your_node_set")(graph)
logits = tf.keras.layers.Dense(1)(pooled_features)
model = tf.keras.Model(input_graph, logits)
```

This can be extended to multiple node sets by combining respective pooled
features using `tf.keras.layers.Add` or `Concatenate`.

Recall from the [input pipeline guide](input_pipeline.md) that the multiple
input graphs (with one component each) from one batch of training data have been
merged into one scalar GraphTensor with distinct components, and that context
features are maintained per component. Therefore, this code snippet produces
`pooled_features` of shape `[batch_size, node_state_size]` separately for each
original input graph.

The code above lets you do simple `"mean"` pooling (or `"sum"` or
`"max_no_inf"` etc.).
Smarter ways of pooling (e.g., with attention) are possible with "convolutions
to context", which are discussed further down.

**Classifying each sampled subgraph**, based on the hidden state at its root
node. (You can skip this if you are unfamiliar with graph sampling.
The [data preparation guide](data_prep.md) provides an introduction.)

```python
...  # As above.
root_states = tfgnn.keras.layers.ReadoutFirstNode(
    node_set_name="node_set_with_root")(graph)
logits = tf.keras.layers.Dense(1)(root_states)
model = tf.keras.Model(input_graph, logits)
```

Be sure to pass the `node_set_name=` that contains the root node, or else it is
indeterminate which node happens to be the first in it.

**In all these cases**, the resulting model for binary classification has the
same signature (a batch of preprocessed graphs to a batch of logits) and can be
trained with code like

```python
model = tf.keras.Model(input_graph, logits)  # As above.
model.compile(tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
model.fit(train_ds, ...)
```


## Modeling with the `tfgnn.keras.layers.GraphUpdate`

Recall that we regard a **Graph Neural Network** on a heterogeneous graph as a
sequence of GraphUpdates. In this section, we define them using the highly
parametric `tfgnn.keras.layers.GraphUpdate` layer class.

```python
def gnn(graph):
  graph = tfgnn.keras.layers.GraphUpdate(...)(graph)
  graph = tfgnn.keras.layers.GraphUpdate(...)(graph)
  ...
  return graph
```


### Using node states

For simplicity, let us start with a node-centric approach: only nodes have input
features, and only nodes carry a hidden state in the GNN. That means, for now,
edges define the adjacency between nodes but do not carry data of their own.
We will return to features and states on edges and context in later sections.

A **GraphUpdate** applies node set updates to node sets in the input graph.

```python
output_graph = tfgnn.keras.layers.GraphUpdate(
    node_sets={
        "author": tfgnn.keras.layers.NodeSetUpdate(...),
        "paper": tfgnn.keras.layers.NodeSetUpdate(...),
        ...})(input_graph)
```

These node set updates happen in parallel: they all see the same `input_graph`,
and their results are stored as updated node features on the `output_graph`.
Node sets not named here are carried through unchanged.

A **node set update** is a Keras layer that can be called like

```python
new_state = node_set_update(graph, node_set_name="...")
```

and returns a new hidden state tensor for the specified node set. Users can use
the library-provided `NodeSetUpdate` class as shown or implement their own Keras
layer to do this.

The library-provided `tfgnn.keras.layers.NodeSetUpdate` combines two kinds of
pieces that do actual computations:

  * Convolutions per edge set towards the updated node set.
  * One next-state layer to compute a new node state from the old node state and
    from the results of the convolutions.

For example:

```python
tfgnn.keras.layers.NodeSetUpdate(  # For node set "author".
    {"writes": tfgnn.keras.layers.SimpleConv(
         tf.keras.layers.Dense(128, "relu"), "mean",
         receiver_tag=tfgnn.SOURCE),
     "affiliated_with": tfgnn.keras.layers.SimpleConv(
         tf.keras.layers.Dense(64, "relu"), "sum",
         receiver_tag=tfgnn.SOURCE)},
    tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(128)))
```

A **convolution** is a Keras layer `conv` that can be called like

```python
conv_result = conv(graph, edge_set_name="some_edge_set")
```

and returns a `conv_result` of shape `[num_nodes, ...]` indexed by its receiver
node set. The receiver node set is chosen when `conv` is initialized. It is set
to one of the edge endpoints, which are `SOURCE` and `TARGET` (or more for
hypergraphs). The terms
**source** and **target** are borrowed from graph theory and denote the
direction of the logical relationship expressed by an edge in the graph (a
"writes" edge goes from an "author" at its source to a "paper" at its target).
By contrast, "**receiver**" and "**sender**" talk about the direction of
dataflow in one particular place of a GNN model: either one of source and target
can be picked as the **receiver**, the other one is the **sender**.

The `SimpleConv` used above applies the passed-in transformation
`Dense(..., "relu")` on the concatenated source and target node states of each
edge and then pools the result for each receiver node, with a user-specified
pooling method like `"sum"` or `"mean"`.

Many other convolutions can be used in this place, from the TF-GNN library or
defined in user code to match the calling convention above. See the separate
section on "Convolutions" for more.

To share convolutions' trained weights between parts of the model, it is
possible to use the same convolution object for multiple edge sets, as long as
the tensor shapes involved are compatible. (Observe how the `edge_set_name` is
only passed at call time.)

A **next-state layer** for a node set is a Keras layer `next_state` that can be
called like

```python
new_state = next_state((state, {edge_set_name: conv_result, ...}, {}))
```

that is, with a tuple of inputs from (1) the node set itself, (2) the specified
convolutions keyed by the edge set involved, (3) the graph context, which this
doc will discuss below. Its result becomes the result of the `NodeSetUpdate`
layer.

The research literature on GNNs has borrowed the term "convolution" from
Convolutional Neural Networks on multi-dimensional grids, with the twist that
neighborhoods in a graph, unlike a grid, are non-uniform and explicitly encoded
by the edges. Notice that TF-GNN adds another twist on top to support
heterogeneous graphs: At each node, a convolution computes one result for each
edge set (aggregated across the variable number of incident edges), but
computing the new node state from the fixed number of edge sets is left to the
next-state layer. The example above shows the library-provided
`NextStateFromConcat`, which concatenates all inputs and sends them through a
user-supplied projection for computing the new node state.

Users can supply their own next-state layers. For example, much of the research
literature discusses homogeneous graphs in which the single convolution result
is used directly as the new node state, which can be expressed in TF-GNN as
follows:

```python
class NextNodeStateFromSingleEdgeSet(tf.keras.layers.Layer):

  def call(self, inputs):
    unused_state, edge_inputs, unused_context_input = inputs  # Unpack.
    if len(edge_inputs) != 1: raise ValueError("Expected input from one edge set")
    single_edge_set_input = list(edge_inputs.values())[0]
    return single_edge_set_input
```

Here, `edge_inputs` is a dict of tensors, keyed by edge set name.

In principle, next-state layers can be shared between parts of the model to
reuse weights, but care must be taken to align the order of str-keyed inputs if
it matters.

**Design note:** Observe how the library-supplied `SimpleConv` and
`NextStateFromConcat` delegate the actual computations to passed-in Keras layers
with plain Tensor inputs and outputs. This leverages the standard tf.keras
package as domain-specific language to express the small feed-forward networks
needed in these places. Instead of the single `Dense` layers seen above, real
models tend to use small sub-networks returned by a helper in the style of

```python
def dense(units, activation="relu", l2_regularization=..., dropout_rate=0.0):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(
          units, activation,
          kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
      tf.keras.layers.Dropout(dropout_rate)])
```

User-defined convolutions and next-state layers are not required to be so
modular and can contain very specific operations if they see fit. This is even
true for library code that implements a specific component of a published GNN
model architecture.

**To summarize,** TF-GNN provides a set of Keras layers that lets you spell out
a GNN as a sequence of GraphUpdates, composed of convolutions over edge sets and
next-state computations for node sets.


### Using edge and context features

This section discusses the use of additional input features stored on edge sets
or the graph context, without modifying them in graph updates. This is a useful,
but conceptually minor addition to the node-centric modeling presented above.

The `tfgnn.keras.layers.MapFeatures` layer introduced above can be initialized
with any combination of `edge_sets_fn=`, `node_set_fn=` and `context_fn=` to
transform the features of any of these graph pieces. Please refer to its
documentation for details. In particular, this is how input features on edge
sets and the context can be embedded or combined in a trainable manner at the
start of the model. For features that are not updated like hidden states, we
recommend using a descriptive name other name than `tfgnn.HIDDEN_STATE`.
To do so, return a singleton dict `{"my_feature": tensor}` instead of the plain
`tensor`.

Edge features can be used as side inputs to convolutions. For example,
`tfgnn.keras.layers.SimpleConv` used above has an init argument
`sender_edge_feature=...`that can be set to the name of an edge feature, which
then gets included in the concatenation of inputs to the user-supplied
transformation.

Context features can be used as side inputs to the next-state layer of a
`NodeSetUpdate` by passing the init argument
`NodeSetUpdate(..., context_feature_name=...)`, and likewise for the
`EdgeSetUpdate` introduced in the next section. The selected feature ends up
in the third component of the input tuple of the next-state layer.

Context features could also be used as side inputs to convolutions. (As of
May 2022, no convolution class provides an option for that. Please let us
know about applications for which this proved to be important.)


### Using context state

This section discusses the use of a context feature for a hidden state that gets
updated with each GraphUpdate.

Why would you do that?

*   If your task is a prediction about the whole graph, a context state that
    represents the relevant properties of the graph is a plausible input to the
    prediction head of your model. Maintaining that state throughout the GNN is
    potentially more expressive than a single pooling of node states at the end
    of the GNN.
*   A context state that gets fed back into node state updates could condition
    them on some global characteristics of the graph.

Formally, adding a context state to a model with node states is equivalent to
adding an extra node set with one node per graph component, and adding extra
edge sets (without features) to connect all other nodes to the extra node
so that its state is available in node set updates.

The code for an update of node set states and the context state reads like:

```python
output_graph = tfgnn.keras.layers.GraphUpdate(
    node_sets={"author": tfgnn.keras.layers.NodeSetUpdate(...), ...},  # As above.
    context=tfgnn.keras.layers.ContextUpdate(
        {"author": tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "mean"),
         "paper": tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "sum")},
        tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(128)))
)(input_graph)
```

If the context state is meant to be used by the convolutions and next-state
layer within each `NodeSetUpdate(...)`, be sure to set the respective init args.
(Many of these classes default to using only node states.)

The `GraphUpdate` layer first runs all node set updates in parallel on the input
graph, then runs the context update on the graph with updated node sets, and
finally returns the graph with updated node sets and context. (If you need the
opposite order, use a `GraphUpdate(context=...)` followed by a
`GraphUpdate(node_sets=...)`.)

A **context update** is a Keras layer `context_update` that can be called like

```python
new_context_state = context_update(graph)
```

and returns a new hidden state tensor for the context. Users can use the
library-provided `ContextUpdate` class as shown, or implement their own Keras
layer to do this.

The library-provided `tfgnn.keras.layers.ContextUpdate` is quite similar to the
`NodeSetUpdate` described above in that it combines two kinds of pieces that do
the actual computations:

  * Layers like `Pool(CONTEXT, ...)` that can be called as `pool(graph,
    node_set_name=...)` to provide an aggregated input from that node set.
  * One next-state layer that can be called on a tuple with the old context
    state, a dict from node set names to their pooled inputs, and (not shown
    here) a dict from edge set names to their pooled inputs.

Please see the class docstring for details. The later section on "convolutions
to context" will discuss pooling in smarter ways with attention.

The context state, stored under feature name `tfgnn.HIDDEN_STATE`, can be
used in node set updates (and the edge set updates discussed below) much like
any context feature.

### Using edge states

The traditional take on graph neural networks makes nodes the stateful entities,
updated directly from their neighbor nodes, while edges provide a stateless
connection to these neighbors. The Graph Nets model, formulated by
[Battaglia&al. (2018)](https://arxiv.org/abs/1806.01261) for homogenous graphs,
equips edges with a state as well. TF-GNN provides a common framework in which
both points of view can be expressed reasonably idiomatically for heterogeneous
graphs and can share code.

A GraphUpdate in Graph Nets style looks as follows:

```python
output_graph = tfgnn.keras.layers.GraphUpdate(
    edge_sets={
        "writes": tfgnn.keras.layers.EdgeSetUpdate(...),  # Author to paper.
        "cites": tfgnn.keras.layers.EdgeSetUpdate(...),  # Paper to paper.
        ...},
    node_sets={
        "author": tfgnn.keras.layers.NodeSetUpdate(...),
        "paper": tfgnn.keras.layers.NodeSetUpdate(...),
        ...},
    context=tfgnn.keras.layers.ContextUpdate(...),
)(input_graph)
```

Be sure to set the right initializer args to enable using edge states, node
states and/or the context state in all places that are meant to receive them.
(Many of these classes default to using only node states.)

The `output_graph` is computed in three steps:

 1. The edge set updates are computed from the input graph.
 2. The node set updates are computed from the graph with updated edge sets.
 3. The context update is computed from a graph with updated edge sets and node
    sets.

An **edge set update** is a Keras layer that can be called like

```python
new_edge_state = edge_set_update(graph, edge_set_name="...")
```

and returns a new hidden state tensor for the specified edge set. Users can use
the library-provided `EdgeSetUpdate` class as shown or implement their own Keras
layer to do this.

The library-provided `tfgnn.keras.layers.EdgeSetUpdate` selects input features
from the edge and its incident nodes, then passes them through a next-state
layer, similar to the `NodeSetUpdate` and `ContextUpdate` discussed above.
Please see its docstring for details.

A **node set update** in a model with edge states has the same syntax as in the
node-centric case but does a different job. As before, it gets called like

```python
new_node_state = node_set_update(graph, node_set_name="...")
```

to compute a new state for each node, but now it reads the old node state,
possibly a context state, and the states of incoming edges – not of the nodes at
their other end, because those have already been used to compute the edge
states.

For the library-supplied `tfgnn.keras.layers.NodeSetUpdate`, this boils down to
replacing convolutions across edge sets (node-to-node) with pooling layers from
edge sets (edge-to-node):

```python
tfgnn.keras.layers.NodeSetUpdate(  # For node set "author".
    {"writes": ​​tfgnn.keras.layers.Pool(tfgnn.SOURCE, "mean"),
     ...},
    tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(128)))
```

The next-state layer works as before, now receiving pooled edge states in place
of the convolution results.

The context state is optional. If present, the **context update** works as
discussed in the previous section. Please see the docstring of `ContextUpdate`
for how to input pooled edge states in addition to pooled node states.

Unlike adding edge *features* to a node-centric model, adding edge *states* and
breaking up node state updates accordingly is a major change in the model.

TF-GNN supports explicit edge state updates so that models described in the
style of Graph Nets can be expressed idiomatically. In formal terms, introducing
states on an edge set is equivalent to replacing those edges by auxiliary nodes
with one incoming and one outgoing edge, in two separate edge sets without
features, and updating these auxiliary nodes from their neighbors before the
original nodes. Between these alternate representations (and likewise for
context states), we recommend choosing the one that maximizes consistency
between the input data pipeline, the model, and the model's description in the
literature. The representations with auxiliary nodes and edges are slightly less
optimized (because these highly systematic edges are stored one by one as if
they were an arbitrary graph).


## Convolutions

### Basic usage: node to node

In the context of TF-GNN, every Keras layer that calls itself a convolution is
expected to work in a NodeSetUpdate as shown above in the section on
node-centric GNN models.

It is easy for users to define their own Keras layer of that kind. A minimal
example (to be refined below) looks like

```python
class MyFirstConvolution(tf.keras.layers.Layer):
  def __init__(self, units, *, receiver_tag, **kwargs):
    super().__init__(**kwargs)
    self._message_fn = dense(units)  # Our little helper above.
    self._receiver_tag = receiver_tag

  def call(self, graph, edge_set_name):
    receiver_states = tfgnn.broadcast_node_to_edges(
        graph, edge_set_name, self._receiver_tag,
        feature_name=tfgnn.HIDDEN_STATE)
    sender_states = tfgnn.broadcast_node_to_edges(
        graph, edge_set_name, tfgnn.reverse_tag(self._receiver_tag),
        feature_name=tfgnn.HIDDEN_STATE)

    message_inputs = [sender_states, receiver_states]
    messages = self._message_fn(tf.concat(message_inputs, axis=-1))

    return tfgnn.pool_edges_to_node(
        graph, edge_set_name, self._receiver_tag, "sum",
        feature_value=messages)
```

The literature on GNNs abounds with definitions of convolutions. TF-GNN contains
a growing collection of these under `tensorflow_gnn.models`.


### Convolutions to context

The concept of a convolution generalizes naturally from the relation of nodes
represented by an edge set to the relation of nodes to the graph context:
instead of explicitly stored edges, there is the containment of nodes in graph
components (which is what context features are indexed by).

TF-GNN allows convolution classes to provide that generalization in code if,
after suitable initialization, they can be called like

```python
conv_result_for_context = conv_to_context(graph, node_set_name="foo")
```

as required for use in

```python
tfgnn.keras.layers.ContextUpdate(node_sets={"foo": conv_to_context})
```

This is especially relevant for convolutions that implement some kind of
attention over the neighbor nodes, because it provides that same kind of
attention for "smart pooling" of node states to the context.

For example, `GATv2Conv(..., receiver_tag=tfgnn.CONTEXT)` supports this,
as long as the context has a feature (`tfgnn.HIDDEN_STATE` or as
specified) that can provide the query input for attention. This helps to reuse
the same implementation of attention for both cases.

**Design note:** Observe how the `receiver_tag=` argument, previously used to
select the `tfgnn.SOURCE` or `tfgnn.TARGET` nodes of an edge set, extends
naturally to `tfgnn.CONTEXT` as a special kind of receiver that goes beyond the
incident node *tags* (hence "receiver *tag*") of a
`GraphTensor.edge_set[...].adjacency` object. This is used in various places
across the TF-GNN library, like the `tfgnn.pool()` and `tfgnn.broadcast()`
operations on GraphTensors. This uniformity comes in handy to express
computations in terms of broadcast and pool that apply not only between incoming
edges and nodes, but just as well between nodes and context, or edges and
context. For example, see the implementation of `tfgnn.softmax()`.


### Reusing convolutions for edge pooling

Above, this doc has discussed the use of an edge feature as an extra input to
the per-edge computation of a convolution, which occurs naturally even in
node-centric models. For some convolutions, notably some that implement
attention, it makes sense to go one step further and remove the input from the
sender node altogether, leaving us with a per-edge computation involving an edge
feature, the receiver node state, and possibly a context feature, followed by
some aggregation towards the receiver. TF-GNN calls this kind of "truncated
convolution" an **EdgePool layer.**

Every convolution that offers init arguments to add an edge feature as input and
to remove the sender node feature as input can be reconfigured to become an
EdgePool layer for pooling towards a node set. For example, `GATv2EdgePool` is a
factory function for `GATv2Conv` layers that sets requisite init
arguments.

Finally, TF-GNN allows EdgePool layers to handle pooling of edge states to
context as well. For example,

```python
edge_pool_to_context = GATv2EdgePool(..., receiver_tag=tfgnn.CONTEXT)
```

can be called like

```python
pooled_edges = edge_pool_to_context(graph, edge_set_name="foo")
```

as required for use in

```python
tfgnn.keras.layers.ContextUpdate(edge_sets={"foo": edge_pool_to_context})
```

This way, the same implementation of GATv2 can be used for "smart pooling"
across all three many-to-one relationships in a GraphTensor: incident edges to
node, nodes to context, and edges to context.


### Defining your own convolution class

Recall the code above for `MyFirstConvolution`, which handles only the basic
node-to-node case. Convolutions that ship with the TF-GNN library are encouraged
to support all the other cases discussed above with the same interface
conventions, whenever it makes sense. For that purpose, TF-GNN provides the
class `tfgnn.keras.layers.AnyToAnyConvolutionBase`. Users can subclass it to
express their own convolutions by implementing the abstract `convolve()` method,
which abstracts away the GraphTensor structure and instead presents input
tensors extracted from the appropriate places in the graph together with
broadcast and pool functions that let users move values between them. The base
class calls `convolve()` from within its implementation of the `call()` method
of Keras.

Here is the equivalent of `MyFirstConvolution`, generalized to work from edges
and/or nodes to nodes or context. (Along the way, this example also adds support
for Keras serialization.)

```python
@tf.keras.utils.register_keras_serializable(package="MyGNNProject")
class ExampleConvolution(tfgnn.keras.layers.AnyToAnyConvolutionBase):

  def __init__(self, units, *,
               receiver_tag=None,
               receiver_feature=tfgnn.HIDDEN_STATE,
               sender_node_feature=tfgnn.HIDDEN_STATE,
               sender_edge_feature=None,
               **kwargs):
    super().__init__(
        receiver_tag=receiver_tag,
        receiver_feature=receiver_feature,
        sender_node_feature=sender_node_feature,
        sender_edge_feature=sender_edge_feature,
        **kwargs)
    self._units = units
    self._message_fn = dense(units)  # Our little helper above.

  def get_config(self):
    # The superclass handles the receiver_tag and *_feature values.
    return dict(units=self._units, **super().get_config())

  def convolve(
      self, *,
      sender_node_input, sender_edge_input, receiver_input,
      broadcast_from_sender_node, broadcast_from_receiver, pool_to_receiver,
      training):
    inputs = []
    if sender_node_input is not None:
      inputs.append(broadcast_from_sender_node(sender_node_input))
    if sender_edge_input is not None:
      inputs.append(sender_edge_input)
    if receiver_input is not None:
      inputs.append(broadcast_from_receiver(receiver_input))
    messages = self._message_fn(tf.concat(inputs, axis=-1))
    return pool_to_receiver(messages, reduce_type="sum")
```

For clarity, the code above spells out the `receiver_tag` and `*_feature`
arguments instead of lumping them into `**kwargs`. (The defaults in the base
class are the same as shown here.)

The `call()` method implemented by the base class has the interface required for
use in `NodeSetUpdate` or `ContextUpdate`: it accepts a `GraphTensor` and a
keyword argument `edge_set_name="..."` or `node_set_name="..."`. Also, a
`receivcer_tag` must be passed at call time if it was not set at init time. The
`call()` method picks out the input features specified at init time and forwards
them to `convolve()`.

Let's see how this pans out for the various cases:

For a **node-to-node** convolution, the layer is initialized with
`receiver_tag=tfgnn.SOURCE` or `tfgnn.TARGET` and then called like
`example_conv(graph, edge_set_name="...")`. By default, `convolve()` gets a
`sender_node_input` and `receiver_input` from the respective node states. The
`sender_edge_input` is present if selected at init time. The broadcast and pool
functions can be called on tensor values as seen above; their `graph`,
`edge_set_name` and `tag` arguments have been bound accordingly.

A **node-to-context** convolution works similarly, except the layer is
initialized with `receiver_tag=tfgnn.CONTEXT` and gets called like
`example_conv(graph, node_set_name="...")`. In `convolve()`, the
`receiver_input` is a context feature. There are no explicit edges, so
`sender_edge_input` will be `None`. Broadcast from and pool to receiver happen
between the node set and context, while broadcasting from the sender node is a
no-op.

For **edge-to-node** or **edge-to-context** pooling, we want a wrapper for layer
creation to supply more appropriate default values:

```python
def ExampleEdgePool(*args, sender_feature=tfgnn.HIDDEN_STATE, **kwargs):
  return ExampleConvolution(*args, sender_node_feature=None,
                            sender_edge_feature=sender_feature, **kwargs)
```

The `receiver_tag` can be set to `tfgnn.SOURCE`, `tfgnn.TARGET` or
`tfgnn.CONTEXT`. The layer can be called like `example_conv(graph,
edge_set_name="...")`. By default, `convolve()` gets the `sender_edge_input`
from the edge state and `receiver_input` from the node or context state, resp.;
it never gets a `sender_node_input`. Broadcast from and pool to receiver happen
between the edge set and the selected receiver (node set or context).

For more details, see the docstring of
`tfgnn.keras.layers.AnyToAnyConvolutionBase`. In particular, it describes the
optional `extra_receiver_ops` in case your custom convolution needs more than
broadcast and pool. (For example, convolutions with attention will often need
softmax across values that will go to the same receiver.) For a fully worked,
realistic example, see the `GATv2Conv` that is bundled with the TF-GNN
library.
