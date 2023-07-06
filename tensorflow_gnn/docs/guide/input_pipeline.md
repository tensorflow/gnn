# Input pipeline

## Introduction

The TF-GNN library supports high-performance file input and preprocessing of
training data for Graph Neural Networks. It uses the `tfgnn.GraphTensor` type
to represent graphs and their attached features. This doc expects that you have
read the [intro to GraphTensor](graph_tensor.md) already.
The two main topics covered here are:

  * Using tf.data to read many "structurally similar" GraphTensors in an
    efficient way from files with tf.Example protos.
  * Common techniques for transforming such GraphTensors into batches of
    training inputs.

For the preceding step of converting source data into TF-GNN's tf.Example format
and defining a schema for it, please see guides on
[Describing your graph](schema.md) and [Data preparation](data_prep.md).

This doc provides an in-depth explanation, sufficient to implement your own
input pipeline. For rapid experimentation, we recommend starting with the
[TF-GNN Runner](runner.md) and its built-in file input and model export.
(You can return here to find more details when needed.)


## File input and parsing

TF-GNN follows TensorFlow's general approach to store training data on disk as
records of serialized tf.Example protocol buffers, which are read into a
`tf.data.Dataset` and streamed through series of transformations, such as
shuffling and batching (covered by the tf.data documentation), parsing tensor
values, and applying problem-specific feature transformations.

A [`tfgnn.GraphTensor`](../api_docs/python/tfgnn/GraphTensor.md) is
a composite tensor (like `tf.SparseTensor` or `tf.RaggedTensor`),
so it can be used as a value in a Dataset. Like any composite tensor,
it comes with a subclass of `tf.TypeSpec`, called
[`tfgnn.GraphTensorSpec`](../api_docs/python/tfgnn/GraphTensorSpec.md),
which defines its node sets and edge sets, and contains the type specs
for all features attached to the graph.

The function `graph = tfgnn.parse_single_example(spec, serialized)`
[[doc](../api_docs/python/tfgnn/parse_single_example.md)] returns a
GraphTensor of shape `[]` that conforms to the given GraphTensorSpec and holds
the values parsed from the tf.string Tensor `serialized` of shape `[]`.
Likewise, for batched datasets, `graph = tfgnn.parse_example(spec, serialized)`
[[doc](../api_docs/python/tfgnn/parse_example.md)] parses a tf.string Tensor
of shape `[n]` into a GraphTensor of shape `[n]`, such that each element
`graph[i]` conforms to the given `spec`.

Notice that the GraphTensorSpec must be known up-front and passed into the
parsing code. It cannot be inferred from the tf.Example data. Besides,
tf.data.Dataset requires its element\_spec to be known before any data is read.
To address this, and similar needs of non-TensorFlow tools that handle graph
data, TF-GNN defines the `tfgnn.GraphSchema` protocol buffer. Whenever
GraphTensor data is saved to disk, we recommend serializing the applicable
GraphSchema with `protobuf.text_format.MessageToString()` and saving it to
a file `graph_schema.pbtxt` next to the actual graph data.

This lets you write code like

```python
dataset = tf.data.TFRecordDataset(filenames=["/tmp/my_graph/data.tfrecord"])
graph_schema = tfgnn.read_schema("/tmp/my_graph/graph_schema.pbtxt")
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
dataset = dataset.map(
    lambda serialized: tfgnn.parse_single_example(graph_tensor_spec, serialized))

for i, graph in enumerate(dataset.take(3)):
  print(f"Input {i}: {graph}")
```

Like tf.Example, `tfgnn.GraphSchema` supports tf.float32, tf.int64 and tf.string
as data types of features. In-process GraphTensors can hold features of any
dtype.

For datasets that have more than one GraphTensor in each example and/or combine
serialized GraphTensor(s) with other data, we recommend to distinguish each
graph tensor by a unique prefix of feature names when creating the tf.Examples
on disk, and to save their graph schemas separately. From the same serialized
input, each GraphTensor can be read out by one call to
`tfgnn.parse_single_example()` (or `tfgnn.parse_example()`) with the `prefix=`
argument set accordingly.


## The big picture: training, export and inference

Before we go deeper into preprocessing, let's think ahead to exporting the
trained model for inference.

For training, the standard approach is to pull all non-trainable transformations
out of the trained model, because it is faster to do them asynchronously in a
buffered Dataset than synchronously inside the training loop. Also, some
accelerators (notably TPUs) do not allow variable-length or string data inside
the trained model. For example, to represent a categorical string feature with
a trained embedding, the preprocessing of the dataset maps strings to indices,
and the main model looks up the index in the trained embedding table.

For inference, the story is different: The trainer exports one SavedModel, and
it depends on the inference platform how much of the preprocessing it should
contain. TF Serving recommends to do inference starting from a batch of
serialized tf.Example protos. It cannot even accept composite tensor inputs.
Also, as an RPC service, it works better if the model's interface is stable
(say, allows the original text representation of the categorical feature from
the example above) and encapsulates details that may vary from one trained
instance of the model to the next (like the mapping from strings to indices).

Therefore, we recommend defining *two* Keras models: a preprocessing model that
gets applied with `Dataset.map()` and the main model that gets trained on the
preprocessed dataset. At export time, Keras makes it easy to combine the two,
properly tracking any TensorFlow Resource objects needed for preprocessing
(such as lookup tables) and their initializers (such as vocabularies).

The following code snippet shows the high-level outline of this approach, using
Keras' Functional API to define the models by composing Keras layers, including
[`tfgnn.keras.layers.ParseExample`](../api_docs/python/tfgnn/keras/layers/ParseExample.md),
which is a thin Keras wrapper around `tfgnn.parse_example()` discussed in the
previous section.

```python
# Read the dataset of tf.Example protos for training.
dataset = tf.data.TFRecordDataset(filenames=[...])
dataset = ...  # Shuffle, repeat and prefetch as appropriate.
dataset = dataset.batch(batch_size)  # Expected before preprocessing.

# Parse the GraphTensor values.
graph_schema = tfgnn.read_schema("...")
example_input_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
dataset = dataset.map(tfgnn.keras.layers.ParseExample(example_input_spec))
preproc_input_spec = dataset.element_spec

# Define and apply the preprocessing model.
preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)
graph = ...  # With preprocessed features, see below.
graph = graph.merge_batch_to_components()  # See section "Merging a batch".
graph, labels = ... # See section "Splitting the label off ...".
preproc_model = tf.keras.Model(preproc_input, (graph, labels))
dataset = dataset.map(preproc_model)

# Define and train the main model.
model_input_spec, _ = dataset.element_spec  # Drop the spec for the labels.
model_input = tf.keras.layers.Input(type_spec=model_input_spec)
logits = ...  # Apply the GNN and read out logits.
model = tf.keras.Model(model_input, logits)
model.compile(...)
model.fit(dataset)

# Export the combined SavedModel for serving.
serving_input = tf.keras.layers.Input(shape=[],  # The batch dim is implied.
                                      dtype=tf.string, name="examples")
preproc_input = tfgnn.keras.layers.ParseExample(example_input_spec)(serving_input)
serving_model_input, _ = preproc_model(preproc_input)  # Drop labels.
serving_logits = model(serving_model_input)
serving_output = {"logits": tf.keras.layers.Layer(name="logits")(serving_logits)}
exported_model = tf.keras.Model(serving_input, serving_output)
exported_model.save("/tmp/exported_keras_model", include_optimizer=False)
```

This recipe can be varied according to the needs of the deployment platform.

The following sections of this document fills in the missing details regarding
`preproc_model`. Building the main `model` is discussed in the separate
[TF-GNN modeling guide](gnn_modeling.md), including the creation of initial
hidden states of the GNN from input features.

About the exported model, there is not much to say: It's a Keras SavedModel,
with serialized tf.Examples as inputs and a dict of predictions as outputs. As
usual, there are three ways to use the SavedModel:

  * Restoring as a TensorFlow SavedModel in terms of SignatureDefs, as done by
    TF Serving and other C++ environments: this works as always. TF-GNN does not
    bring in any custom ops (as of May 2022); all its functionality is
    implemented in Python in terms of standard TF ops.
  * Restoring into a Python environment with `tf.saved_model.load()` as nested
    objects with tf.functions or with `tf.keras.models.load_model()` as a
    fully-fledged Keras model: this requires a compatible version of the TF-GNN
    library to be imported up-front, in order to define the GraphTensor type
    and, for the latter case, TF-GNN's Keras layers.

By the way, the odd duplication of `"logits"` in the dict of `serving_outputs`
above is unrelated to TF-GNN: It addresses an oddity in TensorFlow whereby the
result of `tf.saved_model.load()` expects a dict of inputs with the keys as
given, while the SignatureDef seen by TF Serving uses the name of the last layer
as its keys.


## Feature preprocessing

### The MapFeatures layer

Typically, feature preprocessing happens locally on nodes and edges, without
regard for adjacencies in the graph. It can employ a variety of techniques for
different types of data. TF-GNN strives to reuse standard Keras implementations
for these. To that end, the
[`tfgnn.keras.layers.MapFeatures`](../api_docs/python/tfgnn/keras/layers/MapFeatures.md)
layer lets you express feature transformations on the graph as a collection of
feature transformations for the various graph pieces (node sets, edge sets, and
context), each constructed by a Python callback.

As an example, here is how you can replace feature `"id"` with `"hashed_id"` on
all node sets, and pass through other features unchanged:

```python
def node_sets_fn(node_set, *, node_set_name):
  features = node_set.get_features_dict()

  # Replace "id" by "hashed_id".
  ids = features.pop("id")
  num_bins = 100000 if node_set_name == "docs" else 20000
  features["hashed_id"] = tf.keras.layers.Hashing(num_bins=num_bins)(ids)

  # More features could be handled here.

  return features

graph = tfgnn.keras.layers.MapFeatures(
    node_sets_fn=node_sets_fn)(preproc_input_graph)
```

After we get a mutable copy of the dict of input features in the first line,
the rest of the callback is all about transforming a dict of tensors, free from
GraphTensor technicalities. The same callback is used for all node sets, with
the node set name as a side input, so there is a lot of flexibility to structure
the code around feature names (as shown, with the node set merely adjusting the
`num_bins` hyperparameter), or around the separate node sets, or a mix of both.

The body of the callback uses Keras'
[Functional API](https://www.tensorflow.org/guide/keras/functional).
The `node_set` argument is a Keras tensor wrapping a `tfgnn.NodeSet`. You can
call its methods as usual to get (nests of) Keras tensors for its fields and
then apply arbitrary Keras layer to them. Keras will take care of tracking them
as layers of your model.

The MapFeatures layer accepts not only a `node_sets_fn(node_set, node_set_name)`
but also an `edge_sets_fn(edge_set, edge_set_name)` and a `context_fn(context)`
to define feature transformations on those graph pieces. You can even pass one
function `fn(graph_piece, **kwargs)` for all three arguments to centralize all
preprocessing logic in one place. The docstring of MapFeatures has more details
and examples.

We recommend calling MapFeatures once inside your preprocessing model (that is,
from within `Dataset.map()`) and once more to initialize the hidden states of
the GNN, at the start of the main model (see the
[modeling guide](gnn_modeling.md)), each time with callbacks that work through
the laundry lists of all feature transformations (possibly controlled by some
higher-level configuration).

By default, MapFeatures ignores `"_readout"` and other auxiliary node sets
whose name starts with `_`. If you need to process some of them (say, to apply
a table lookup to a string-valued label feature), you need to pass a suitable
[regular expression](https://docs.python.org/3/library/re.html?#regular-expression-syntax)
as `MapFeatures(..., allowed_aux_node_sets_pattern=r"...")`. Likewise for
edge sets.


### The shape of features

There is a subtlety in the code above regarding tensor shapes: Preprocessing
starts on a GraphTensor of shape `[batch_size]`, in which the nodes and edges
are indexed separately as 0,1,2,... in each node/edge set for each input
example. The sizes of node sets and edge sets are allowed to vary between the
input examples, so their features become RaggedTensors of shape
`[batch_size, (set_size), ...]` with further dimensions (possibly ragged)
according to the per-node/edge shape of the feature.

Most Keras layers for preprocessing "just work" with ragged inputs like these,
notably Discretization, Hashing, Rescaling, IntegerLookup and StringLookup.

If not, there are several ways to work around it:

 1. Temporarily merge `batch_size` and `set_size` into one dimension to get
    shape `[total_size, ...]` with one less ragged dimension while applying
    some `layer`. This can be done by
    `tf.keras.layers.TimeDistributed(layer)(feature)` (whose name alludes to the
    use of axis 1 for time steps in a sequence, where we have nodes/edges in a
    graph), or manually with the tf.RaggedTensor methods
    `feature.from_values(layer(feature.values))`, which are also available on
    the symbolic tensors of Keras' Functional API. This only makes sense for
    transformations that do not depend on the boundaries between input examples.
 2. Defer the transformation after the call to `.merge_batch_to_components()`
    described in the next section, which removes the ragged `set_size` dimension
    permanently. This also works best for transformations that do not depend on
    the boundaries between input examples.
 3. Write a custom Keras layer that calls `tf.map_fn(fn, feature, ...)` to apply
    `fn` separately on the feature for each example in the batch, shaped
    `[set_size, ...]`. It also works for a nest of multiple features. This lets
    you handle one example at a time. Under the hood, it uses a tf.while\_loop
    over the batch.


## Merging a batch of inputs into components of one graph

For training the model, TF-GNN requires each batch of input graphs to be merged
into one single graph in which the nodes of each node set and the edges of each
edge set are indexed contiguously as 0, 1, 2, .... across the input examples.
This way, broadcasting data from nodes to edges and pooling data from edges to
nodes (the key operations for GNNs) can work on one flat index space. No edges
are added in the process, so the input graphs remain disconnected from each
other.

To keep track of the boundaries between inputs, GraphTensor supports the notion
of **graph components**, which are explained in greater detail by the
[intro to GraphTensor](graph_tensor.md). In a nutshell, their story is this:

Each graph has some number of components; for input graphs, typically that
number is 1.
Each NodeSet and EdgeSet stores its size not as one number of nodes/edges per
graph, but as a vector with the number of nodes/edges in each component. Merging
a batch performs concatenation, not addition, of these sizes.

For example, consider reading three GraphTensors from disk with 4, 5 and 6
nodes, resp., in the NodeSet `"docs"`. Parsing them as a batch of size 3 creates
a GraphTensor of shape `[3]` with `graph.node_sets["docs"].sizes` equal to
`[[4], [5], [6]]`. The edges in each graph refer to node indices 0,...,3;
0,...,4; and 0,...,5, respectively. Likewise, node features have a shape
`[3, (node_indices), ...]` where `(node_indices)` is a ragged dimension.
The result of `graph.merge_batch_to_components()` is a new GraphTensor with
shape `[]`, node set sizes `[4, 5, 6]`, node indices 0,...,14, and feature
shape `[15, ...]`, with nodes concatenated in order.

The context features in a GraphTensor are stored per component, not per graph.
Pooling from nodes/edges to the context and likewise broadcasting from the
context to nodes/edges respects the breakdown of node/edge sets per component.
This, together with the absence of edges between components, makes sure that the
standard building blocks for graph neural networks (broadcast/pool between
edges/nodes/context) do not leak information between the examples in a batch of
inputs, even after the batch of graphs has been merged into components of a
single graph.


## Splitting the labels out of the graph

TF-GNN's machinery for processing graph data uses `GraphTensor` and its
serialization format a lot. Therefore, training labels typically enter the input
pipeline as a feature on some piece of a `GraphTensor`. We suggest to transform
training labels alongside features on the graph and split them off at the end
of preprocessing, once the `GraphTensor` and its other features have reached
their final shape.

The code for doing this varies with the exact location of the label feature.

If the input data contains a `"_readout"` node set, and if the labels have been
stored as feature, say, `"class_id"` on that node set, they can be split out
as follows:

```python
labels = tfgnn.keras.layers.Readout(node_set_name="_readout",
                                    feature_name="class_id")(graph)
graph = graph.remove_features(node_sets={"_readout": ["class_id"]})
assert "class_id" not in graph.node_sets["_readout"].features
```

The removal of the feature can be skipped if the rest of the code makes sure
that the GNN model does not get to see the `"_readout"` node set and its
features.

On the other hand, if the label is stored on an ordinary node set, say `"docs"`,
it needs to be read out from those nodes for which a prediction will
be made. The code pattern for that looks similar to the readout of final
node states from the GNN, which is discussed in more detail in the
[modeling guide](gnn_modeling.md).

If an auxiliary "_readout" node set is present that references the `"seed"`
nodes of `"docs"`, their labels can be split out like

```python
labels = tfgnn.keras.layers.StructuredReadout("seed", feature_name="class_id")(graph)
graph = graph.remove_features(node_sets={"docs": ["class_id"]})
assert "class_id" not in graph.node_sets["docs"].features
```

If the input dataset relies on the pre-`"_readout"` convention to simply
do readout from the first `"docs"` node of each sampled subgraph, the code
would look like

```python
labels = tfgnn.keras.layers.ReadoutFirstNode(node_set_name="docs",
                                             feature_name="class_id")(graph)
graph = graph.remove_features(node_sets={"docs": ["class_id"]})
assert "class_id" not in graph.node_sets["docs"].features
```

At serving time, inputs will be missing the `"labels"` feature, of course.
The preprocessing model needs to handle that gracefully (maybe based on the
default values supplied by feature parsing), or else needs to be rebuilt for
export with an option to switch off handling the label.


## Distributed training

Keras supports data-parallel training through TensorFlow's Distribution Strategy
API like this:

```python
strategy = ....  # As appropriate for the available hardware.
with strategy.scope():
  # The model and all its layers are created under the strategy scope.
  model = ...
model.fit(dataset)
```

The `dataset` provides the training data for each replica in each step.
There are fundamentally two ways how to do this:

1.  If `dataset` is a plain `tf.data.Dataset`, each element is interpreted as a
    logical batch of training inputs for all replicas, so Keras needs to go in
    and rewire the dataset after the fact to produce per-replica batches. This
    works for tensors of shapes that all start with `[batch_size, ....]`.
2.  If `dataset` is a `tf.distribute.DistributedDataset`, it is a tuple of
    actual `Datasets`, one per replica, and their elements provide the
    ready-to-use training inputs, whatever their shape.

The GraphTensors created by `.merge_batch_to_components()` no longer have a
shape starting with `[batch_size, ....]`. Still, the first approach can be made
to work: in the absence of the fixed-shape requirement from Cloud TPUs (see
below), the dataset can feed batched GraphTensors to the trained Model, and
`.merge_batch_to_components()` is called at the start of the model itself.
For TPUs, users could try to batch, merge and pad per replica, then batch again
with the number of replicas as batch size, and finally get rid of the useless
leading dimension `[1, ...]` on each replica. However, this gets complex.

Therefore, the rest of this doc follows the second approach, which is more
general and more straightforward. A DistributedDataset is created from a
callback that gets invoked for each replica (possibly on different hosts)
like so:

```python
def _make_dataset(input_context):
  ds = ...  # As usual, but from the subset of input files for this replica.
  ds = ds.batch(input_context.get_per_replica_batch_size(global_batch_size))
  ds = dataset.map(tfgnn.keras.layers.ParseExample(example_input_spec))
  # Apply preprocessing.
  preproc_model = ...  # Built in this context, code as above.
  ds = ds.map(preproc_model)
  return ds
dataset = strategy.distribute_datasets_from_function(_make_dataset)
```


## Training on Cloud TPU

TF-GNN supports training on Google's
[Tensor Processing Units](https://www.tensorflow.org/guide/tpu) (TPUv2+) using
the `tf.distribute.TPUStrategy`. The just-in-time compiler for TPUs requires
that all component tensors of a GraphTensor are Tensors (not RaggedTensors) and
that their shapes are statically known. (To be precise, the
`tf.keras.layers.Input` object at the start of the model may have a somewhat
flexible type\_spec to allow different sizes for training, eval and export, but
the Datasets passed to `tf.keras.Model.fit()` and `.evaluate()` must have
statically known shapes.)


### Padding

Input graphs typically have variable sizes of node sets and edge sets, and so do
the GraphTensor inputs to a model that are created by merging a batch of input
graphs (see above). Users need to call
[`tfgnn.keras.layers.PadToTotalSizes`](../api_docs/python/tfgnn/keras/layers/PadToTotalSizes.md)
on it (or the underlying `tfgnn.pad_to_total_sizes()` function) to fill in nodes
etc. until all size constraints are met exactly. All padding items go into a
separate graph component (or multiple ones, if one component is not enough to
fill up the number of components).

If the model expects a minimum number of nodes per component in a particular
node set, say, to use `tfgnn.keras.layers.ReadoutFirstNode`, this must hold for
padding components as well. To that end, the size constraints need to be created
with `min_nodes_per_component = {node_set_name: min_nodes, ...}` as shown below.

Next to the padded graph, `PadToTotalSizes` returns a boolean mask that's `True`
for input components and `False` for padding components. The mask is a
component-indexed Tensor like a context feature, so broadcasting from context to
some node or edge set converts it to a mask on those nodes or edges. In a
training and validation dataset, Keras accepts weights as a third item after
model inputs and labels, and a boolean mask converts automatically to zero/one
weights. In the training code (not shown in this doc), be sure to use
`Model.compile(weighted_metrics=...)` instead of plain `metrics=...` so that
the mask takes effect not just for the loss but also for the metrics.

Padding fits as follows into the skeleton code shown above:

```python
def _make_dataset(input_context):
  ds = ... # As above.
  ds = ds.batch(per_replica_batch_size)
  ds = dataset.map(tfgnn.keras.layers.ParseExample(example_input_spec))
  # Apply preprocessing.
  preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)  # As before.
  graph = ...  # Feature preprocessing, as before.
  graph = graph.merge_batch_to_components()  # As before.
  graph, mask = tfgnn.keras.layers.PadToTotalSizes(size_constraints)(graph)
  graph, labels = ... # Splitting the label off the *padded* tensor.
  mask = ...  # If necessary, broadcast from context to align with labels.
  preproc_model = tf.keras.Model(preproc_input, (graph, labels, mask))
  ds = ds.map(preproc_model)
  return ds

dataset = strategy.distribute_datasets_from_function(_make_dataset)
```


### Setting the size constraints and a batching strategy

The simplest way to set the `size_constraints` for a dataset is to scan all
input examples from the dataset, find for each tensor the maximum size it needs,
and multiply that by the desired batch size. On top of that comes some allowance
for how padding one thing needs room elsewhere (e.g., padding an edge set
requires room for nodes as endpoints and a padding component; adding a padding
component requires adding each node set's minimum number of nodes per
component).

This simple approach is implemented by
`tfgnn.find_tight_size_constraints(dataset, target_batch_size=...,
min_nodes_per_component=...)`
[[doc](../api_docs/python/tfgnn/find_tight_size_constraints.md)],
a utility function to be run on the dataset before actual training starts.
This is necessary if you need to accommodate all examples of a dataset
(say, for validation), irrespective of how they come together in batches
(which may vary for Datasets that use a non-deterministic order for speed).
It also works well if the maximum size of each tensor is "close enough" to
the mean size. For a dataset of sampled subgraphs, it may be possible to
tighten the sampler limits towards that.

However, many practical applications have infrequent but important examples with
large sizes, and allocating space for the rare coincidence of a whole batch of
infrequent large examples is very wasteful: in terms of accelerator memory, but
also in terms of all the computation wasted on padding elements. TF-GNN offers
two ways around this:

  * Dynamic batching: `tfgnn.dynamic_batch()` replaces the usual
    `Dataset.batch()` and batches as many consecutive graphs as will fit the
    constraints. However, it comes with a substantial delay if the effective
    batch size is large, and we do not recommend it unconditionally at this
    time.
  * "Overbooking": `tfgnn.learn_fit_or_skip_size_constraints()` lets you find
    size constraints that will "almost always" suffice for a batch of a fixed
    batch\_size. You can specify one or more desired success rates and batch
    sizes and explore the resulting size constraints for each combination (e.g.,
    the size constraint that fits a batch of size 100 with probability 0.99).

Using the fit-or-skip ("overbooking") approach consists of three steps:

At the start of the trainer program, before the actual model training,
**determine the size constraints from a random sample** of the input data, with
hparams tweaked to the problem at hand.

```python
sample_dataset = ...  # The input examples, not batched yet.
sample_dataset = sample_dataset.shuffle(...)  # Need a random sample.
sample_dataset = sample_dataset.map(
    tfgnn.keras.layers.ParseExample(example_input_spec))
size_constraints = tfgnn.learn_fit_or_skip_size_constraints(
    sample_dataset, per_replica_batch_size, min_nodes_per_component={...},
    success_ratio=0.99, sample_size=20000)
```

Extend construction of the training dataset with a filter to **skip oversized
batches** that would fail in padding, and **pad remaining batches** as before,
but with the tighter constraints.

```python
def _make_dataset(input_context):
  ds = ... # As above.
  ds = ds.batch(per_replica_batch_size)
  ds = dataset.map(tfgnn.keras.layers.ParseExample(example_input_spec))
  ds = tfgnn.dataset_filter_with_summary(
      ds, functools.partial(tfgnn.satisfies_total_sizes,
                            total_sizes=size_constraints))
  # Apply preprocessing.
  preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)  # As before.
  graph = ...  # Feature preprocessing, as before.
  graph = graph.merge_batch_to_components()  # As before.
  graph, mask = tfgnn.keras.layers.PadToTotalSizes(size_constraints)(graph)
  graph, labels = ... # Splitting the label off the *padded* tensor, as before.
  preproc_model = tf.keras.Model(preproc_input, (graph, labels, mask))
  ds = ds.map(preproc_model)
  return ds
```

This technique requires careful monitoring of the effects of filtering. Outdated
or incorrect size\_constraints could silently discard important parts of the
input data. To that end, the helper function
`ds = tfgnn.dataset_filter_with_summary(ds, predicate)`
[[doc](../api_docs/python/tfgnn/dataset_filter_with_summary.md)]
returns `ds.filter(predicate)`, but with a side output of the removed fraction
to TensorBoard. Even so, we primarily recommend this approach as a speed
improvement for training, not for testing.
