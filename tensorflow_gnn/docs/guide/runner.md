# The TF-GNN Runner

## Overview

*TensorFlow GNN (TF-GNN)* provides a general purpose graph learning codebase in
the `./runner` package. The `runner` (also know and stylzed as *Orchestrator*)
orchestrates the end to end training of models implemented with *TF-GNN*. This
toolkit is intended to support *TF-GNN* modeling by addressing (and solving)
common technical pain points: It aims to enable the practice of state of the art
Graph Neural Network techniques, research and benchmarks. With out of the box
solutions for data reading and processing; common graph learning objectives like
graph generation, [`Deep Graph InfoMax`](https://arxiv.org/abs/1809.10341) and
node/graph classification;
[`Integrated Gradients`](https://research.google/pubs/pub49909/) and `Cloud
TPU`, the codebase aspires to empower the programmer in any graph learning
application.

This document introduces the package's abstractions and how to best use them for
quick start graph learning in *TF-GNN*.

## Quick Start

For programmers motivated to jump right in, the following snippet demonstrates
end to end data reading, feature processing, model training, model validation
and model export using the *Orchestrator*:

```python
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner


graph_schema = tfgnn.read_schema("/tmp/graph_schema.pbtxt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

# len(train_ds_provider.get_dataset(...)) == 8191.
train_ds_provider = runner.TFRecordDatasetProvider(file_pattern="...")
# len(valid_ds_provider.get_dataset(...)) == 1634.
valid_ds_provider = runner.TFRecordDatasetProvider(file_pattern="...")

# Use `embedding` feature as the only node feature.
initial_node_states = lambda node_set, node_set_name: node_set["embedding"]

map_features = tfgnn.keras.layers.MapFeatures(node_sets_fn=initial_node_states)

# Binary classification by the root node.
task = runner.RootNodeBinaryClassification(
    "nodes",
    label_fn=runner.ContextLabelFn("label"))

trainer = runner.KerasTrainer(
    strategy=tf.distribute.TPUStrategy(...),
    model_dir="...",
    steps_per_epoch=8191 // 128,  # global_batch_size == 128
    validation_per_epoch=2,
    validation_steps=1634 // 128)  # global_batch_size == 128

runner.run(
    train_ds_provider=train_ds_provider,
    train_padding=runner.FitOrSkipPadding(gtspec, train_ds_provider),
    # model_fn is a function: Callable[[tfgnn.GraphTensorSpec], tf.keras.Model].
    # Where the returned model both takes and returns a scalar `GraphTensor` for
    # its inputs and outputs.
    model_fn=model_fn,
    optimizer_fn=tf.keras.optimizers.Adam,
    epochs=4,
    trainer=trainer,
    task=task,
    gtspec=gtspec,
    global_batch_size=128,
    feature_processors=[map_features],
    valid_ds_provider=valid_ds_provider)
```

The rest of this document introduces and explains the above building blocks, how
to reuse them and how to implement your own. For an example of `model_fn` and
the orchestration entry point (`runner.run`), skip to the [end](#orchestration)
of this document.

## The Toolkit (and its Building Blocks)

Running (an all inclusive term for everything from data reading to training,
validation and export) is orchestrated by four abstractions: the
`DatasetProvider`, `Task`, `Trainer` and `GraphTensorProcessorFn`. The runner
provides instances for common cases (e.g., the `TFRecordDatasetProvider`, the
`NodeClassification` task, the `KerasTrainer`), but collaborators are free to
define their own. Each abstraction is introduced and explained below.

### Data Reading

```python
class DatasetProvider(abc.ABC):

  @abc.abstractmethod
  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    raise NotImplementedError()
```

The `DatasetProvider` provides a `tf.data.Dataset`. The returned `Dataset` is
expected not to be batched and contain serialized `tf.Examples` of
`GraphTensor`. Any sharding according to the input `tf.distribute.InputContext`
is left to the implementation. Two implementations for reading from disk (with
pattern matching by `tf.io.gfile.glob` and arbitrary interleaving by
`tf.data.Dataset.interleave`) are provided:

*   `SimpleDatasetProvider`, for reading and interleaving files matching a
    pattern,
*   `SimpleSampleDatasetsProvider`, for reading, interleaving and sampling files
    matching several different patterns.

Two implementations for reading `TFRecord` from disk (with pattern matching by
`tf.io.gfile.glob`) are provided:

*   `TFRecordDatasetProvider`, for reading `TFRecord` files matching a pattern,
*   `SampleTFRecordDatasetsProvider`, for reading and sampling `TFRecord` files
    matching several different patterns.

Contributors have free rein in their implementation of `get_dataset`, e.g.:
in-memory generation of synthetic graphs or real time conversion of different
graph persistence formats.

### Task Preprocessing, Prediction and Objectives

```python
class Task(abc.ABC):

  @abc.abstractmethod
  def preprocess(self, inputs: GraphTensor) -> tuple[Union[GraphTensor, Sequence[GraphTensor]], Field]:
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(self, *args: GraphTensor) -> Field:
    raise NotImplementedError()

  @abc.abstractmethod
  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    raise NotImplementedError()

  @abc.abstractmethod
  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    raise NotImplementedError()
```

A `Task` represents a learning objective for a GNN model and defines all the
non-GNN pieces around the base GNN. A `Task` is expected to define preprocessing
for a `tf.data.Dataset` (of `GraphTensor`) and produce prediction outputs (via
`predict(...)`). `predict(...)` typically performs the addition of the readout
and prediction heads (see step 3 of
[The big picture](gnn_modeling.md#the-big-picture-initialization-graph-updates-and-readout)).
The `Task` also provides any losses and metrics for that objective. Common
implementations for classification and regression (by graph or root node) are
provided:

*   `GraphBinaryClassification`,
*   `GraphMeanAbsoluteError`,
*   `GraphMeanAbsolutePercentageError`,
*   `GraphMeanSquaredError`,
*   `GraphMeanSquaredLogarithmicError`,
*   `GraphMeanSquaredLogScaledError`,
*   `GraphMulticlassClassification`,
*   `RootNodeBinaryClassification`,
*   `RootNodeMeanAbsoluteError`,
*   `RootNodeMeanAbsolutePercentageError`,
*   `RootNodeMeanSquaredError`,
*   `RootNodeMeanSquaredLogarithmicError`,
*   `RootNodeMeanSquaredLogScaledError`,
*   `RootNodeMulticlassClassification`.

Collaborators may contribute new graph learning objectives with a Python
`object` that subclasses `Task` and implements its abstract methods. For
example, an imagined `RadiaInfomax` that:

*   For a dataset,

    *   Masks arbitrary nodes,
    *   Creates psuedo labels;

*   For an arbitrary input (where that input is the base GNN output for those
    `GraphTensor` returned by `preprocess(...)`),

    *   Adds a head to `R^4` from the root node hidden state;

*   For a loss and metrics,

    *   Uses cosine similarity.

```python
class RadiaInfomax(runner.Task):

  def preprocess(self, inputs: GraphTensor) -> tuple[GraphTensor, Field]:
    return mask_some_nodes(gt), create_psuedolabels()

  def predict(self, inputs: GraphTensor) -> Field:
    # A single `GraphTensor` input corresponding to the base GNN output given
    # the `GraphTensor` returned by `preprocess(...)`.
    tfgnn.check_scalar_graph_tensor(inputs, name="RadiaInfomax")
    activations = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="nodes",
        feature_name=tfgnn.HIDDEN_STATE)(inputs)
    return tf.keras.layers.Dense(
        4,  # Apply RadiaInfomax in R^4.
        name="logits")(activations)

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return [tf.keras.losses.CosineSimilarity(),]

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return [tf.keras.metrics.CosineSimilarity(),]
```

### Training

```python
class Trainer(abc.ABC):

  @property
  @abc.abstractmethod
  def model_dir(self) -> str:
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def strategy(self) -> tf.distribute.Strategy:
    raise NotImplementedError()

  @abc.abstractmethod
  def train(
      self,
      model_fn: Callable[[], tf.keras.Model],
      train_ds_provider: DatasetProvider,
      *,
      epochs: int = 1,
      valid_ds_provider: Optional[DatasetProvider] = None) -> tf.keras.Model:
    raise NotImplementedError()
```

A `Trainer` provides any training and validation loops. These may be uses of
`tf.keras.Model.fit` or arbitrary custom training loops. The `Trainer` provides
accessors to training properties (like its `tf.distribute.Strategy` and
`model_dir`) and is expected to return a trained `tf.keras.Model`. A version of
the Keras `fit` training loop is provided with extra functionality (like
performing validation more than once per epoch):

*   `KerasFit`.

Collaborators may contribute new training and validation loops with a Python
object that subclasses `Trainer` and implements its abstract methods. For
example, a custom training loop with look ahead gradients.

### GraphTensor Processing

```python
class GraphTensorProcessorFn(Protocol):

  def __call__(self, inputs: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    raise NotImplementedError()
```

Any Python callable of such signature&mdash;`GraphTensor` ->
`Union[tfgnn.GraphTensor, Tuple[tfgnn.GraphTensor, tfgnn.Field]]`&mdash;is
valid.

A `GraphTensorProcessorFn` performs feature processing on the `GraphTensor` of a
dataset. Importantly: all `GraphTensorProcessorFn` are applied in a
`tf.data.Dataset.map` call (and correspondingly executed on CPU). All
`GraphTensorProcessorFn` are collected in a `tf.keras.Model` specifically for
feature processing. The final model exported by [orchestration](#orchestration)
will contain both the feature processing model and the client GNN.

TIP: A `tf.keras.Model` or `tf.keras.layers.Layer`, whose inputs and outputs are
scalar `GraphTensor`, matches the `GraphTensorProcessorFn` protocol (and may be
used as one).

BEST PRACTICE: `tfgnn.keras.layers.MapFeatures` is a `tf.keras.layers.Layer`
like described. Use it for all your feature processing!

## Orchestration

Orchestration (a term for the composition, wiring and execution of the above
abstractions) happens via a single `run` method with signature:

```python
def run(*,
        train_ds_provider: DatasetProvider,
        model_fn: Callable[[tfgnn.GraphTensorSpec], tf.keras.Model],
        optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
        trainer: Trainer,
        task: Task,
        gtspec: tfgnn.GraphTensorSpec,
        global_batch_size: int,
        epochs: int = 1,
        drop_remainder: bool = False,
        export_dirs: Optional[Sequence[str]] = None,
        model_exporters: Optional[Sequence[ModelExporter]] = None,
        feature_processors: Optional[Sequence[GraphTensorProcessorFn]] = None,
        valid_ds_provider: Optional[DatasetProvider] = None,
        train_padding: Optional[GraphTensorPadding] = None,
        valid_padding: Optional[GraphTensorPadding] = None,
        tf_data_service_config: Optional[TFDataServiceConfig] = None):
```

The `model_fn` is expected to take a `tfgnn.GraphTensorSpec` and return a
`tf.keras.Model` whose inputs and outputs are scalar `GraphTensor` (see steps
1-2 of
[The big picture](gnn_modeling.md#the-big-picture-initialization-graph-updates-and-readout)),
`export_dirs` are locations for a trained and saved model and any
`feature_processors` are applied in sequence order. All other arguments may be
supplied with out of the box or custom implementations of the respective
protocol or base class.

An example `model_fn` built with a ready-to-use Model Template:

```python
from tensorflow_gnn.models import mt_albis

def model_fn(gtspec: tfgnn.GraphTensorSpec):
  """Builds a GNN from Model Template "Albis"."""
  graph = inputs = tf.keras.layers.Input(type_spec=gtspec)
  for _ in range(4):
    graph = mt_albis.MtAlbisGraphUpdate(
        units=32,
        message_dim=32,
        receiver_tag=tfgnn.SOURCE,
        # More hyperparameters like edge_dropout_rate can be added here.
    )(graph)
  return tf.keras.Model(inputs, graph)
```
