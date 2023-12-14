# Module: runner

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

A general purpose runner for TF-GNN.

## Classes

[`class ContextLabelFn`](./runner/ContextLabelFn.md): Reads out a `tfgnn.Field`
from the `GraphTensor` context.

[`class DatasetProvider`](./runner/DatasetProvider.md): Helper class that
provides a standard way to create an ABC using inheritance.

[`class DotProductLinkPrediction`](./runner/DotProductLinkPrediction.md):
Implements edge score as dot product of features of endpoint nodes.

[`class FitOrSkipPadding`](./runner/FitOrSkipPadding.md): Calculates fit or skip
`SizeConstraints` for `GraphTensor` padding.

[`class GraphBinaryClassification`](./runner/GraphBinaryClassification.md):
Graph binary (or multi-label) classification from pooled node states.

[`class GraphMeanAbsoluteError`](./runner/GraphMeanAbsoluteError.md): Regression
from pooled node states with mean absolute error.

[`class GraphMeanAbsolutePercentageError`](./runner/GraphMeanAbsolutePercentageError.md):
Regression from pooled node states with mean absolute percentage error.

[`class GraphMeanSquaredError`](./runner/GraphMeanSquaredError.md): Regression
from pooled node states with mean squared error.

[`class GraphMeanSquaredLogScaledError`](./runner/GraphMeanSquaredLogScaledError.md):
Regression from pooled node states with mean squared log scaled error.

[`class GraphMeanSquaredLogarithmicError`](./runner/GraphMeanSquaredLogarithmicError.md):
Regression from pooled node states with mean squared logarithmic error.

[`class GraphMulticlassClassification`](./runner/GraphMulticlassClassification.md):
Graph multiclass classification from pooled node states.

[`class GraphTensorPadding`](./runner/GraphTensorPadding.md): Collects
`GraphtTensor` padding helpers.

[`class GraphTensorProcessorFn`](./runner/GraphTensorProcessorFn.md): A class
for `GraphTensor` processing.

[`class HadamardProductLinkPrediction`](./runner/HadamardProductLinkPrediction.md):
Implements edge score as hadamard product of features of endpoint nodes.

[`class IntegratedGradientsExporter`](./runner/IntegratedGradientsExporter.md):
Exports a Keras model with an additional integrated gradients signature.

[`class KerasModelExporter`](./runner/KerasModelExporter.md): Exports a Keras
model (with Keras API) via `tf.keras.models.save_model`.

[`class KerasTrainer`](./runner/KerasTrainer.md): Trains using the
`tf.keras.Model.fit` training loop.

[`class KerasTrainerCheckpointOptions`](./runner/KerasTrainerCheckpointOptions.md):
Provides Keras Checkpointing related configuration options.

[`class KerasTrainerOptions`](./runner/KerasTrainerOptions.md): Provides Keras
training related options.

[`class ModelExporter`](./runner/ModelExporter.md): Saves a Keras model.

[`class NodeBinaryClassification`](./runner/NodeBinaryClassification.md): Node
binary (or multi-label) classification via structured readout.

[`class NodeMeanAbsoluteError`](./runner/NodeMeanAbsoluteError.md): Node
regression with mean absolute error via structured readout.

[`class NodeMeanAbsolutePercentageError`](./runner/NodeMeanAbsolutePercentageError.md):
Node regression with mean absolute percentage error via structured readout.

[`class NodeMeanSquaredError`](./runner/NodeMeanSquaredError.md): Node
regression with mean squared error via structured readout.

[`class NodeMeanSquaredLogScaledError`](./runner/NodeMeanSquaredLogScaledError.md):
Node regression with mean squared log scaled error via structured readout.

[`class NodeMeanSquaredLogarithmicError`](./runner/NodeMeanSquaredLogarithmicError.md):
Node regression with mean squared log error via structured readout.

[`class NodeMulticlassClassification`](./runner/NodeMulticlassClassification.md):
Node multiclass classification via structured readout.

[`class ParameterServerStrategy`](./runner/ParameterServerStrategy.md): A
`ParameterServerStrategy` convenience wrapper.

[`class PassthruDatasetProvider`](./runner/PassthruDatasetProvider.md): Builds a
`tf.data.Dataset` from a pass thru dataset.

[`class PassthruSampleDatasetsProvider`](./runner/PassthruSampleDatasetsProvider.md):
Builds a sampled `tf.data.Dataset` from multiple pass thru datasets.

[`class RootNodeBinaryClassification`](./runner/RootNodeBinaryClassification.md):
Root node binary (or multi-label) classification.

[`class RootNodeLabelFn`](./runner/RootNodeLabelFn.md): Reads out a
`tfgnn.Field` from the `GraphTensor` root (i.e. first) node.

[`class RootNodeMeanAbsoluteError`](./runner/RootNodeMeanAbsoluteError.md): Root
node regression with mean absolute error.

[`class RootNodeMeanAbsolutePercentageError`](./runner/RootNodeMeanAbsolutePercentageError.md):
Root node regression with mean absolute percentage error.

[`class RootNodeMeanSquaredError`](./runner/RootNodeMeanSquaredError.md): Root
node regression with mean squared error.

[`class RootNodeMeanSquaredLogScaledError`](./runner/RootNodeMeanSquaredLogScaledError.md):
Root node regression with mean squared log scaled error.

[`class RootNodeMeanSquaredLogarithmicError`](./runner/RootNodeMeanSquaredLogarithmicError.md):
Root node regression with mean squared logarithmic error.

[`class RootNodeMulticlassClassification`](./runner/RootNodeMulticlassClassification.md):
Root node multiclass classification.

[`class RunResult`](./runner/RunResult.md): Holds the return values of
`run(...)`.

[`class SampleTFRecordDatasetsProvider`](./runner/SampleTFRecordDatasetsProvider.md):
Builds a sampling `tf.data.Dataset` from multiple filenames.

[`class SimpleDatasetProvider`](./runner/SimpleDatasetProvider.md): Builds a
`tf.data.Dataset` from a list of files.

[`class SimpleSampleDatasetsProvider`](./runner/SimpleSampleDatasetsProvider.md):
Builds a sampling `tf.data.Dataset` from multiple filenames.

[`class SubmoduleExporter`](./runner/SubmoduleExporter.md): Exports a Keras
submodule.

[`class TFDataServiceConfig`](./runner/TFDataServiceConfig.md): Provides tf.data
service related configuration options.

[`class TFRecordDatasetProvider`](./runner/TFRecordDatasetProvider.md): Builds a
`tf.data.Dataset` from a list of files.

[`class TPUStrategy`](./runner/TPUStrategy.md): A `TPUStrategy` convenience
wrapper.

[`class Task`](./runner/Task.md): Defines a learning objective for a GNN.

[`class TightPadding`](./runner/TightPadding.md): Calculates tight
`SizeConstraints` for `GraphTensor` padding.

[`class Trainer`](./runner/Trainer.md): A class for training and validation of a
Keras model.

## Functions

[`export_model(...)`](./runner/export_model.md): Exports a Keras model without
traces s.t. it is loadable without TF-GNN.

[`incrementing_model_dir(...)`](./runner/incrementing_model_dir.md): Create,
given some `dirname`, an incrementing model directory.

[`integrated_gradients(...)`](./runner/integrated_gradients.md): Integrated
gradients.

[`one_node_per_component(...)`](./runner/one_node_per_component.md): Returns a
`Mapping` `node_set_name: 1` for every node set in `gtspec`.

[`run(...)`](./runner/run.md): Runs training (and validation) of a model on
task(s) with the given data.

## Type Aliases

[`Loss`](./runner/Loss.md)

[`Losses`](./runner/Losses.md)

[`Metric`](./runner/Loss.md)

[`Metrics`](./runner/Metrics.md)

[`Predictions`](./runner/Predictions.md)
