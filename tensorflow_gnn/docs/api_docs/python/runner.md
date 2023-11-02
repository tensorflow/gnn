<!-- lint-g3mark -->

# Module: runner

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

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
Classification by the label provided in the graph context.

[`class GraphMeanAbsoluteError`](./runner/GraphMeanAbsoluteError.md): Mean
absolute error task.

[`class
GraphMeanAbsolutePercentageError`](./runner/GraphMeanAbsolutePercentageError.md):
Mean absolute percentage error task.

[`class GraphMeanSquaredError`](./runner/GraphMeanSquaredError.md): Mean squared
error task.

[`class
GraphMeanSquaredLogScaledError`](./runner/GraphMeanSquaredLogScaledError.md):
Mean squared log scaled error task.

[`class
GraphMeanSquaredLogarithmicError`](./runner/GraphMeanSquaredLogarithmicError.md):
Mean squared logarithmic error task.

[`class
GraphMulticlassClassification`](./runner/GraphMulticlassClassification.md):
Classification by the label provided in the graph context.

[`class GraphTensorPadding`](./runner/GraphTensorPadding.md): Collects
`GraphtTensor` padding helpers.

[`class GraphTensorProcessorFn`](./runner/GraphTensorProcessorFn.md): A class
for `GraphTensor` processing.

[`class
HadamardProductLinkPrediction`](./runner/HadamardProductLinkPrediction.md):
Implements edge score as hadamard product of features of endpoint nodes.

[`class IntegratedGradientsExporter`](./runner/IntegratedGradientsExporter.md):
Exports a Keras model with an additional integrated gradients signature.

[`class KerasModelExporter`](./runner/KerasModelExporter.md): Exports a Keras
model (with Keras API) via `tf.keras.models.save_model`.

[`class KerasTrainer`](./runner/KerasTrainer.md): Trains using the
`tf.keras.Model.fit` training loop.

[`class
KerasTrainerCheckpointOptions`](./runner/KerasTrainerCheckpointOptions.md):
Provides Keras Checkpointing related configuration options.

[`class KerasTrainerOptions`](./runner/KerasTrainerOptions.md): Provides Keras
training related options.

[`class ModelExporter`](./runner/ModelExporter.md): Saves a Keras model.

[`class ParameterServerStrategy`](./runner/ParameterServerStrategy.md): A
`ParameterServerStrategy` convenience wrapper.

[`class PassthruDatasetProvider`](./runner/PassthruDatasetProvider.md): Builds a
`tf.data.Dataset` from a pass thru dataset.

[`class
PassthruSampleDatasetsProvider`](./runner/PassthruSampleDatasetsProvider.md):
Builds a sampled `tf.data.Dataset` from multiple pass thru datasets.

[`class
RootNodeBinaryClassification`](./runner/RootNodeBinaryClassification.md):
Classification by root node label.

[`class RootNodeLabelFn`](./runner/RootNodeLabelFn.md): Reads out a
`tfgnn.Field` from the `GraphTensor` root (i.e. first) node.

[`class RootNodeMeanAbsoluteError`](./runner/RootNodeMeanAbsoluteError.md): Mean
absolute error task.

[`class
RootNodeMeanAbsoluteLogarithmicError`](./runner/RootNodeMeanAbsoluteLogarithmicError.md):
Root node mean absolute logarithmic error task.

[`class
RootNodeMeanAbsolutePercentageError`](./runner/RootNodeMeanAbsolutePercentageError.md):
Mean absolute percentage error task.

[`class RootNodeMeanSquaredError`](./runner/RootNodeMeanSquaredError.md): Mean
squared error task.

[`class
RootNodeMeanSquaredLogScaledError`](./runner/RootNodeMeanSquaredLogScaledError.md):
Mean squared log scaled error task.

[`class
RootNodeMeanSquaredLogarithmicError`](./runner/RootNodeMeanSquaredLogarithmicError.md):
Mean squared logarithmic error task.

[`class
RootNodeMulticlassClassification`](./runner/RootNodeMulticlassClassification.md):
Classification by root node label.

[`class RunResult`](./runner/RunResult.md): Holds the return values of
`run(...)`.

[`class
SampleTFRecordDatasetsProvider`](./runner/SampleTFRecordDatasetsProvider.md):
Builds a sampling `tf.data.Dataset` from multiple filenames.

[`class SimpleDatasetProvider`](./runner/SimpleDatasetProvider.md): Builds a
`tf.data.Dataset` from a list of files.

[`class
SimpleSampleDatasetsProvider`](./runner/SimpleSampleDatasetsProvider.md): Builds
a sampling `tf.data.Dataset` from multiple filenames.

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

[`incrementing_model_dir(...)`](./runner/incrementing_model_dir.md): Create,
given some `dirname`, an incrementing model directory.

[`integrated_gradients(...)`](./runner/integrated_gradients.md): Integrated
gradients.

[`one_node_per_component(...)`](./runner/one_node_per_component.md): Returns a
`Mapping` `node_set-name: 1` for every node set in `gtspec`.

[`run(...)`](./runner/run.md): Runs training (and validation) of a model on
task(s) with the given data.

## Type Aliases

[`Loss`](./runner/Loss.md)

[`Losses`](./runner/Losses.md)

[`Metric`](./runner/Loss.md)

[`Metrics`](./runner/Metrics.md)

[`Predictions`](./runner/Predictions.md)
