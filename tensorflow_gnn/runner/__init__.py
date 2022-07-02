"""A general purpose runner for TF-GNN."""
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.input import datasets
from tensorflow_gnn.runner.tasks import classification
from tensorflow_gnn.runner.tasks import dgi
from tensorflow_gnn.runner.tasks import regression
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import attribution
from tensorflow_gnn.runner.utils import model
from tensorflow_gnn.runner.utils import model_dir
from tensorflow_gnn.runner.utils import model_export
from tensorflow_gnn.runner.utils import model_templates
from tensorflow_gnn.runner.utils import strategies

# Input
SimpleDatasetProvider = datasets.SimpleDatasetProvider
SimpleSampleDatasetsProvider = datasets.SimpleSampleDatasetsProvider

SampleTFRecordDatasetsProvider = datasets.SampleTFRecordDatasetsProvider
TFRecordDatasetProvider = datasets.TFRecordDatasetProvider

# Model directory
incrementing_model_dir = model_dir.incrementing_model_dir

# Model export
IntegratedGradientsExporter = attribution.IntegratedGradientsExporter
KerasModelExporter = model_export.KerasModelExporter
ModelExporter = orchestration.ModelExporter
SubmoduleExporter = model_export.SubmoduleExporter

# Model helpers
chain_first_output = model.chain_first_output

# Orchestration
DatasetProvider = orchestration.DatasetProvider
GraphTensorProcessorFn = orchestration.GraphTensorProcessorFn
run = orchestration.run
Trainer = orchestration.Trainer
Task = orchestration.Task

# Strategies
ParameterServerStrategy = strategies.ParameterServerStrategy

# Tasks
#
# Unsupervised
DeepGraphInfomax = dgi.DeepGraphInfomax
# Classification
RootNodeBinaryClassification = classification.RootNodeBinaryClassification
RootNodeMulticlassClassification = classification.RootNodeMulticlassClassification
GraphBinaryClassification = classification.GraphMulticlassClassification
GraphMulticlassClassification = classification.GraphMulticlassClassification
# Regression
GraphMeanAbsoluteError = regression.GraphMeanAbsoluteError
GraphMeanAbsolutePercentageError = regression.GraphMeanAbsolutePercentageError
GraphMeanSquaredError = regression.GraphMeanSquaredError
GraphMeanSquaredLogarithmicError = regression.GraphMeanSquaredLogarithmicError
GraphMeanSquaredLogScaledError = regression.GraphMeanSquaredLogScaledError
RootNodeMeanAbsoluteError = regression.RootNodeMeanAbsoluteError
RootNodeMeanAbsolutePercentageError = regression.RootNodeMeanAbsolutePercentageError
RootNodeMeanSquaredError = regression.RootNodeMeanSquaredError
RootNodeMeanSquaredLogarithmicError = regression.RootNodeMeanSquaredLogarithmicError
RootNodeMeanSquaredLogScaledError = regression.RootNodeMeanSquaredLogScaledError

# Training
KerasTrainer = keras_fit.KerasTrainer
KerasOptions = keras_fit.KerasOptions

# Model templates
ModelFromInitAndUpdates = model_templates.ModelFromInitAndUpdates

del orchestration
del datasets
del classification
del dgi
del regression
del keras_fit
del attribution
del model
del model_dir
del model_export
del model_templates
del strategies
