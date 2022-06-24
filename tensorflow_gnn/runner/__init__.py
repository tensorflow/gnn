"""A general purpose runner for TF-GNN."""
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.input import datasets
from tensorflow_gnn.runner.tasks import attribution
from tensorflow_gnn.runner.tasks import classification
from tensorflow_gnn.runner.tasks import dgi
from tensorflow_gnn.runner.tasks import regression
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import model_dir
from tensorflow_gnn.runner.utils import model_export
from tensorflow_gnn.runner.utils import model_templates
from tensorflow_gnn.runner.utils import strategies

# Input
SimpleDatasetProvider = datasets.SimpleDatasetProvider
SimpleSampleDatasetsProvider = datasets.SimpleSampleDatasetsProvider

TFRecordDatasetProvider = datasets.TFRecordDatasetProvider
SampleTFRecordDatasetsProvider = datasets.SampleTFRecordDatasetsProvider

# Model directory
incrementing_model_dir = model_dir.incrementing_model_dir

# Model export
KerasModelExporter = model_export.KerasModelExporter
SubmoduleExporter = model_export.SubmoduleExporter
ModelExporter = orchestration.ModelExporter

# Orchestration
DatasetProvider = orchestration.DatasetProvider
GraphTensorProcessorFn = orchestration.GraphTensorProcessorFn
run = orchestration.run
Trainer = orchestration.Trainer
Task = orchestration.Task

# Strategies
ParameterServerStrategy = strategies.ParameterServerStrategy

# Tasks
IntegratedGradients = attribution.IntegratedGradients
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
del attribution
del classification
del dgi
del regression
del keras_fit
del model_dir
del model_templates
del strategies
