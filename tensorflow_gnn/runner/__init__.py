"""A general purpose runner for TF-GNN."""
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.input import datasets
from tensorflow_gnn.runner.tasks import attribution
from tensorflow_gnn.runner.tasks import dgi
from tensorflow_gnn.runner.tasks import graph_classification
from tensorflow_gnn.runner.tasks import node_classification
from tensorflow_gnn.runner.tasks import regression
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import model_dir
from tensorflow_gnn.runner.utils import strategies

# Input
SimpleDatasetProvider = datasets.SimpleDatasetProvider
SimpleSampleDatasetsProvider = datasets.SimpleSampleDatasetsProvider

TFRecordDatasetProvider = datasets.TFRecordDatasetProvider
SampleTFRecordDatasetsProvider = datasets.SampleTFRecordDatasetsProvider

# Model directory
incrementing_model_dir = model_dir.incrementing_model_dir

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
RootNodeBinaryClassification = node_classification.RootNodeBinaryClassification
RootNodeMulticlassClassification = node_classification.RootNodeMulticlassClassification
GraphMulticlassClassification = graph_classification.GraphMulticlassClassification
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

del orchestration
del datasets
del attribution
del dgi
del graph_classification
del node_classification
del regression
del keras_fit
del model_dir
del strategies
