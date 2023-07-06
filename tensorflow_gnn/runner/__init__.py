# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A general purpose runner for TF-GNN."""
from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.input import datasets
from tensorflow_gnn.runner.tasks import classification
from tensorflow_gnn.runner.tasks import link_prediction
from tensorflow_gnn.runner.tasks import regression
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import attribution
from tensorflow_gnn.runner.utils import label_fns
from tensorflow_gnn.runner.utils import model_dir
from tensorflow_gnn.runner.utils import model_export
from tensorflow_gnn.runner.utils import padding as padding_utils
from tensorflow_gnn.runner.utils import strategies

# Attribution
integrated_gradients = attribution.integrated_gradients

# Input
PassthruDatasetProvider = datasets.PassthruDatasetProvider
PassthruSampleDatasetsProvider = datasets.PassthruSampleDatasetsProvider
SimpleDatasetProvider = datasets.SimpleDatasetProvider
SimpleSampleDatasetsProvider = datasets.SimpleSampleDatasetsProvider
SampleTFRecordDatasetsProvider = datasets.SampleTFRecordDatasetsProvider
TFRecordDatasetProvider = datasets.TFRecordDatasetProvider

# Interfaces
DatasetProvider = interfaces.DatasetProvider
GraphTensorPadding = interfaces.GraphTensorPadding
GraphTensorProcessorFn = interfaces.GraphTensorProcessorFn
ModelExporter = interfaces.ModelExporter
Trainer = interfaces.Trainer
Task = interfaces.Task
RunResult = interfaces.RunResult

# Label fns
ContextLabelFn = label_fns.ContextLabelFn
RootNodeLabelFn = label_fns.RootNodeLabelFn

# Model directory
incrementing_model_dir = model_dir.incrementing_model_dir

# Model export
IntegratedGradientsExporter = attribution.IntegratedGradientsExporter
KerasModelExporter = model_export.KerasModelExporter
SubmoduleExporter = model_export.SubmoduleExporter

# Orchestration
run = orchestration.run
TFDataServiceConfig = orchestration.TFDataServiceConfig

# Padding
one_node_per_component = padding_utils.one_node_per_component
FitOrSkipPadding = padding_utils.FitOrSkipPadding
TightPadding = padding_utils.TightPadding

# Strategies
ParameterServerStrategy = strategies.ParameterServerStrategy
TPUStrategy = strategies.TPUStrategy

# NOTE: Tasks cross TensorFlow distribute strategies are tested end to end in
# `distribute_test.py`. If adding a new Task, please also add it to the test
# combinations found there. (See `_all_task_and_processors_combinations`
# in `distribute_test.py`.)
#
# Tasks (Classification)
RootNodeBinaryClassification = classification.RootNodeBinaryClassification
RootNodeMulticlassClassification = classification.RootNodeMulticlassClassification
GraphBinaryClassification = classification.GraphMulticlassClassification
GraphMulticlassClassification = classification.GraphMulticlassClassification
# Tasks (Link Prediction)
DotProductLinkPrediction = link_prediction.DotProductLinkPrediction
HadamardProductLinkPrediction = link_prediction.HadamardProductLinkPrediction
# Tasks (Regression)
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
RootNodeMeanAbsoluteLogarithmicError = (
    regression.RootNodeMeanAbsoluteLogarithmicError
)

# Training
KerasTrainer = keras_fit.KerasTrainer
KerasTrainerOptions = keras_fit.KerasTrainerOptions
KerasTrainerCheckpointOptions = keras_fit.KerasTrainerCheckpointOptions

del interfaces
del orchestration
del datasets
del classification
del link_prediction
del regression
del keras_fit
del attribution
del label_fns
del model_dir
del model_export
del padding_utils
del strategies
