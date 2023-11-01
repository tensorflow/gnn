# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Contrastive losses.

Users of TF-GNN can use these layers by importing them next to the core library:

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import contrastive_losses
```
"""
from tensorflow_gnn.models.contrastive_losses import layers
from tensorflow_gnn.models.contrastive_losses import metrics
from tensorflow_gnn.models.contrastive_losses import tasks
from tensorflow_gnn.utils import api_utils

# NOTE: This package is covered by tensorflow_gnn/api_def/api_symbols_test.py.
# Please see there for instructions how to reflect API changes.
# LINT.IfChange

CorruptionSpec = layers.CorruptionSpec
Corruptor = layers.Corruptor
DeepGraphInfomaxLogits = layers.DeepGraphInfomaxLogits
DropoutFeatures = layers.DropoutFeatures
ShuffleFeaturesGlobally = layers.ShuffleFeaturesGlobally
TripletEmbeddingSquaredDistances = layers.TripletEmbeddingSquaredDistances

BarlowTwinsTask = tasks.BarlowTwinsTask
ContrastiveLossTask = tasks.ContrastiveLossTask
DeepGraphInfomaxTask = tasks.DeepGraphInfomaxTask
TripletLossTask = tasks.TripletLossTask
VicRegTask = tasks.VicRegTask

AllSvdMetrics = metrics.AllSvdMetrics
coherence = metrics.coherence
numerical_rank = metrics.numerical_rank
pseudo_condition_number = metrics.pseudo_condition_number
rankme = metrics.rankme
self_clustering = metrics.self_clustering


# Remove all names added by module imports, unless explicitly allowed here.
api_utils.remove_submodules_except(__name__, [])
# LINT.ThenChange(../../api_def/contrastive_losses-symbols.txt)
