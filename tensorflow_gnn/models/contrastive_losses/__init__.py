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
from tensorflow_gnn.models.contrastive_losses import tasks

CorruptionSpec = layers.CorruptionSpec
DeepGraphInfomaxLogits = layers.DeepGraphInfomaxLogits
DropoutFeatures = layers.DropoutFeatures
ShuffleFeaturesGlobally = layers.ShuffleFeaturesGlobally

BarlowTwinsTask = tasks.BarlowTwinsTask
DeepGraphInfomaxTask = tasks.DeepGraphInfomaxTask
VicRegTask = tasks.VicRegTask

del layers
del tasks
