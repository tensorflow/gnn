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
"""GraphSAGE.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import graph_sage
```
"""

from tensorflow_gnn.models.graph_sage import layers

GCNGraphSAGENodeSetUpdate = layers.GCNGraphSAGENodeSetUpdate
GraphSAGEAggregatorConv = layers.GraphSAGEAggregatorConv
GraphSAGEPoolingConv = layers.GraphSAGEPoolingConv
GraphSAGENextState = layers.GraphSAGENextState
GraphSAGEGraphUpdate = layers.GraphSAGEGraphUpdate

# Prune imported module symbols so they're not accessible implicitly.
del layers
