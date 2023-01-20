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
"""Transformer-style multi-head attention.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import multi_head_attention
```
"""

from tensorflow_gnn.models.multi_head_attention import config_dict
from tensorflow_gnn.models.multi_head_attention import layers

MultiHeadAttentionConv = layers.MultiHeadAttentionConv
MultiHeadAttentionEdgePool = layers.MultiHeadAttentionEdgePool
MultiHeadAttentionHomGraphUpdate = layers.MultiHeadAttentionHomGraphUpdate
MultiHeadAttentionMPNNGraphUpdate = layers.MultiHeadAttentionMPNNGraphUpdate
graph_update_get_config_dict = config_dict.graph_update_get_config_dict
graph_update_from_config_dict = config_dict.graph_update_from_config_dict

# Prune imported module symbols so they're not accessible implicitly.
del config_dict
del layers
