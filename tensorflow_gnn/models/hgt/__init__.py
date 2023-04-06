# Copyright 2022 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Heterogeneous Graph Transformers.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import hgt
```
"""
from tensorflow_gnn.models.hgt import config_dict
from tensorflow_gnn.models.hgt import layers

HGTGraphUpdate = layers.HGTGraphUpdate
graph_update_get_config_dict = config_dict.graph_update_get_config_dict
graph_update_from_config_dict = config_dict.graph_update_from_config_dict

del config_dict
del layers
