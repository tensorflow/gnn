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
"""Experimental (unstable) parts of the public interface of TensorFlow GNN.

A symbol `foo` exposed here is available to library users as

```
import tensorflow_gnn as tfgnn

tfgnn.experimental.foo()
```

This is the preferred way to expose individual functions on track to inclusion
into the stable public interface of TensorFlow GNN.

Beyond these symbols, there are also experimental sub-libraries that
need to be imported separately (`from tensorflow_gnn.experimental import foo`).
That is for special cases only.
"""

from tensorflow_gnn.graph import readout
from tensorflow_gnn.graph import tensor_utils

context_readout_into_feature = readout.context_readout_into_feature
segment_random_index_shuffle = tensor_utils.segment_random_index_shuffle

del readout
del tensor_utils
