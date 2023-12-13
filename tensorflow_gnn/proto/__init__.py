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
"""The protocol message (protobuf) types defined by TensorFlow GNN.

This package is automatically included in the top-level tfgnn package:

```
import tensorflow_gnn as tfgnn
graph_schema = tfgnn.proto.GraphSchema()
```

Users are also free to import it separately as

```
import tensorflow_gnn.proto as tfgnn_proto
graph_schema = tfgnn_proto.GraphSchema()
```

...which, together with using its more targeted BUILD dependency,
can help to shrink the bazel-bin/**/*.runfiles/ directory.
"""

from tensorflow_gnn.proto import graph_schema
from tensorflow_gnn.utils import api_utils

# NOTE: This package is covered by tensorflow_gnn/api_def/api_symbols_test.py.
# Please see there for instructions how to reflect API changes.
# LINT.IfChange

# Same order as in the proto file.
GraphSchema = graph_schema.GraphSchema
Feature = graph_schema.Feature
BigQuery = graph_schema.BigQuery
Metadata = graph_schema.Metadata
Context = graph_schema.Context
NodeSet = graph_schema.NodeSet
EdgeSet = graph_schema.EdgeSet
GraphType = graph_schema.GraphType
OriginInfo = graph_schema.OriginInfo

# Remove all names added by module imports, unless explicitly allowed here.
api_utils.remove_submodules_except(
    __name__,
    [
        # Workaround for Beam/pickle, required by
        # `experimental/sampler/beam/sampler.py`.
        # TODO(b/316135889): remove once fixed.
        'graph_schema_pb2',
    ],
)
# LINT.ThenChange()../api_def/tfgnn-symbols.txt)
