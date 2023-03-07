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
"""The tfgnn.keras package."""

from tensorflow_gnn.keras import builders
from tensorflow_gnn.keras import initializers
from tensorflow_gnn.keras import keras_tensors  # To register the types.
from tensorflow_gnn.keras import layers  # Provided as subpackage.

ConvGNNBuilder = builders.ConvGNNBuilder
clone_initializer = initializers.clone_initializer

# Prune imported module symbols so they're not accessible implicitly,
# except those meant to be used as subpackages, like tfgnn.keras.layers.
# Please use the same order as for the import statements at the top.
del builders
del initializers
del keras_tensors
