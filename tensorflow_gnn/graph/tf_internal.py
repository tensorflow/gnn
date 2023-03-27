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
"""A central place to manage TF-GNN's dependencies on non-public TF&Keras APIs.

TODO(b/188399175): Use the public ExtensionType API instead.
"""

# The following imports work in all supported versions of TF.
# pylint: disable=g-direct-tensorflow-import,g-import-not-at-top,g-bad-import-order
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec

# The remaining imports vary by TF version, so they are not covered by an
# explicit BUILD dep. (See `tags=["ignore_for_dep=...", ...]`.)
# Instead, this file depends on TensorFlow as a whole
import tensorflow as tf  # pylint: disable=unused-import

try:
  from tensorflow.python.framework import type_spec_registry
except ImportError:
  type_spec_registry = None  # Not available before TF 2.12.
try:
  from keras.engine import keras_tensor
except ImportError:
  # Path as seen in pip packages as of TF/Keras 2.13.
  from keras.src.engine import keras_tensor  # pytype: disable=import-error
try:
  from keras.layers import core as core_layers
except ImportError:
  # Path as seen in pip packages as of TF/Keras 2.13.
  from keras.src.layers import core as core_layers  # pytype: disable=import-error

CompositeTensor = composite_tensor.CompositeTensor
BatchableTypeSpec = type_spec.BatchableTypeSpec
type_spec_register = (
    type_spec_registry.register if type_spec_registry is not None
    else type_spec.register)

try:
  # These types are semi-public as of TF/Keras 2.13.
  # Whenever possible, get them the official way.
  KerasTensor = tf.keras.__internal__.KerasTensor
  RaggedKerasTensor = tf.keras.__internal__.RaggedKerasTensor
except AttributeError:
  KerasTensor = keras_tensor.KerasTensor
  RaggedKerasTensor = keras_tensor.RaggedKerasTensor
# These KerasTensor helpers are still private in TF/Keras 2.13.
register_keras_tensor_specialization = (
    keras_tensor.register_keras_tensor_specialization)
delegate_property = core_layers._delegate_property  # pylint: disable=protected-access
delegate_method = core_layers._delegate_method  # pylint: disable=protected-access
# TFClassMethodDispatcher = core_layers.TFClassMethodDispatcher

# Delete imports, in their order above.
del composite_tensor
del type_spec
del tf
del type_spec_registry
del keras_tensor
del core_layers
