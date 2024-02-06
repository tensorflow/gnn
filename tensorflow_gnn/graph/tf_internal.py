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

import os

##
## Part 1: TensorFlow symbols
##

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

CompositeTensor = composite_tensor.CompositeTensor
BatchableTypeSpec = type_spec.BatchableTypeSpec
type_spec_register = (
    type_spec_registry.register if type_spec_registry is not None
    else type_spec.register)

type_spec_get_name = (
    type_spec_registry.get_name if type_spec_registry is not None
    else type_spec.get_name)

type_spec_lookup = (
    type_spec_registry.lookup if type_spec_registry is not None
    else type_spec.lookup)

OpDispatcher = tf.__internal__.dispatch.OpDispatcher


##
## Part 2: Keras symbols, compatible with `tf.keras.*`
##

# pytype: disable=import-error

if tf.__version__.startswith("2.12."):
  # tf.keras is keras 2.12, which does not yet have the `src` subdirectory.
  from keras import backend as keras_backend
  from keras.engine import input_layer
  from keras.engine import keras_tensor as kt
  from keras.layers import core as core_layers
  # In 2.12, these symbols are not exposed yet under tf.keras.__internal__.
  KerasTensor = kt.KerasTensor
  RaggedKerasTensor = kt.RaggedKerasTensor

elif tf.__version__.startswith("2.13.") or tf.__version__.startswith("2.14."):
  KerasTensor = tf.keras.__internal__.KerasTensor
  RaggedKerasTensor = tf.keras.__internal__.RaggedKerasTensor
  # tf.keras is keras.
  # For TF 2.14, there also exists a tf_keras package, but TF does not use it.
  from keras.src import backend as keras_backend
  from keras.src.engine import input_layer
  from keras.src.engine import keras_tensor as kt
  from keras.src.layers import core as core_layers

elif tf.__version__.startswith("2.15."):
  KerasTensor = tf.keras.__internal__.KerasTensor
  RaggedKerasTensor = tf.keras.__internal__.RaggedKerasTensor
  # OSS TensorFlow 2.15 can choose between keras 2.15 and tf_keras 2.15 but
  # THESE ARE DIFFERENT PACKAGES WITH SEPARATE GLOBAL REGISTRIES (b/324019542)
  # so it is essential that we pick the right one by replicating the logic from
  # https://github.com/tensorflow/tensorflow/blob/r2.15/tensorflow/python/util/lazy_loader.py#L96
  if os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"):
    from tf_keras.src import backend as keras_backend
    from tf_keras.src.layers import core as core_layers
    from tf_keras.src.engine import input_layer
    from tf_keras.src.engine import keras_tensor as kt
  else:
    from keras.src import backend as keras_backend
    from keras.src.layers import core as core_layers
    from keras.src.engine import input_layer
    from keras.src.engine import keras_tensor as kt

elif hasattr(tf, "_keras_internal"):  # Special case: internal.
  KerasTensor = tf.keras.__internal__.KerasTensor
  RaggedKerasTensor = tf.keras.__internal__.RaggedKerasTensor
  kt = tf._keras_internal.engine.keras_tensor  # pylint: disable=protected-access
  core_layers = tf._keras_internal.layers.core  # pylint: disable=protected-access
  input_layer = tf._keras_internal.engine.input_layer  # pylint:disable=protected-access
  keras_backend = tf._keras_internal.backend  # pylint: disable=protected-access

else:  # TF2.16 and onwards.
  # ../__init__.py has already checked that tf.keras has version 2, not 3,
  # which implies that tf.keras is tf_keras, and we do not second-guess
  # the selection logic.
  KerasTensor = tf.keras.__internal__.KerasTensor
  RaggedKerasTensor = tf.keras.__internal__.RaggedKerasTensor
  from tf_keras.src import backend as keras_backend
  from tf_keras.src.layers import core as core_layers
  from tf_keras.src.engine import input_layer
  from tf_keras.src.engine import keras_tensor as kt

# pytype: enable=import-error

register_keras_tensor_specialization = kt.register_keras_tensor_specialization
delegate_property = core_layers._delegate_property  # pylint: disable=protected-access
delegate_method = core_layers._delegate_method  # pylint: disable=protected-access
unique_keras_object_name = keras_backend.unique_object_name

# tensorflow_gnn/experimental/sampler/eval_dag.py uses the module object itself.
keras_input_layer_module = input_layer

# Delete imports, in their order above.
del composite_tensor
del type_spec
del tf
del type_spec_registry
del keras_backend
del core_layers
del kt
