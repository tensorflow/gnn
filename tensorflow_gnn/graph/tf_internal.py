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
"""A central place to manage TF-GNN's dependencies on non-public TF APIs.

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

CompositeTensor = composite_tensor.CompositeTensor
BatchableTypeSpec = type_spec.BatchableTypeSpec
type_spec_register = (
    type_spec_registry.register if type_spec_registry is not None
    else type_spec.register)

del composite_tensor
del type_spec
del type_spec_registry
