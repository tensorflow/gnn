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
"""Beam executor for bulk sampling pipelines.

Allows to run sampling model using Apache Beam. Each sampling model operates on
batches of independent examples. The typical sampling model takes batch of seed
nodes as its input and step-by-step adds new neighbors by calling sampling
layers. While executing on Beam, we take this to the extreme: there is a single
batch (as PCollection) containing all examples. Because `PCollection` is not
ordered set of entites, we key all entities by unique "example id". Those
strings are used instead of an batch dimension indices to unqiquely identify
individual examples. The sampling layers are translated into Beam stages (as
`PTransform`) and tensors are replaced with `PCollection`s containing fixed size
lists of "tensor values", as `[value_1, value_2, .., value_n]`, where `n` is
fixed within each `PCollection`. Each tensor value is  fixed-size list of
`numpy.ndarray`s containing flattened tensor or composite tensor components.
The following flattening rule is applied to any tensor value `t`:
  * dense: `[t]`;
  * ragged: `[t.flat_values, *t.nested_row_lengths()]`;
  * other:  `tf.nest.flatten(t)`;
with results tensor components being converted to the `numpy.ndarray`s.
"""

from typing import List, Tuple
import apache_beam as beam
from apache_beam.coders import typecoders
from apache_beam.typehints import typehints

import numpy as np

PCollection = beam.pvalue.PCollection
# Global unique identifier of a particular example.
ExampleId = bytes
# Tensor or flattened composite tensor.
Value = List[np.ndarray]
# Collection of values belonging to particular example.
Values = List[Value]
# Stage input/output as a batch of example values keyed by unique example ids.
PValues = PCollection[Tuple[ExampleId, Values]]


class NDArrayCoder(beam.coders.Coder):
  """Beam coder for Numpy N-dimensional array of TF-compatible data types.

  Supports all numeric data types and bytes (represented as `np.object_`).
  The numpy array is serialized as a tuple of `(dtype, shape, flat values)`.
  For numeric values serialization we rely on `tobytes()` and `frombuffer` from
  the numpy library. It, seems, has the best speed/space tradeoffs. Tensorflow
  represents `tf.string` as `np.object_` (as `np.string_` is for arrays
  containing fixed-width byte strings, which can lead to lots of wasted
  memory). Because `np.object_` is an array of references to arbitrary
  objects, we could not rely on numpy native serialization and using
  `IterableCoder` from the Beam library instead.

  NOTE: for some simple stages the execution time may be dominated by data
  serialization/deserialization, so any imporvement here translates directly to
  the total execution costs.
  """

  def __init__(self):
    encoded_struct = typehints.Tuple[str, typehints.Tuple[int, ...], bytes]
    self._coder = typecoders.registry.get_coder(encoded_struct)
    self._bytes_coder = typecoders.registry.get_coder(typehints.Iterable[bytes])

  def encode(self, value: np.ndarray) -> bytes:
    if value.dtype == np.object_:
      flat_values = self._bytes_coder.encode(value.flat)
    else:
      flat_values = value.tobytes()
    return self._coder.encode((value.dtype.str, value.shape, flat_values))

  def decode(self, encoded: bytes) -> np.ndarray:
    dtype_str, shape, serialized_values = self._coder.decode(encoded)
    dtype = np.dtype(dtype_str)
    if dtype == np.object_:
      flat_values = np.array(
          self._bytes_coder.decode(serialized_values), dtype=np.object_
      )
    else:
      flat_values = np.frombuffer(serialized_values, dtype=dtype)
    return np.reshape(flat_values, shape)

  def is_deterministic(self):
    return True

  def to_type_hint(self):
    return np.ndarray


beam.coders.registry.register_coder(np.ndarray, NDArrayCoder)
