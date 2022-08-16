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
"""Write tfexample protos from the OGB dataset to Parquet or tfrecords.

These are helper functions for the convert_ogb_dataset python script.
"""
import math
from typing import Iterator, List, Tuple

import numpy
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf

Example = tf.train.Example
Array = numpy.ndarray

DataTable = List[Tuple[str, Array]]


def generate_examples(features: DataTable,
                      indices: Tuple[int, int]) -> Iterator[Example]:
  """Zip a set of features and yield example protos from it."""
  start_index, end_index = indices
  for i in range(start_index, end_index):
    example = Example()
    for name, array in features:
      feat = example.features.feature[name]
      if array.dtype in (numpy.int64, numpy.int32, int):
        value = feat.int64_list.value
      elif array.dtype in (numpy.float32, numpy.float64):
        value = feat.float_list.value
      elif array.dtype.type in (numpy.bytes_,):
        value = feat.bytes_list.value
      elif array.dtype.type in (numpy.string_, numpy.str_):
        value = [word.encode("utf-8") for word in feat.bytes_list.value]
      else:
        raise NotImplementedError(
            "Invalid type for {}: {}".format(name, array.dtype))
      feature = array[i]
      if feature.shape == ():  # pylint: disable=g-explicit-bool-comparison
        value.append(feature)
      else:
        value.extend(feature.flat)
    yield example


def write_tfrecords(filenames: List[str],
                    features: DataTable,
                    num_items: int):
  """Fill up Example protos with each node feature and output to tfrecords."""

  examples = iter(generate_examples(features, (0, num_items)))
  if len(filenames) > 1:
    # Write to multiple shards.
    per_file = int(math.ceil(num_items / len(filenames)))
    for filename in filenames:
      with tf.io.TFRecordWriter(filename) as file_writer:
        for _ in range(per_file):
          try:
            example = next(examples)
            file_writer.write(example.SerializeToString())
          except StopIteration:
            break
  else:
    # Write to a single shard.
    assert len(filenames) == 1
    with tf.io.TFRecordWriter(filenames[0]) as file_writer:
      for example in examples:
        file_writer.write(example.SerializeToString())


def write_parquet(filenames: List[str],
                  features: DataTable,
                  unused_num_items: int):
  """Fill up Example protos with each node feature and output to a parquet file."""
  assert len(filenames) == 1
  feature_names = [name for name, _ in features]
  feature_arrays = [array for _, array in features]
  # Convert arrays to Arrow tensors if needed.
  feature_arrays = [
      pa.Tensor.from_numpy(array) if isinstance(array[0], Array) else array
      for array in feature_arrays]
  table = pa.Table.from_arrays(feature_arrays, names=feature_names)
  pq.write_table(table, filenames[0])
