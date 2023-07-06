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
"""Binary to export OGBN datasets to disk as .npz format."""

import os

from absl import app
from absl import flags

from tensorflow_gnn.experimental.in_memory import datasets

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', os.path.expanduser('~/data/tfgnn/ogbn'),
    'Directory where dataset .npz file will be saved')
_DATASET_NAME = flags.DEFINE_string(
    'dataset', 'ogbn-mag', 'Name of OGBN dataset.')


def main(unused_argv) -> None:
  dataset_name = _DATASET_NAME.value
  out_file = os.path.join(_OUTPUT_DIR.value, dataset_name + '.npz')
  datasets.get_in_memory_graph_data(dataset_name).save(out_file)
  print('wrote ' + out_file)


if __name__ == '__main__':
  app.run(main)
